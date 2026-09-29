"""
main.py - FastAPI application for OCR + Translation using Qwen2.5-VL OpenVINO.

Architecture:
  startup  → load OpenVINO model once
  
  GET  /outputs/{filename} → serve translated image
  GET  /health  → liveness check
  GET  /info    → detailed runtime info
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import openvino as ov
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps
import io

import config
from services.ocr_service import OCRService
from services.font_style_service import FontStyleService
from services.inpaint_service import InpaintService
from services.text_renderer import TextRenderer

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ──────────────────────────────────────────────────────────────────────────────
# Application-level state (populated at startup)
# ──────────────────────────────────────────────────────────────────────────────
app_state: dict[str, Any] = {
    "ocr_service": None,
    "font_style_service": None,
    "inpaint_service": None,
    "text_renderer": None,
    "device": config.OPENVINO_DEVICE,
    "available_devices": [],
    "openvino_version": ov.__version__,
}


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan: model loaded ONCE at startup, released at shutdown
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Discover OpenVINO devices ──────────────────────────────────────────
    core = ov.Core()
    available_devices = core.available_devices
    app_state["available_devices"] = available_devices

    print("\n" + "=" * 60)
    print("Available OpenVINO devices:")
    for d in available_devices:
        print(f"  {d}")
    print(f"Selected device: {config.OPENVINO_DEVICE}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────
    model_path = Path(config.MODEL_PATH)
    if not model_path.exists():
        logger.error(
            "Model directory not found: %s\n"
            "Run the model download script first. See models/README.md.",
            model_path,
        )
        raise RuntimeError(f"Model not found at {model_path}")

    print(f"Loading model from: {model_path}")
    print("This may take a few minutes on first run (model compilation)...")

    try:
        import openvino_genai

        pipe = openvino_genai.VLMPipeline(
            str(model_path),
            config.OPENVINO_DEVICE,
        )

        print("Model loaded successfully.")
        print("=" * 60 + "\n")

    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)
        raise RuntimeError(f"Model load failure: {exc}") from exc

    # ── Initialise services ───────────────────────────────────────────────
    app_state["ocr_service"] = OCRService(pipe=pipe)
    app_state["font_style_service"] = FontStyleService()
    app_state["inpaint_service"] = InpaintService()
    app_state["text_renderer"] = TextRenderer()

    yield  # ← application runs here

    # ── Shutdown (nothing special needed; GC handles model) ───────────────
    logger.info("Shutting down — releasing resources.")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Qwen2.5-VL OCR Translation Service",
    description=(
        "Backend-only image OCR and translation service using "
        "Qwen2.5-VL-7B-Instruct via OpenVINO."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: validate services are ready
# ──────────────────────────────────────────────────────────────────────────────
def _require_services():
    if app_state["ocr_service"] is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialised — model may still be loading.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# POST /translate
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/translate")
async def translate(
    image: UploadFile = File(..., description="Input image file"),
    source_language: str = Form(..., description="Language of the original text (e.g. English)"),
    target_language: str = Form(..., description="Language to translate into (e.g. French)"),
):
    """
    Full pipeline:
      1. Read uploaded image
      2. OCR with Qwen2.5-VL → list of {text, bbox}
      3. Translate each text
      4. Inpaint original text regions
      5. Render translated text inside original bboxes
      6. Save output image
      7. Return JSON with detections + image URL
    """
    _require_services()

    total_start = time.perf_counter()

    # ── 1. Validate & read image ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEBUG: [PIPELINE START] Received translation request")
    print(f"DEBUG: Source Language: {source_language}")
    print(f"DEBUG: Target Language: {target_language}")
    print(f"DEBUG: Image Filename: {image.filename}")
    print("=" * 60 + "\n")

    logger.info("Request received | source=%s | target=%s | file=%s",
                source_language, target_language, image.filename)

    if not source_language.strip():
        raise HTTPException(status_code=422, detail="source_language is required")
    if not target_language.strip():
        raise HTTPException(status_code=422, detail="target_language is required")

    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty")

    try:
        pil_image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Cannot open image: {exc}"
        ) from exc

    orig_w, orig_h = pil_image.size
    logger.info("Image dimensions: %dx%d", orig_w, orig_h)
    logger.info("Source language: %s | Target language: %s | Device: %s",
                source_language, target_language, config.OPENVINO_DEVICE)

    # ── 2. OCR & Translation (Single Pass) ─────────────────────────────────
    print("DEBUG: [STEP 1 & 2] Starting OCR + Translation via Qwen2.5-VL...")
    ocr_start = time.perf_counter()
    try:
        detections = app_state["ocr_service"].detect(pil_image, source_language, target_language)
    except Exception as exc:
        logger.exception("OCR failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"OCR error: {exc}") from exc
    ocr_time = time.perf_counter() - ocr_start

    print(f"DEBUG: [STEP 1 & 2 COMPLETE] OCR found {len(detections)} text regions.")
    for idx, det in enumerate(detections):
        print(f"       -> Region {idx+1}: {det['text']} at bbox {det['bbox']}")
        print(f"          Translated: '{det['translated_text']}'")
    
    logger.info("OCR+Translate detections: %d | time: %.2fs", len(detections), ocr_time)
    
    translation_time = 0.0 # Bypassed because it's bundled in OCR

    # ── 3. Local font-style analysis (no model, OpenCV only) ─────────────
    print("\nDEBUG: [STEP 2b] Estimating font color/size locally (no model)...")
    style_start = time.perf_counter()
    try:
        style_svc: FontStyleService = app_state["font_style_service"]
        detections = style_svc.enrich(pil_image, detections)
    except Exception as exc:
        logger.exception("Font-style analysis failed: %s", exc)
    style_time = time.perf_counter() - style_start
    logger.info("Font-style analysis time: %.3fs", style_time)

    # ── 4. Inpainting ─────────────────────────────────────────────────────
    print("\nDEBUG: [STEP 3] Starting Image Inpainting (removing original text)...")
    inpaint_start = time.perf_counter()
    try:
        inpaint_svc: InpaintService = app_state["inpaint_service"]
        inpainted_image = inpaint_svc.remove_text(pil_image, detections)
    except Exception as exc:
        logger.exception("Inpainting failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Inpainting error: {exc}"
        ) from exc
    inpaint_time = time.perf_counter() - inpaint_start
    logger.info("Inpainting time: %.2fs", inpaint_time)

    # ── 5. Render translated text ─────────────────────────────────────────
    print("\nDEBUG: [STEP 4] Starting Text Rendering (drawing translated text)...")
    render_start = time.perf_counter()
    try:
        renderer: TextRenderer = app_state["text_renderer"]
        final_image = renderer.render_all(inpainted_image, detections)
    except Exception as exc:
        logger.exception("Text rendering failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Render error: {exc}"
        ) from exc
    render_time = time.perf_counter() - render_start
    logger.info("Rendering time: %.2fs", render_time)

    # ── 5b. Enforce output dimensions == input dimensions ────────────────
    # Inpaint (numpy/cv2 round-trip) and renderer (.copy()+draw) preserve
    # size, but guard against any accidental resize so output pixels match input.
    if final_image.size != (orig_w, orig_h):
        logger.warning(
            "Dimension mismatch: input=%dx%d output=%dx%d — resizing back to input size",
            orig_w, orig_h, final_image.size[0], final_image.size[1],
        )
        final_image = final_image.resize((orig_w, orig_h), Image.LANCZOS)
    logger.info("Output dimensions: %dx%d (input was %dx%d)",
                final_image.size[0], final_image.size[1], orig_w, orig_h)

    # ── 6. Save output image ──────────────────────────────────────────────
    print("\nDEBUG: [STEP 5] Saving final output image...")
    output_filename = f"{uuid.uuid4().hex}.png"
    output_path = config.OUTPUTS_DIR / output_filename
    try:
        final_image.save(str(output_path), format="PNG")
        print(f"DEBUG: [PIPELINE COMPLETE] Saved to {output_path}")
        print("=" * 60 + "\n")
    except Exception as exc:
        logger.exception("Failed to save output image: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Failed to save output image: {exc}"
        ) from exc

    total_time = time.perf_counter() - total_start

    # ── 7. Logging summary ────────────────────────────────────────────────
    logger.info(
        "SUMMARY | detections=%d | ocr=%.2fs | translation=%.2fs | "
        "inpaint=%.2fs | render=%.2fs | total=%.2fs",
        len(detections),
        ocr_time,
        translation_time,
        inpaint_time,
        render_time,
        total_time,
    )

    # ── 8. Response ───────────────────────────────────────────────────────
    response_detections = [
        {
            "text": d["text"],
            "translated_text": d.get("translated_text", ""),
            "bbox": d["bbox"],
        }
        for d in detections
    ]

    return JSONResponse(
        content={
            "success": True,
            "source_language": source_language,
            "target_language": target_language,
            "detections": response_detections,
            "image_url": f"/outputs/{output_filename}",
            "timing": {
                "ocr_seconds": round(ocr_time, 3),
                "translation_seconds": round(translation_time, 3),
                "inpaint_seconds": round(inpaint_time, 3),
                "render_seconds": round(render_time, 3),
                "total_seconds": round(total_time, 3),
            },
        }
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /outputs/{filename}
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/outputs/{filename}")
async def get_output(filename: str):
    """Serve a generated translated image by filename."""
    # Security: reject path traversal attempts
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = config.OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        path=str(file_path),
        media_type="image/png",
        filename=filename,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Liveness check — returns 200 when the service is ready."""
    ready = app_state["ocr_service"] is not None
    return JSONResponse(
        content={
            "status": "ok" if ready else "loading",
            "model": config.MODEL_NAME,
            "device": config.OPENVINO_DEVICE,
        },
        status_code=200 if ready else 503,
    )


# ──────────────────────────────────────────────────────────────────────────────
# GET /info
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/info")
async def info():
    """Detailed runtime information about the service."""
    return JSONResponse(
        content={
            "model": config.MODEL_NAME,
            "model_path": config.MODEL_PATH,
            "device": config.OPENVINO_DEVICE,
            "openvino_version": app_state["openvino_version"],
            "available_devices": app_state["available_devices"],
            "font_path": config.FONT_PATH,
            "inpaint_radius": config.INPAINT_RADIUS,
            "bbox_padding": config.BBOX_PADDING,
        }
    )
