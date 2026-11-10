from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

import config

logger = logging.getLogger("screen_analyzer_test")
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1024
MODEL_REQUIRED_FILES = ("config.json", "openvino_language_model.xml")

ANALYSIS_PROMPT = """You are an IT support expert diagnosing a computer issue from a screenshot.
Inspect the screenshot carefully. Identify visible applications, error messages,
warnings, and relevant system state. Distinguish what is directly visible from
what is only a possible explanation. Do not invent an error or claim the system
crashed unless the screenshot supports that conclusion.

Return a concise, actionable diagnosis using exactly these sections:
### 1. Issue Observed
Describe what is visible, or state that no clear issue is visible.

### 2. Root Cause
Explain the most likely cause based on the visible evidence and note uncertainty.

### 3. Resolution Steps
Give numbered, practical troubleshooting steps. If no issue is visible, say what
additional information would help diagnose one."""


class AnalysisResponse(BaseModel):
    analysis: str
    model: str
    image_width: int
    image_height: int


app_state: dict[str, Any] = {"pipeline": None}
inference_lock = threading.Lock()


def format_analysis(raw_text: str) -> str:
    text = raw_text.strip()
    if not text:
        raise RuntimeError("The vision model returned an empty analysis.")
    if all(
        heading in text
        for heading in (
            "### 1. Issue Observed",
            "### 2. Root Cause",
            "### 3. Resolution Steps",
        )
    ):
        return text

    return (
        "### 1. Issue Observed\n"
        f"{text}\n\n"
        "### 2. Root Cause\n"
        "The model did not provide a separate root-cause section.\n\n"
        "### 3. Resolution Steps\n"
        "1. Review the visible error or application state described above.\n"
        "2. Collect any related error details or logs before making changes."
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    if config.OCR_BACKEND != "local":
        raise RuntimeError("The test analyzer requires OCR_BACKEND=local to use the downloaded Qwen3 model.")

    model_path = Path(config.MODEL_PATH)
    logger.info(
        "Starting Qwen3 Screen Analyzer | backend=%s | device=%s | model_path=%s",
        config.OCR_BACKEND,
        config.OPENVINO_DEVICE,
        model_path,
    )
    missing_files = [name for name in MODEL_REQUIRED_FILES if not (model_path / name).is_file()]
    if missing_files:
        raise RuntimeError(
            f"Qwen3 model is missing from {model_path} ({', '.join(missing_files)}). "
            "Run ensure_model.py from the Image Translation directory first."
        )

    try:
        import openvino_genai

        logger.info("Loading Qwen3 model from %s on %s", model_path, config.OPENVINO_DEVICE)
        app_state["pipeline"] = openvino_genai.VLMPipeline(
            str(model_path),
            config.OPENVINO_DEVICE,
        )
        logger.info("Qwen3 screen analyzer is ready.")
    except Exception as exc:
        logger.exception("Could not load the Qwen3 screen analyzer.")
        raise RuntimeError(f"Could not load Qwen3 model: {exc}") from exc

    yield
    app_state["pipeline"] = None


app = FastAPI(
    title="Qwen3 Screen Analyzer Test",
    description="Upload a screenshot and diagnose it with the local Qwen3-VL model.",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
def upload_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen3 Screen Analyzer Test</title>
  <style>
    body { font: 16px system-ui, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px; background: #10131b; color: #eef0f6; }
    main { background: #1a1f2b; padding: 24px; border-radius: 12px; }
    button { padding: 10px 16px; margin-top: 12px; cursor: pointer; }
    #status { color: #b7c0d0; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #11151e; padding: 18px; border-radius: 8px; line-height: 1.55; }
    img { display: none; max-width: 100%; max-height: 420px; margin-top: 16px; border-radius: 8px; }
  </style>
</head>
<body>
  <main>
    <h1>Qwen3 Screen Analyzer</h1>
    <p>Upload a screenshot to get an IT-support diagnosis from the local Qwen3 model.</p>
    <form id="form">
      <input id="image" name="image" type="file" accept="image/*" required>
      <br><button id="submit" type="submit">Analyze screenshot</button>
    </form>
    <p id="status" role="status">Checking Qwen3 service…</p>
    <img id="preview" alt="Uploaded screenshot preview">
    <pre id="result" aria-live="polite"></pre>
    <h2>Debug log</h2>
    <pre id="debug" aria-live="polite"></pre>
  </main>
  <script>
    const form = document.getElementById("form");
    const input = document.getElementById("image");
    const button = document.getElementById("submit");
    const status = document.getElementById("status");
    const result = document.getElementById("result");
    const preview = document.getElementById("preview");
    const debug = document.getElementById("debug");
    const log = (message) => {
      const time = new Date().toLocaleTimeString();
      debug.textContent += `[${time}] ${message}\\n`;
      debug.scrollTop = debug.scrollHeight;
    };
    fetch("/health")
      .then(async (response) => {
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || `HTTP ${response.status}`);
        status.textContent = `Ready · ${body.model} · ${body.device}`;
        log(`Service ready: ${JSON.stringify(body)}`);
      })
      .catch((error) => {
        status.textContent = `Could not check service: ${error.message}`;
        log(`Health check failed: ${error.message}`);
      });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!input.files.length) return;
      const file = input.files[0];
      const data = new FormData();
      data.append("image", file);
      button.disabled = true;
      status.textContent = "Uploading screenshot…";
      result.textContent = "";
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
      log(`Selected ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB, ${file.type || "unknown type"}).`);
      log("Sending screenshot to /analyze-frame.");
      const startedAt = Date.now();
      const timer = setInterval(() => {
        const seconds = Math.floor((Date.now() - startedAt) / 1000);
        status.textContent = seconds < 10
          ? `Analyzing with Qwen3… ${seconds}s`
          : `Qwen3 is still analyzing… ${seconds}s. First inference can take a while.`;
      }, 1000);
      const healthPoll = setInterval(() => {
        fetch("/health")
          .then((response) => response.json())
          .then((body) => log(`Service status: ${body.status}, device: ${body.device}.`))
          .catch((error) => log(`Health poll failed: ${error.message}`));
      }, 15000);
      try {
        const response = await fetch("/analyze-frame", { method: "POST", body: data });
        log(`Server responded with HTTP ${response.status}.`);
        const body = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
        result.textContent = body.analysis;
        const seconds = ((Date.now() - startedAt) / 1000).toFixed(1);
        status.textContent = `Analysis complete in ${seconds}s · ${body.model} · ${body.image_width} × ${body.image_height}`;
        log(`Analysis complete in ${seconds}s.`);
      } catch (error) {
        status.textContent = `Analysis failed: ${error.message || "Unknown error"}`;
        log(`Request failed: ${error.message || "Unknown error"}`);
      } finally {
        clearInterval(timer);
        clearInterval(healthPoll);
        button.disabled = false;
      }
    });
  </script>
</body>
</html>"""


@app.get("/health")
def health() -> dict[str, str]:
    ready = app_state["pipeline"] is not None
    logger.info("Health check: %s", "ready" if ready else "loading")
    return {
        "status": "ok" if ready else "loading",
        "model": config.MODEL_NAME,
        "device": config.OPENVINO_DEVICE,
    }


@app.post("/analyze-frame", response_model=AnalysisResponse)
def analyze_frame(image: UploadFile = File(...)) -> AnalysisResponse:
    pipeline = app_state["pipeline"]
    if pipeline is None:
        logger.error("Analysis request rejected: Qwen3 pipeline is not ready.")
        raise HTTPException(status_code=503, detail="Qwen3 model is not ready.")

    image_bytes = image.file.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        logger.warning("Analysis request rejected: empty upload (filename=%s).", image.filename)
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        logger.warning("Analysis request rejected: upload exceeds 20 MB (filename=%s).", image.filename)
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB upload limit.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            screenshot = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Analysis request rejected: invalid image (filename=%s).", image.filename)
        raise HTTPException(status_code=422, detail="Uploaded file is not a valid image.") from exc

    original_width, original_height = screenshot.size
    logger.info(
        "Received screenshot | filename=%s | bytes=%d | dimensions=%dx%d",
        image.filename,
        len(image_bytes),
        original_width,
        original_height,
    )
    largest_dimension = max(screenshot.size)
    if largest_dimension > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / largest_dimension
        screenshot = screenshot.resize(
            (
                max(1, round(original_width * scale)),
                max(1, round(original_height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        logger.info("Resized screenshot for inference to %dx%d.", *screenshot.size)

    try:
        import openvino_genai

        generation_config = openvino_genai.GenerationConfig()
        generation_config.max_new_tokens = 600
        generation_config.do_sample = False

        logger.info("Starting Qwen3 inference.")
        inference_started = time.perf_counter()
        with inference_lock:
            result = pipeline.generate(
                ANALYSIS_PROMPT,
                images=[ov.Tensor(np.asarray(screenshot))],
                generation_config=generation_config,
            )
        logger.info("Qwen3 inference completed in %.2fs.", time.perf_counter() - inference_started)
        raw_text = result.texts[0].strip() if result.texts else str(result).strip()
        analysis = format_analysis(raw_text)
        logger.info("Returning analysis response (%d characters).", len(analysis))
    except Exception as exc:
        logger.exception("Qwen3 screen analysis failed.")
        raise HTTPException(status_code=500, detail=f"Qwen3 analysis failed: {exc}") from exc
    finally:
        image.file.close()

    return AnalysisResponse(
        analysis=analysis,
        model=config.MODEL_NAME,
        image_width=original_width,
        image_height=original_height,
    )
