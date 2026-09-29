"""
services/ocr_service.py

Responsibilities:
  - Accept a PIL image (or numpy array).
  - Send it to the loaded Qwen2.5-VL-7B OpenVINO model.
  - Parse the JSON response.
  - Validate and clip every detected bounding-box.
  - Return a list of validated OCR detections.

The model instance is injected at startup (loaded once in main.py lifespan).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

def get_ocr_prompt(source_language: str, target_language: str) -> str:
    return (
        f"Detect every visible text region in the image (assuming it is mostly {source_language}) "
        f"and translate the text into {target_language}. "
        "Rules: "
        "1. One entry per visual line or UI element — NEVER split a single line/word into fragments. "
        "2. Merge fragments belonging to the same line into one entry. "
        "3. Tight pixel bounding box [x1, y1, x2, y2] hugging the exact glyphs, "
        "in ORIGINAL image pixel coordinates (not 0-1000 normalized). "
        "4. Reading order top-to-bottom, left-to-right. "
        "5. Exact original text + natural translation (keep code/commands like "
        "'netstat -ano | findstr :8000' untranslated if a translation makes no sense). "
        "Do not summarize or describe the image. Do not omit small text. "
        "Return ONLY a JSON array. No explanation. No markdown. "
        "Use this exact format:\n"
        '[{"text": "...", "translated_text": "...", "bbox": [x1, y1, x2, y2]}, ...]'
    )


# ──────────────────────────────────────────────────────────────────────────────
# OCRService
# ──────────────────────────────────────────────────────────────────────────────
class OCRService:
    """Wraps the Qwen2.5-VL OpenVINO model to perform OCR on images using openvino_genai."""

    def __init__(self, pipe):
        """
        Args:
            pipe: Loaded openvino_genai.VLMPipeline.
        """
        self.pipe = pipe

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def detect(self, image: Image.Image, source_language: str, target_language: str) -> list[dict[str, Any]]:
        """
        Run OCR and translation simultaneously on *image* and return validated detections.

        Returns:
            List of dicts: {"text": str, "bbox": [x1, y1, x2, y2]}
            Coordinates are in original image pixel space.
        """
        orig_w, orig_h = image.size
        logger.info("OCR | original size: %dx%d", orig_w, orig_h)

        t0 = time.perf_counter()
        raw_response = self._run_inference(image, source_language, target_language)
        t1 = time.perf_counter()
        logger.info("OCR | inference took %.2fs", t1 - t0)
        
        print("\n" + "-" * 50)
        print("DEBUG: [OCR] Raw Model Output:")
        print(raw_response)
        print("-" * 50 + "\n")
        
        logger.debug("OCR | raw model response: %s", raw_response)

        detections = self._parse_response(raw_response)
        detections = self._validate_detections(detections, orig_w, orig_h)

        logger.info("OCR | %d valid detections", len(detections))
        return detections

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────
    def _run_inference(self, image: Image.Image, source_language: str, target_language: str) -> str:
        """Send image + prompt to the model and return the raw text output."""
        import openvino as ov
        import openvino_genai

        img_tensor = ov.Tensor(np.array(image))

        config = openvino_genai.GenerationConfig()
        config.max_new_tokens = 2048
        config.do_sample = False

        prompt = get_ocr_prompt(source_language, target_language)

        print("\n" + "-" * 50)
        print("DEBUG: [OCR+Translate] Generating text with prompt:")
        print(prompt)
        print("-" * 50 + "\n")

        res = self.pipe.generate(
            prompt,
            images=[img_tensor],
            generation_config=config,
        )

        return res.texts[0].strip() if hasattr(res, "texts") and res.texts else str(res).strip()

    # ─────────────────────────────────────────────────────────────────────────
    # Parsing
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_response(raw: str) -> list[dict[str, Any]]:
        """
        Extract a JSON array from the model's raw output.
        Handles cases where the model wraps the JSON in markdown code fences
        or adds leading/trailing text.
        """
        # 1. Try direct parse
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown fences and retry
        stripped = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # 3. Find the first [...] block
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        logger.warning("OCR | could not parse model response as JSON; returning []")
        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────
    MIN_BOX_W = 8
    MIN_BOX_H = 8
    NMS_IOU_THRESHOLD = 0.3

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _filter_and_merge(cls, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop tiny boxes, merge overlapping fragments (e.g. 'Ici:' + 'ere:').

        Sorts by area (largest first), greedily unions boxes with IoU >
        NMS_IOU_THRESHOLD. Merged text is concatenated with a space so split
        words reassemble instead of double-drawing on top of each other.
        """
        # 2. Filter tiny noise boxes (UI speckles, icons)
        kept = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            if (x2 - x1) < cls.MIN_BOX_W or (y2 - y1) < cls.MIN_BOX_H:
                logger.warning("OCR | dropping tiny box %s text='%.30s'", det["bbox"], det["text"])
                continue
            kept.append(det)
        # Sort largest-first so fragments merge INTO the dominant box
        kept.sort(key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]), reverse=True)
        merged: list[dict[str, Any]] = []
        for det in kept:
            placed = False
            for m in merged:
                if cls._iou(det["bbox"], m["bbox"]) > cls.NMS_IOU_THRESHOLD:
                    # Union bbox
                    mx1, my1, mx2, my2 = m["bbox"]
                    x1, y1, x2, y2 = det["bbox"]
                    m["bbox"] = [min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2)]
                    # Concatenate split fragments (avoid exact duplicates)
                    if det["text"] not in m["text"]:
                        m["text"] = (m["text"] + " " + det["text"]).strip()
                    if det.get("translated_text") and det["translated_text"] not in (m.get("translated_text") or ""):
                        m["translated_text"] = ((m.get("translated_text") or "") + " " + det["translated_text"]).strip()
                    placed = True
                    break
            if not placed:
                merged.append(det)
        # Reading order: top-to-bottom, then left-to-right (stable render)
        merged.sort(key=lambda d: (d["bbox"][1] // 10, d["bbox"][0]))
        if len(merged) != len(detections):
            logger.info("OCR | NMS: %d -> %d boxes (tiny/overlap removed)", len(detections), len(merged))
        return merged

    @staticmethod
    def _validate_detections(
        detections: list[dict[str, Any]],
        img_w: int,
        img_h: int,
    ) -> list[dict[str, Any]]:
        """
        Validate each detection:
          - text must be a non-empty string
          - bbox must have exactly 4 numeric values
          - x1 < x2, y1 < y2
          - coordinates are clipped to image boundaries

        Invalid detections are skipped (logged as warnings) without crashing.
        """
        valid: list[dict[str, Any]] = []

        for idx, det in enumerate(detections):
            try:
                text = det.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("empty or non-string text")

                bbox_raw = det.get("bbox")
                if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                    raise ValueError(f"bbox must have 4 values, got: {bbox_raw}")

                x1, y1, x2, y2 = [float(v) for v in bbox_raw]

                if x1 >= x2:
                    raise ValueError(f"x1 ({x1}) >= x2 ({x2})")
                if y1 >= y2:
                    raise ValueError(f"y1 ({y1}) >= y2 ({y2})")

                # Clip to image boundaries
                x1 = max(0.0, min(x1, img_w - 1))
                y1 = max(0.0, min(y1, img_h - 1))
                x2 = max(0.0, min(x2, img_w))
                y2 = max(0.0, min(y2, img_h))

                if x1 >= x2 or y1 >= y2:
                    raise ValueError("bbox collapsed to zero area after clipping")

                translated_text = det.get("translated_text", "")
                if not isinstance(translated_text, str):
                    translated_text = ""

                # NOTE: font/background style is NO LONGER predicted by Qwen.
                # It is filled in later by services/font_style_service.py
                # (local OpenCV analysis, zero model cost). Keep keys present
                # with None defaults so downstream code has a stable schema.
                # Backward-compat: if an old model output still contains style
                # fields, preserve them; otherwise leave None for enrichment.
                def _get_rgb_optional(key):
                    val = det.get(key)
                    if isinstance(val, list) and len(val) == 3:
                        try:
                            return tuple(int(v) for v in val)
                        except (ValueError, TypeError):
                            pass
                    return None

                _font_rgb = _get_rgb_optional("font_color_rgb")
                _bg_rgb = _get_rgb_optional("background_color_rgb")
                _fsize = det.get("font_size")
                if not isinstance(_fsize, (int, float)):
                    _fsize = None

                valid.append(
                    {
                        "text": text.strip(),
                        "translated_text": translated_text.strip(),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "font_color_rgb": _font_rgb,
                        "font_color_hex": det.get("font_color_hex"),
                        "font_size": int(_fsize) if _fsize else None,
                        "font_weight": det.get("font_weight", None),
                        "font_style": det.get("font_style", None),
                        "font_family": det.get("font_family", None),
                        "background_color_rgb": _bg_rgb,
                        "background_color_hex": det.get("background_color_hex"),
                    }
                )

            except Exception as exc:
                logger.warning("OCR | skipping detection #%d — %s | raw=%s", idx, exc, det)

        return OCRService._filter_and_merge(valid)
