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

# ──────────────────────────────────────────────────────────────────────────────
# Prompt
# ──────────────────────────────────────────────────────────────────────────────
def get_ocr_prompt(source_language: str, target_language: str) -> str:
    return (
        f"Detect every visible text region in the image (assuming it is mostly {source_language}) "
        f"and translate the text into {target_language}. "
        "Do not summarize the image. "
        "Do not describe the image. "
        "Do not omit small text. "
        "Do not omit headings, labels, buttons, menus, captions, or text near the edges. "
        "Return every detected text region with its exact pixel bounding box. "
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
        config.max_new_tokens = 4096
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

                valid.append(
                    {
                        "text": text.strip(),
                        "translated_text": translated_text.strip(),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }
                )

            except Exception as exc:
                logger.warning("OCR | skipping detection #%d — %s | raw=%s", idx, exc, det)

        return valid
