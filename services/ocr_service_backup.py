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
OCR_PROMPT = (
    "Detect every visible text region in the image. "
    "Do not summarize the image. "
    "Do not describe the image. "
    "Do not omit small text. "
    "Do not omit headings, labels, buttons, menus, captions, or text near the edges. "
    "Return every detected text region with its exact pixel bounding box. "
    "Return ONLY a JSON array. No explanation. No markdown. "
    "Use this exact format:\n"
    '[{"text": "...", "bbox": [x1, y1, x2, y2]}, ...]'
)


# ──────────────────────────────────────────────────────────────────────────────
# OCRService
# ──────────────────────────────────────────────────────────────────────────────
class OCRService:
    """Wraps the Qwen2.5-VL OpenVINO model to perform OCR on images."""

    def __init__(self, model, processor):
        """
        Args:
            model:     Loaded OVModelForVisualCausalLM (or equivalent).
            processor: Corresponding AutoProcessor / Qwen2VLProcessor.
        """
        self.model = model
        self.processor = processor

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        """
        Run OCR on *image* and return validated detections.

        Returns:
            List of dicts: {"text": str, "bbox": [x1, y1, x2, y2]}
            Coordinates are in original image pixel space.
        """
        orig_w, orig_h = image.size
        logger.info("OCR | original size: %dx%d", orig_w, orig_h)

        t0 = time.perf_counter()
        raw_response = self._run_inference(image)
        t1 = time.perf_counter()
        logger.info("OCR | inference took %.2fs", t1 - t0)
        logger.debug("OCR | raw model response: %s", raw_response)

        detections = self._parse_response(raw_response)
        detections = self._validate_detections(detections, orig_w, orig_h)

        logger.info("OCR | %d valid detections", len(detections))
        return detections

    # ─────────────────────────────────────────────────────────────────────────
    # Inference
    # ─────────────────────────────────────────────────────────────────────────
    def _run_inference(self, image: Image.Image) -> str:
        """Send image + prompt to the model and return the raw text output."""
        from qwen_vl_utils import process_vision_info  # type: ignore

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ]

        # Build prompt text
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Build image inputs
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )

        # Decode only the newly generated tokens
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()

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

                valid.append(
                    {
                        "text": text.strip(),
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    }
                )

            except Exception as exc:
                logger.warning("OCR | skipping detection #%d — %s | raw=%s", idx, exc, det)

        return valid
