"""
services/ocr_service.py

OCR + translation via a Qwen-VL OpenVINO model (single pass). Parses the
JSON response, scales normalized bboxes to pixels, validates every
detection. The model instance is injected at startup (loaded once in
main.py lifespan).
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
        "3. Bounding box as NORMALIZED 0-1000 coordinates [x1, y1, x2, y2] "
        "(0,0 = top-left, 1000,1000 = bottom-right), hugging the exact glyphs. "
        "Do NOT guess pixels — the image may be resized internally. "
        "4. Reading order top-to-bottom, left-to-right. "
        "5. Exact original text + natural translation (keep code/commands like "
        "'netstat -ano | findstr :8000' untranslated if a translation makes no sense). "
        "Do not summarize or describe the image. Do not omit small text. "
        "Return ONLY a JSON array. No explanation. No markdown. "
        "Use this exact format:\n"
        '[{"text": "...", "translated_text": "...", "bbox": [x1, y1, x2, y2]}, ...]'
    )


# OCRService
class OCRService:
    """Wraps a Qwen-VL OpenVINO model to perform OCR on images using openvino_genai."""

    def __init__(self, pipe):
        """
        Args:
            pipe: Loaded openvino_genai.VLMPipeline.
        """
        self.pipe = pipe

    # Public API
    def detect(self, image: Image.Image, source_language: str, target_language: str) -> list[dict[str, Any]]:
        """
        Run OCR and translation simultaneously on *image* and return validated detections.

        Returns:
            List of dicts: {"text", "translated_text", "bbox": [x1, y1, x2, y2]}
            Coordinates are in original image pixel space (normalized model
            output is scaled here — see _validate_detections).
        """
        import config as _config

        orig_w, orig_h = image.size
        logger.info("OCR | original size: %dx%d", orig_w, orig_h)

        t0 = time.perf_counter()
        raw_response = self._run_inference(image, source_language, target_language)
        t1 = time.perf_counter()
        logger.info("OCR | inference took %.2fs", t1 - t0)
        logger.debug("OCR | raw model response: %s", raw_response)

        detections = self._parse_response(raw_response)
        detections = self._validate_detections(
            detections, orig_w, orig_h,
            coord_mode=getattr(_config, "OCR_COORD_MODE", "auto"),
        )

        logger.info("OCR | %d valid detections", len(detections))
        return detections

    # Inference
    def _run_inference(self, image: Image.Image, source_language: str, target_language: str) -> str:
        """Send image + prompt to the model and return the raw text output."""
        import openvino as ov
        import openvino_genai

        img_tensor = ov.Tensor(np.array(image))

        config = openvino_genai.GenerationConfig()
        config.max_new_tokens = 2048
        config.do_sample = False

        prompt = get_ocr_prompt(source_language, target_language)

        res = self.pipe.generate(
            prompt,
            images=[img_tensor],
            generation_config=config,
        )

        return res.texts[0].strip() if hasattr(res, "texts") and res.texts else str(res).strip()

    # Parsing
    @staticmethod
    def _try_json_list(s: str) -> list[dict[str, Any]] | None:
        """Return parsed list or None (never raises)."""
        try:
            data = json.loads(s)
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_response(raw: str) -> list[dict[str, Any]]:
        """
        Extract a JSON array from the model's raw output.

        Stages (first hit wins):
          1. Direct parse.
          2. Strip markdown fences and retry.
          3. First '[' … last ']' slice (drops leading/trailing prose even
             when the prose itself contains brackets — the old greedy regex
             over-matched and died on exactly that).
          4. Common repairs: trailing commas, raw control chars (strict=False).
          5. Per-object fallback: parse each {...} individually so ONE corrupt
             entry can't nuke 26 good detections (this incident).

        Every failure logs msg + position + snippet instead of swallowing it.
        """
        hit = OCRService._try_json_list(raw)
        if hit is not None:
            return hit

        stripped = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        if stripped != raw:
            hit = OCRService._try_json_list(stripped)
            if hit is not None:
                return hit

        start, end = raw.find("["), raw.rfind("]")
        block = raw[start:end + 1] if 0 <= start < end else ""
        if block and block != raw.strip():
            hit = OCRService._try_json_list(block)
            if hit is not None:
                logger.info("OCR | recovered JSON via bracket slice (%d chars)", len(block))
                return hit
        elif block:
            pass  # block == raw, already tried; fall through to repairs
        else:
            logger.warning("OCR | no [...] block found; raw len=%d head=%.200r", len(raw), raw[:200])

        candidate = block or raw
        repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
        if repaired != candidate:
            hit = OCRService._try_json_list(repaired)
            if hit is not None:
                logger.info("OCR | recovered JSON after trailing-comma repair")
                return hit
        try:
            data = json.loads(candidate, strict=False)
            if isinstance(data, list):
                logger.info("OCR | recovered JSON with strict=False")
                return data
        except json.JSONDecodeError as exc:
            logger.warning("OCR | strict=False also failed: %s at %d snippet=%.120r",
                           exc.msg, exc.pos, candidate[max(0, exc.pos - 60):exc.pos + 60])

        salvaged: list[dict[str, Any]] = []
        for m in re.finditer(r"\{[^{}]*\}", candidate):
            obj_str = m.group()
            obj = None
            try:
                obj = json.loads(obj_str)
            except json.JSONDecodeError:
                try:
                    import ast
                    obj = ast.literal_eval(obj_str)  # tolerates single quotes
                except (ValueError, SyntaxError):
                    continue
            if isinstance(obj, dict) and obj.get("text") and obj.get("bbox"):
                salvaged.append(obj)
        if salvaged:
            logger.warning("OCR | salvaged %d/%d objects individually (full-array parse failed)",
                           len(salvaged), candidate.count('"text"'))
            return salvaged

        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            logger.warning("OCR | could not parse model response: %s at char %d of %d; head=%.200r tail=%.200r",
                           exc.msg, exc.pos, len(candidate), candidate[:200], candidate[-200:])
        else:
            logger.warning("OCR | response parsed but was not a list; head=%.200r", candidate[:200])
        return []

    # Validation
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
        coord_mode: str = "auto",
    ) -> list[dict[str, Any]]:
        """
        Validate each detection:
          - text must be a non-empty string
          - bbox must have exactly 4 numeric values
          - x1 < x2, y1 < y2
          - coordinates are scaled (if 0-1000 normalized) then clipped to image

        Coordinate handling: Qwen's vision preprocessor smart-resizes the
        image internally, so asked-for "original pixels" come back in an
        unknown resized space (boxes land small/shifted). The prompt now asks
        for 0-1000 normalized coords, scaled here deterministically.
          - "normalized": always scale x*W/1000, y*H/1000
          - "pixels": use as-is (legacy)
          - "auto": normalized iff EVERY bbox value in the response is <=1000

        Invalid detections are skipped (logged as warnings) without crashing.
        """
        mode = (coord_mode or "auto").lower()
        if mode == "auto":
            try:
                all_vals = [float(v) for det in detections
                            for v in (det.get("bbox") or [])]
                mode = "normalized" if all_vals and max(all_vals) <= 1000 else "pixels"
            except (ValueError, TypeError):
                mode = "pixels"
        logger.info("OCR | coord mode: %s (image %dx%d)", mode, img_w, img_h)
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

                if mode == "normalized":
                    x1, x2 = x1 * img_w / 1000.0, x2 * img_w / 1000.0
                    y1, y2 = y1 * img_h / 1000.0, y2 * img_h / 1000.0

                if x1 >= x2:
                    raise ValueError(f"x1 ({x1}) >= x2 ({x2})")
                if y1 >= y2:
                    raise ValueError(f"y1 ({y1}) >= y2 ({y2})")

                x1 = max(0.0, min(x1, img_w - 1))
                y1 = max(0.0, min(y1, img_h - 1))
                x2 = max(0.0, min(x2, img_w))
                y2 = max(0.0, min(y2, img_h))

                if x1 >= x2 or y1 >= y2:
                    raise ValueError("bbox collapsed to zero area after clipping")

                translated_text = det.get("translated_text", "")
                if not isinstance(translated_text, str):
                    translated_text = ""

                # Style fields come from FontStyleService, not Qwen — None here
                # means "fill in later"; preserved when already present.
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
