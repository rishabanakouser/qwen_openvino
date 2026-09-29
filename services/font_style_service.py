"""
services/font_style_service.py

Local, model-free font analysis to replace Qwen-predicted style fields.

Why this exists:
  - Asking Qwen2.5-VL for font_color / font_size / weight / background
    roughly doubles output tokens -> slower decode + laptop freeze.
  - Qwen now returns only {text, translated_text, bbox}.
  - This service fills {font_color_rgb, background_color_rgb,
    font_size, font_weight, ...} with OpenCV + Pillow in ~ms per box.

Method (per bbox, no DL model):
  1. Crop bbox from the ORIGINAL PIL image.
  2. background = median of border pixels (robust to text inside).
  3. Otsu threshold on grayscale crop -> text-vs-background mask.
     Text cluster = minority cluster; disambiguated by distance to
     border-median background color.
  4. font_color = mean RGB of text-mask pixels.
  5. font_size ~= bbox height * 0.8 (single-line heuristic),
     clamped to config.FONT_SIZE_[MIN|MAX].
  6. font_weight = bold if text-pixel density high, else normal.
     (stroke-heavy text fills more of its box)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, int(rgb[0]))),
        max(0, min(255, int(rgb[1]))),
        max(0, min(255, int(rgb[2]))),
    )


class FontStyleService:
    """
    Enrich Qwen detections with locally-estimated style.

    Usage:
        svc = FontStyleService()
        detections = svc.enrich(pil_image, detections)
    """

    def __init__(
        self,
        font_size_min: int = config.FONT_SIZE_MIN,
        font_size_max: int = config.FONT_SIZE_MAX,
        bold_density_threshold: float = 0.35,
    ):
        self.font_size_min = font_size_min
        self.font_size_max = font_size_max
        self.bold_density_threshold = bold_density_threshold

    # ── Public API ────────────────────────────────────────────────
    def enrich(
        self, image: Image.Image, detections: list[dict]
    ) -> list[dict]:
        """Fill missing style keys in-place; always returns same list."""
        if not detections:
            return detections
        img_np = np.array(image.convert("RGB"))  # HxWx3, RGB
        h, w = img_np.shape[:2]
        for det in detections:
            try:
                self._enrich_one(img_np, w, h, det)
            except Exception as exc:
                logger.warning(
                    "FontStyle | failed for bbox %s: %s — using defaults",
                    det.get("bbox"), exc,
                )
                self._apply_defaults(det)
        return detections

    # ── Per-box ───────────────────────────────────────────────────
    def _enrich_one(self, img_np: np.ndarray, img_w: int, img_h: int, det: dict) -> None:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        if x2 - x1 < 3 or y2 - y1 < 3:
            self._apply_defaults(det)
            return

        crop = img_np[y1:y2, x1:x2]  # RGB
        ch, cw = crop.shape[:2]

        # 1. Background from border pixels (top/bottom rows + left/right cols)
        border = np.concatenate([
            crop[0:1, :, :].reshape(-1, 3),
            crop[-1:, :, :].reshape(-1, 3),
            crop[:, 0:1, :].reshape(-1, 3),
            crop[:, -1:, :].reshape(-1, 3),
        ], axis=0)
        bg_rgb = tuple(int(v) for v in np.median(border, axis=0))

        # 2. Otsu split -> two clusters
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_mask = binary == 255
        black_mask = ~white_mask
        n_white = int(np.count_nonzero(white_mask))
        n_black = int(np.count_nonzero(black_mask))

        if n_white == 0 or n_black == 0:
            # Flat box — fall back to border bg + contrast text color
            font_rgb = self._contrast_color(bg_rgb)
            text_density = 0.0
        else:
            mean_white = tuple(float(v) for v in crop[white_mask].mean(axis=0))
            mean_black = tuple(float(v) for v in crop[black_mask].mean(axis=0))
            # Which cluster matches the border background?
            d_white = sum((a - b) ** 2 for a, b in zip(mean_white, bg_rgb))
            d_black = sum((a - b) ** 2 for a, b in zip(mean_black, bg_rgb))
            if d_white < d_black:
                bg_est, font_pixels = mean_white, crop[black_mask]
                text_density = n_black / (ch * cw)
            else:
                bg_est, font_pixels = mean_black, crop[white_mask]
                text_density = n_white / (ch * cw)
            bg_rgb = tuple(int(round(v)) for v in bg_est)
            # Use median + dark/bright core instead of plain mean so
            # anti-aliased edge pixels don't wash the color to gray.
            # e.g. black text on white gave (167,167,167) with mean.
            bg_luma = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
            med = np.median(font_pixels.reshape(-1, 3), axis=0)
            med_luma = 0.299 * med[0] + 0.587 * med[1] + 0.114 * med[2]
            if med_luma < bg_luma:
                # Dark text: take darker core (20th percentile)
                core = np.percentile(font_pixels.reshape(-1, 3), 20, axis=0)
            else:
                # Light text: take brighter core (80th percentile)
                core = np.percentile(font_pixels.reshape(-1, 3), 80, axis=0)
            # Blend median + core for stability
            font_rgb = tuple(int(round((m + c) / 2)) for m, c in zip(med, core))

        # 3. Font size heuristic: box height ~= ascender+descender.
        # Pillow truetype size roughly maps to that height.
        box_h = y2 - y1
        est_size = int(round(box_h * 0.8))
        est_size = max(self.font_size_min, min(self.font_size_max, est_size))

        # 4. Weight heuristic: dense/stroke-heavy text -> bold.
        weight = "bold" if text_density > self.bold_density_threshold else "normal"

        # Only fill keys Qwen left empty (None) — never overwrite model values
        # if backward-compat old outputs still carry them.
        det["font_color_rgb"] = det.get("font_color_rgb") or font_rgb
        det["font_color_hex"] = det.get("font_color_hex") or _rgb_to_hex(det["font_color_rgb"])
        det["background_color_rgb"] = det.get("background_color_rgb") or bg_rgb
        det["background_color_hex"] = det.get("background_color_hex") or _rgb_to_hex(bg_rgb)
        det["font_size"] = det.get("font_size") or est_size
        det["font_weight"] = det.get("font_weight") or weight
        det["font_style"] = det.get("font_style") or "normal"
        det["font_family"] = det.get("font_family") or "sans-serif"

    # ── Helpers ───────────────────────────────────────────────────
    def _apply_defaults(self, det: dict) -> None:
        box_h = det["bbox"][3] - det["bbox"][1] if det.get("bbox") else 24
        det["font_color_rgb"] = det.get("font_color_rgb") or (0, 0, 0)
        det["font_color_hex"] = det.get("font_color_hex") or "#000000"
        det["background_color_rgb"] = det.get("background_color_rgb")
        det["background_color_hex"] = det.get("background_color_hex")
        det["font_size"] = det.get("font_size") or max(
            self.font_size_min, min(self.font_size_max, int(box_h * 0.8) or 24)
        )
        det["font_weight"] = det.get("font_weight") or "normal"
        det["font_style"] = det.get("font_style") or "normal"
        det["font_family"] = det.get("font_family") or "sans-serif"

    @staticmethod
    def _contrast_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
        # ITU-R BT.601 luma — pick black or white text for readability
        luma = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        return (0, 0, 0) if luma > 128 else (255, 255, 255)
