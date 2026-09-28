"""
services/text_renderer.py

TextRenderer draws translated text inside the original bounding-box
coordinates.  It:
  - Automatically calculates the largest font size that fits.
  - Wraps long text to multiple lines.
  - Centers the text (both horizontally and vertically) inside the bbox.
  - Never lets text overflow outside the bbox.
  - Uses a configurable TrueType font (defaults to DejaVuSans).
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

import config

logger = logging.getLogger(__name__)

# Fallback when no TTF font is available
_PILLOW_DEFAULT_FONT = ImageFont.load_default()


class TextRenderer:
    """
    Renders translated text inside bounding boxes on a PIL image.

    Parameters
    ----------
    font_path : str | None
        Path to a .ttf file.  Falls back to Pillow's built-in font if not found.
    font_size_max : int
        Largest font size to try (reduced until text fits).
    font_size_min : int
        Smallest allowable font size before giving up.
    """

    def __init__(
        self,
        font_path: Optional[str] = config.FONT_PATH,
        font_size_max: int = config.FONT_SIZE_MAX,
        font_size_min: int = config.FONT_SIZE_MIN,
    ):
        self.font_path = font_path if font_path and Path(font_path).exists() else None
        self.font_size_max = font_size_max
        self.font_size_min = font_size_min

        if not self.font_path:
            logger.warning(
                "TextRenderer | font not found at '%s'. Using Pillow default font.",
                font_path,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def render_text_in_bbox(
        self,
        image: Image.Image,
        translated_text: str,
        bbox: list[int],
        text_color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """
        Draw *translated_text* centred inside *bbox* on *image*.

        Args:
            image:           PIL RGB image to draw on (modified in-place, copy returned).
            translated_text: String to render.
            bbox:            [x1, y1, x2, y2] in original image pixel coords.
            text_color:      RGB tuple for the text colour.

        Returns:
            New PIL image with text rendered.
        """
        image = image.copy()

        if not translated_text.strip():
            return image

        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            logger.warning("TextRenderer | degenerate bbox %s, skipping", bbox)
            return image

        draw = ImageDraw.Draw(image)

        # Find the best (font, wrapped_lines) pair that fits inside the box
        font, lines = self._fit_text(translated_text, box_w, box_h)

        # Compute total text block height
        line_height = self._line_height(font)
        total_h = line_height * len(lines)

        # Vertical start (centre block inside bbox)
        start_y = y1 + (box_h - total_h) // 2

        for i, line in enumerate(lines):
            line_w = self._text_width(font, line)
            # Horizontal centre
            start_x = x1 + (box_w - line_w) // 2
            draw.text(
                (start_x, start_y + i * line_height),
                line,
                font=font,
                fill=text_color,
            )

        return image

    def render_all(
        self,
        image: Image.Image,
        detections: list[dict],
        text_color: tuple[int, int, int] = (0, 0, 0),
    ) -> Image.Image:
        """
        Render translated text for every detection on the image.

        Each detection must have: {"bbox": [...], "translated_text": "..."}.
        """
        for det in detections:
            try:
                translated = det.get("translated_text", "")
                bbox = det["bbox"]
                image = self.render_text_in_bbox(image, translated, bbox, text_color)
            except Exception as exc:
                logger.warning(
                    "TextRenderer | failed for det %s: %s", det, exc
                )
        return image

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, size=size)
            except Exception:
                pass
        # Pillow ≥ 10: load_default accepts size kwarg
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return _PILLOW_DEFAULT_FONT

    @staticmethod
    def _text_width(font, text: str) -> int:
        try:
            bbox = font.getbbox(text)  # Pillow ≥ 9.2
            return bbox[2] - bbox[0]
        except AttributeError:
            return font.getlength(text)  # type: ignore

    @staticmethod
    def _line_height(font) -> int:
        try:
            ascent, descent = font.getmetrics()
            return ascent + descent
        except AttributeError:
            return 16  # safe fallback

    def _wrap_text(self, text: str, font, max_width: int) -> list[str]:
        """Wrap *text* so each line fits within *max_width* pixels."""
        words = text.split()
        if not words:
            return [text]

        lines: list[str] = []
        current_line = ""

        for word in words:
            candidate = (current_line + " " + word).strip() if current_line else word
            if self._text_width(font, candidate) <= max_width:
                current_line = candidate
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word

        if current_line:
            lines.append(current_line)

        return lines or [text]

    def _fit_text(
        self,
        text: str,
        box_w: int,
        box_h: int,
    ) -> tuple[ImageFont.FreeTypeFont, list[str]]:
        """
        Binary-search for the largest font size where the wrapped text
        block fits entirely inside (box_w × box_h).

        Returns (font, list_of_lines).
        """
        best_font = self._load_font(self.font_size_min)
        best_lines = [text]

        for size in range(self.font_size_max, self.font_size_min - 1, -1):
            font = self._load_font(size)
            lines = self._wrap_text(text, font, box_w)
            line_h = self._line_height(font)
            total_h = line_h * len(lines)
            max_line_w = max(self._text_width(font, ln) for ln in lines)

            if max_line_w <= box_w and total_h <= box_h:
                best_font = font
                best_lines = lines
                break  # largest fitting size found

        return best_font, best_lines
