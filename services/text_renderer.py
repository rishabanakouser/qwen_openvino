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

    # Public API
    def render_text_in_bbox(
        self,
        image: Image.Image,
        det: dict,
    ) -> Image.Image:
        """
        Draw translated text inside bbox on *image* using extracted font details.

        Args:
            image: PIL RGB image to draw on (modified in-place, copy returned).
            det:   Detection dictionary with text, bbox, and font details.

        Returns:
            New PIL image with text rendered.
        """
        image = image.copy()
        
        translated_text = det.get("translated_text", "")
        if not translated_text.strip():
            return image

        x1, y1, x2, y2 = det["bbox"]
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            logger.warning("TextRenderer | degenerate bbox %s, skipping", det["bbox"])
            return image

        draw = ImageDraw.Draw(image)

        est_size = det.get("font_size") or 24
        try:
            est_size = int(est_size)
        except (ValueError, TypeError):
            est_size = 24
        est_size = max(self.font_size_min, min(self.font_size_max, est_size))
        font_color = det.get("font_color_rgb") or (0, 0, 0)
        font_weight = det.get("font_weight") or "normal"
        font_style = det.get("font_style") or "normal"
        font_family = det.get("font_family") or "sans-serif"

        # Fit against expanded width (EN->FR grows ~20%), draw centered on
        # the ORIGINAL box so bleed is symmetric (~10% each side).
        fit_w = max(8, int(box_w * 1.2))
        fit_h = max(8, box_h)
        font, lines = self._fit_text(
            translated_text, fit_w, fit_h,
            start_size=est_size,
            family=font_family, weight=font_weight, style=font_style,
        )

        line_height = self._line_height(font)
        total_h = line_height * len(lines)

        start_y = y1 + (box_h - total_h) // 2

        for i, line in enumerate(lines):
            line_w = self._text_width(font, line)
            start_x = x1 + (box_w - line_w) // 2
            draw.text(
                (start_x, start_y + i * line_height),
                line,
                font=font,
                fill=font_color,
            )

        return image

    def render_all(
        self,
        image: Image.Image,
        detections: list[dict],
    ) -> Image.Image:
        """
        Render translated text for every detection on the image.
        """
        for det in detections:
            try:
                image = self.render_text_in_bbox(image, det)
            except Exception as exc:
                logger.warning(
                    "TextRenderer | failed for det %s: %s", det, exc
                )
        return image

    # Internal helpers
    def _load_font(self, size: int, family: str = "", weight: str = "", style: str = "") -> ImageFont.FreeTypeFont:
        is_bold = "bold" in weight.lower()
        is_italic = "italic" in style.lower()
        
        font_name = "arial"
        if "times" in family.lower(): font_name = "times"
        elif "courier" in family.lower(): font_name = "cour"
        elif "segoe" in family.lower(): font_name = "segoeui"

        suffix = ""
        if is_bold and is_italic:
            suffix = "bi" if font_name in ["arial", "times"] else "z"
        elif is_bold:
            suffix = "bd" if font_name in ["arial", "times", "cour"] else "b"
        elif is_italic:
            suffix = "i"
            
        system_font_path = Path(f"C:/Windows/Fonts/{font_name}{suffix}.ttf")
        
        if system_font_path.exists():
            try:
                return ImageFont.truetype(str(system_font_path), size=size)
            except Exception:
                pass

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
            bbox = font.getbbox(text)
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
        """Wrap *text* so each line fits within *max_width* pixels.

        Handles long unbroken tokens (code, URLs, commands) by hard-splitting
        them character-wise so `netstat -ano | findstr :8000` never overflows.
        """
        words = text.split()
        if not words:
            return [text]

        split_words: list[str] = []
        for word in words:
            if self._text_width(font, word) <= max_width:
                split_words.append(word)
                continue
            chunk = ""
            for ch in word:
                cand = chunk + ch
                if self._text_width(font, cand) <= max_width and chunk:
                    chunk = cand
                else:
                    if chunk:
                        split_words.append(chunk)
                    chunk = ch
                    # Single char wider than box (tiny box) — keep it anyway
                    if self._text_width(font, chunk) > max_width:
                        split_words.append(chunk)
                        chunk = ""
            if chunk:
                split_words.append(chunk)

        lines: list[str] = []
        current_line = ""

        for word in split_words:
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
        start_size: int | None = None,
        family: str = "",
        weight: str = "",
        style: str = "",
    ) -> tuple[ImageFont.FreeTypeFont, list[str]]:
        """
        Top-down search for the largest font size where the wrapped text
        block fits entirely inside (box_w × box_h).

        Starts at min(start_size, FONT_SIZE_MAX) so we never upscale beyond
        the locally-estimated size — only shrink to fit.

        Returns (font, list_of_lines). Falls back to smallest size.
        """
        top = self.font_size_max
        if start_size is not None:
            try:
                top = min(int(start_size), self.font_size_max)
            except (ValueError, TypeError):
                pass
        top = max(top, self.font_size_min)

        best_font = self._load_font(self.font_size_min, family, weight, style)
        best_lines = self._wrap_text(text, best_font, box_w)

        for size in range(top, self.font_size_min - 1, -1):
            font = self._load_font(size, family, weight, style)
            lines = self._wrap_text(text, font, box_w)
            if not lines:
                continue
            line_h = self._line_height(font)
            total_h = line_h * len(lines)
            max_line_w = max(self._text_width(font, ln) for ln in lines)

            if max_line_w <= box_w and total_h <= box_h:
                return font, lines
            best_font, best_lines = font, lines

        return best_font, best_lines
