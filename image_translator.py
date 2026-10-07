
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

logger = logging.getLogger("image_translator")


# ── Prompt ────────────────────────────────────────────────────────────────
def build_ocr_prompt(source_language: str, target_language: str) -> str:
    return (
        f"Detect and extract EVERY visible text region in the image (assuming it is mostly {source_language}) "
        "with maximum completeness. Do NOT omit, summarize, merge, or ignore ANY text, "
        "regardless of size or position. Capture all text including large and small text, "
        "headings, titles, labels, buttons, menus, navigation/sidebar text, captions, "
        "tooltips, footnotes, numbers, symbols, text near the edges/corners, partially "
        "visible text, and text inside UI elements. Each distinct visible text region "
        "must be returned separately. Preserve the text exactly as it appears. "
        f"Translate each detected text region into {target_language} "
        "(keep code/commands like 'netstat -ano | findstr :8000' untranslated "
        "if a translation makes no sense). "
        "Return ONLY a valid JSON array, one object per text region in this exact format:\n"
        '[{"text": "original text", "translated_text": "translation", "bbox": [x1, y1, x2, y2]}, ...]\n'
        "The bbox must tightly enclose the corresponding text and use NORMALIZED "
        "0-1000 coordinates (0,0 = top-left, 1000,1000 = bottom-right). "
        "Do NOT guess pixels — the image may be resized internally. "
        "Every object MUST contain exactly these three keys in this order — "
        "never omit \"text\", never repeat a key, always close arrays with ]. "
        "Do not return Markdown, explanations, descriptions, or any text outside the JSON array."
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, int(rgb[0]))),
        max(0, min(255, int(rgb[1]))),
        max(0, min(255, int(rgb[2]))),
    )


# ── JSON recovery ─────────────────────────────────────────────────────────
def _try_json_list(s: str) -> list[dict[str, Any]] | None:
    try:
        data = json.loads(s)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def parse_model_response(raw: str) -> list[dict[str, Any]]:
    """Extract a JSON array from raw VLM output, with multi-stage recovery."""
    hit = _try_json_list(raw)
    if hit is not None:
        return hit

    stripped = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    if stripped != raw:
        hit = _try_json_list(stripped)
        if hit is not None:
            return hit

    start, end = raw.find("["), raw.rfind("]")
    block = raw[start:end + 1] if 0 <= start < end else ""
    if block and block != raw.strip():
        hit = _try_json_list(block)
        if hit is not None:
            logger.info("Recovered JSON via bracket slice (%d chars)", len(block))
            return hit
    elif not block:
        logger.warning("No [...] block found; raw len=%d head=%.200r", len(raw), raw[:200])

    candidate = block or raw
    repaired = re.sub(r",(\s*[}\]])", r"\1", candidate)
    if repaired != candidate:
        hit = _try_json_list(repaired)
        if hit is not None:
            logger.info("Recovered JSON after trailing-comma repair")
            return hit
    closed = re.sub(r"(\[[^\[\]]*)\}", r"\1]}", candidate)
    if closed != candidate:
        hit = _try_json_list(closed)
        if hit is not None:
            logger.info("Recovered JSON after unclosed-array repair")
            return hit
    try:
        data = json.loads(candidate, strict=False)
        if isinstance(data, list):
            logger.info("Recovered JSON with strict=False")
            return data
    except json.JSONDecodeError as exc:
        logger.warning("strict=False also failed: %s at %d snippet=%.120r",
                       exc.msg, exc.pos, candidate[max(0, exc.pos - 60):exc.pos + 60])

    salvaged: list[dict[str, Any]] = []
    for m in re.finditer(r"\{[^{}]*\}", candidate):
        obj_str = re.sub(r"(\[[^\[\]]*)\}", r"\1]}", m.group())
        obj = None
        try:
            obj = json.loads(obj_str)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(obj_str)
            except (ValueError, SyntaxError):
                continue
        if isinstance(obj, dict) and (obj.get("text") or obj.get("translated_text")) and obj.get("bbox"):
            salvaged.append(obj)
    if salvaged:
        logger.warning("Salvaged %d objects individually (full-array parse failed)", len(salvaged))
        return salvaged

    try:
        json.loads(candidate)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse model response: %s at char %d of %d; head=%.200r tail=%.200r",
                       exc.msg, exc.pos, len(candidate), candidate[:200], candidate[-200:])
    else:
        logger.warning("Response parsed but was not a list; head=%.200r", candidate[:200])
    return []


# ── Bbox validation + NMS ─────────────────────────────────────────────────
MIN_BOX_W = 8
MIN_BOX_H = 8
NMS_IOU_THRESHOLD = 0.3


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _filter_and_merge(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if (x2 - x1) < MIN_BOX_W or (y2 - y1) < MIN_BOX_H:
            logger.warning("Dropping tiny box %s text='%.30s'", det["bbox"], det["text"])
            continue
        kept.append(det)
    kept.sort(key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]), reverse=True)
    merged: list[dict[str, Any]] = []
    for det in kept:
        placed = False
        for m in merged:
            if _iou(det["bbox"], m["bbox"]) > NMS_IOU_THRESHOLD:
                mx1, my1, mx2, my2 = m["bbox"]
                x1, y1, x2, y2 = det["bbox"]
                m["bbox"] = [min(mx1, x1), min(my1, y1), max(mx2, x2), max(my2, y2)]
                if det["text"] not in m["text"]:
                    m["text"] = (m["text"] + " " + det["text"]).strip()
                if det.get("translated_text") and det["translated_text"] not in (m.get("translated_text") or ""):
                    m["translated_text"] = ((m.get("translated_text") or "") + " " + det["translated_text"]).strip()
                placed = True
                break
        if not placed:
            merged.append(det)
    merged.sort(key=lambda d: (d["bbox"][1] // 10, d["bbox"][0]))
    return merged


def validate_detections(
    detections: list[dict[str, Any]],
    img_w: int,
    img_h: int,
    coord_mode: str = "auto",
) -> list[dict[str, Any]]:
    """Scale (if 0-1000 normalized), clip, validate; drop the invalid."""
    mode = (coord_mode or "auto").lower()
    if mode == "auto":
        try:
            all_vals = [float(v) for det in detections for v in (det.get("bbox") or [])]
            mode = "normalized" if all_vals and max(all_vals) <= 1000 else "pixels"
        except (ValueError, TypeError):
            mode = "pixels"
    logger.info("Coord mode: %s (image %dx%d)", mode, img_w, img_h)
    valid: list[dict[str, Any]] = []
    for idx, det in enumerate(detections):
        try:
            text = det.get("text", "")
            if not isinstance(text, str) or not text.strip():
                fallback = det.get("translated_text", "")
                if isinstance(fallback, str) and fallback.strip():
                    logger.warning("Detection #%d missing text — rendering its translation anyway", idx)
                    text = fallback
                else:
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
            valid.append({"text": text.strip(), "translated_text": translated_text.strip(),
                          "bbox": [int(x1), int(y1), int(x2), int(y2)],
                          "font_color_rgb": None, "background_color_rgb": None,
                          "font_size": None, "font_weight": None,
                          "font_style": None, "font_family": None})
        except Exception as exc:
            logger.warning("Skipping detection #%d — %s | raw=%s", idx, exc, det)
    return _filter_and_merge(valid)


# ── Local font/style analysis (no model) ──────────────────────────────────
def _estimate_style(
    img_np: "np.ndarray",
    det: dict,
    font_size_min: int,
    font_size_max: int,
    bold_density_threshold: float = 0.35,
) -> None:
    import numpy as _np

    img_h, img_w = img_np.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
    x1, y1 = max(0, min(x1, img_w - 1)), max(0, min(y1, img_h - 1))
    x2, y2 = max(0, min(x2, img_w)), max(0, min(y2, img_h))
    if x2 - x1 < 3 or y2 - y1 < 3:
        det.update({"font_color_rgb": (0, 0, 0), "font_size": font_size_min,
                    "font_weight": "normal", "font_style": "normal",
                    "font_family": "sans-serif"})
        return

    crop = img_np[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    border = _np.concatenate([crop[0:1].reshape(-1, 3), crop[-1:].reshape(-1, 3),
                              crop[:, 0:1].reshape(-1, 3), crop[:, -1:].reshape(-1, 3)], axis=0)
    bg_rgb = tuple(int(v) for v in _np.median(border, axis=0))

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_mask, black_mask = binary == 255, binary != 255
    n_white, n_black = int(_np.count_nonzero(white_mask)), int(_np.count_nonzero(black_mask))

    if n_white == 0 or n_black == 0:
        luma = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        font_rgb = (0, 0, 0) if luma > 128 else (255, 255, 255)
        text_density = 0.0
    else:
        mean_white = tuple(float(v) for v in crop[white_mask].mean(axis=0))
        mean_black = tuple(float(v) for v in crop[black_mask].mean(axis=0))
        d_white = sum((a - b) ** 2 for a, b in zip(mean_white, bg_rgb))
        d_black = sum((a - b) ** 2 for a, b in zip(mean_black, bg_rgb))
        if d_white < d_black:
            bg_est, font_pixels = mean_white, crop[black_mask]
            text_density = n_black / (ch * cw)
        else:
            bg_est, font_pixels = mean_black, crop[white_mask]
            text_density = n_white / (ch * cw)
        bg_rgb = tuple(int(round(v)) for v in bg_est)
        bg_luma = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        med = _np.median(font_pixels.reshape(-1, 3), axis=0)
        med_luma = 0.299 * med[0] + 0.587 * med[1] + 0.114 * med[2]
        pct = 20 if med_luma < bg_luma else 80
        core = _np.percentile(font_pixels.reshape(-1, 3), pct, axis=0)
        font_rgb = tuple(int(round((m + c) / 2)) for m, c in zip(med, core))

    box_h = y2 - y1
    est_size = max(font_size_min, min(font_size_max, int(round(box_h * 0.8)) or 24))
    det["font_color_rgb"] = det.get("font_color_rgb") or font_rgb
    det["font_color_hex"] = det.get("font_color_hex") or _rgb_to_hex(det["font_color_rgb"])
    det["background_color_rgb"] = det.get("background_color_rgb") or bg_rgb
    det["background_color_hex"] = det.get("background_color_hex") or _rgb_to_hex(bg_rgb)
    det["font_size"] = det.get("font_size") or est_size
    det["font_weight"] = det.get("font_weight") or ("bold" if text_density > bold_density_threshold else "normal")
    det["font_style"] = det.get("font_style") or "normal"
    det["font_family"] = det.get("font_family") or "sans-serif"


# ── Inpainting ────────────────────────────────────────────────────────────
def remove_text(
    image: Image.Image,
    detections: list[dict],
    inpaint_radius: int = 5,
    bbox_padding: int = 2,
    flat_bg_std_threshold: float = 18.0,
) -> Image.Image:
    """Erase original text: solid fill on flat bg, else TELEA inpaint."""
    if not detections:
        return image
    img_np = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for det in detections:
        try:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            bg_color = det.get("background_color_rgb")
            px1, py1 = max(0, x1 - bbox_padding), max(0, y1 - bbox_padding)
            px2, py2 = min(w, x2 + bbox_padding), min(h, y2 + bbox_padding)
            if px2 <= px1 or py2 <= py1:
                continue
            is_flat = float(gray[py1:py2, px1:px2].std()) < flat_bg_std_threshold
            if bg_color and is_flat:
                cv2.rectangle(img_bgr, (px1, py1), (px2, py2),
                              (int(bg_color[2]), int(bg_color[1]), int(bg_color[0])), -1)
            else:
                mask[py1:py2, px1:px2] = 255
        except Exception as exc:
            logger.warning("Inpaint skipping bad bbox %s: %s", det.get("bbox"), exc)
    if cv2.countNonZero(mask) > 0:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        img_bgr = cv2.inpaint(img_bgr, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


# ── Text rendering ────────────────────────────────────────────────────────
_PILLOW_DEFAULT_FONT = ImageFont.load_default()


def _load_font(font_path: Optional[str], size: int, family: str = "",
               weight: str = "", style: str = "") -> ImageFont.FreeTypeFont:
    is_bold, is_italic = "bold" in weight.lower(), "italic" in style.lower()
    name = "arial"
    if "times" in family.lower():
        name = "times"
    elif "courier" in family.lower():
        name = "cour"
    elif "segoe" in family.lower():
        name = "segoeui"
    suffix = ""
    if is_bold and is_italic:
        suffix = "bi" if name in ["arial", "times"] else "z"
    elif is_bold:
        suffix = "bd" if name in ["arial", "times", "cour"] else "b"
    elif is_italic:
        suffix = "i"
    sys_path = Path(f"C:/Windows/Fonts/{name}{suffix}.ttf")
    if sys_path.exists():
        try:
            return ImageFont.truetype(str(sys_path), size=size)
        except Exception:
            pass
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return _PILLOW_DEFAULT_FONT


def _text_width(font, text: str) -> int:
    try:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0]
    except AttributeError:
        return font.getlength(text)  # type: ignore


def _line_height(font) -> int:
    try:
        ascent, descent = font.getmetrics()
        return ascent + descent
    except AttributeError:
        return 16


def _wrap_text(text: str, font, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [text]
    split_words: list[str] = []
    for word in words:
        if _text_width(font, word) <= max_width:
            split_words.append(word)
            continue
        chunk = ""
        for ch in word:
            cand = chunk + ch
            if _text_width(font, cand) <= max_width and chunk:
                chunk = cand
            else:
                if chunk:
                    split_words.append(chunk)
                chunk = ch
                if _text_width(font, chunk) > max_width:
                    split_words.append(chunk)
                    chunk = ""
        if chunk:
            split_words.append(chunk)
    lines: list[str] = []
    current = ""
    for word in split_words:
        cand = (current + " " + word).strip() if current else word
        if _text_width(font, cand) <= max_width:
            current = cand
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _fit_text(text: str, box_w: int, box_h: int, font_path: Optional[str],
              size_max: int, size_min: int, start_size: int | None = None,
              family: str = "", weight: str = "", style: str = ""):
    top = min(int(start_size), size_max) if start_size is not None else size_max
    top = max(top, size_min)
    best_font = _load_font(font_path, size_min, family, weight, style)
    best_lines = _wrap_text(text, best_font, box_w)
    for size in range(top, size_min - 1, -1):
        font = _load_font(font_path, size, family, weight, style)
        lines = _wrap_text(text, font, box_w)
        if not lines:
            continue
        line_h = _line_height(font)
        if max(_text_width(font, ln) for ln in lines) <= box_w and line_h * len(lines) <= box_h:
            return font, lines
        best_font, best_lines = font, lines
    return best_font, best_lines


def _skip_reason(det: dict) -> str | None:
    if not (det.get("translated_text") or "").strip():
        return "empty-translation"
    try:
        x1, y1, x2, y2 = det["bbox"]
    except (KeyError, TypeError, ValueError):
        return "bad-bbox"
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return "degenerate-bbox"
    return None


def render_all(
    image: Image.Image,
    detections: list[dict],
    font_path: Optional[str] = None,
    font_size_max: int = 40,
    font_size_min: int = 6,
) -> tuple[Image.Image, int, list[tuple[str, str, object]]]:
    """Draw translations; returns (image, rendered_count, [(reason, text, bbox)])."""
    rendered = 0
    skipped: list[tuple[str, str, object]] = []
    for det in detections:
        reason = _skip_reason(det)
        if reason is not None:
            skipped.append((reason, det.get("text", ""), det.get("bbox")))
            continue
        try:
            image = _render_one(image, det, font_path, font_size_max, font_size_min)
            rendered += 1
        except Exception as exc:
            logger.warning("Render failed for det %s: %s", det, exc)
            skipped.append((f"render-error: {exc}", det.get("text", ""), det.get("bbox")))
    return image, rendered, skipped


def _render_one(image: Image.Image, det: dict, font_path: Optional[str],
                size_max: int, size_min: int) -> Image.Image:
    image = image.copy()
    translated_text = det.get("translated_text", "")
    x1, y1, x2, y2 = det["bbox"]
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"degenerate bbox {det['bbox']}")
    try:
        est_size = int(det.get("font_size") or 24)
    except (ValueError, TypeError):
        est_size = 24
    est_size = max(size_min, min(size_max, est_size))
    color = det.get("font_color_rgb") or (0, 0, 0)
    draw = ImageDraw.Draw(image)
    fit_w, fit_h = max(8, int(box_w * 1.2)), max(8, box_h)
    font, lines = _fit_text(translated_text, fit_w, fit_h, font_path, size_max, size_min,
                            start_size=est_size, family=det.get("font_family") or "sans-serif",
                            weight=det.get("font_weight") or "normal",
                            style=det.get("font_style") or "normal")
    line_h, total_h = _line_height(font), _line_height(font) * len(lines)
    start_y = y1 + (box_h - total_h) // 2
    for i, line in enumerate(lines):
        start_x = x1 + (box_w - _text_width(font, line)) // 2
        draw.text((start_x, start_y + i * line_h), line, font=font, fill=tuple(color))
    return image


# ── Orchestrator ──────────────────────────────────────────────────────────
class ImageTranslator:
    """Portable image translator. Create once, call translate_image per image."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "qwen2.5-vl:7b",
        timeout: int = 300,
        max_tokens: int = 3500,
        font_path: Optional[str] = None,
        font_size_max: int = 40,
        font_size_min: int = 6,
        inpaint_radius: int = 5,
        bbox_padding: int = 2,
        coord_mode: str = "auto",
        client: Any = None,
    ):
        """
        Args:
            base_url: OpenAI-compatible endpoint, e.g. http://host:11434/v1
            client: optional prebuilt client (OpenAI-compatible, exposes
                    client.chat.completions.create) — handy for tests/DI.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.font_path = font_path
        self.font_size_max = font_size_max
        self.font_size_min = font_size_min
        self.inpaint_radius = inpaint_radius
        self.bbox_padding = bbox_padding
        self.coord_mode = coord_mode
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    @staticmethod
    def _pil_to_data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _ocr_translate(self, image: Image.Image, source: str, target: str) -> str:
        last_exc: Exception | None = None
        messages = [{"role": "user", "content": [
            {"type": "text", "text": build_ocr_prompt(source, target)},
            {"type": "image_url", "image_url": {"url": self._pil_to_data_url(image)}},
        ]}]
        for attempt in (1, 2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=self.max_tokens, temperature=0)
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("empty response content from model")
                return text
            except Exception as exc:
                last_exc = exc
                logger.warning("VLM attempt %d/2 failed: %s", attempt, exc)
                if attempt == 1:
                    time.sleep(2)
        raise RuntimeError(f"VLM inference failed: {last_exc}") from last_exc

    def translate_image(
        self, image: Image.Image, source_language: str, target_language: str
    ) -> tuple[Image.Image, list[dict[str, Any]]]:
        """
        Translate all text in *image*.

        Returns (output_image, detections) where each detection is
        {"text", "translated_text", "bbox"}. Output has input dimensions.
        Raises ValueError/RuntimeError with the reason instead of silently
        returning the input image.
        """
        image = image.convert("RGB")
        orig_w, orig_h = image.size

        raw = self._ocr_translate(image, source_language, target_language)
        detections = validate_detections(parse_model_response(raw), orig_w, orig_h, self.coord_mode)
        if not detections:
            raise ValueError("No text regions detected in the image — nothing to translate.")

        img_np = np.array(image)
        for det in detections:
            try:
                _estimate_style(img_np, det, self.font_size_min, self.font_size_max)
            except Exception as exc:
                logger.warning("Style analysis failed for bbox %s: %s", det.get("bbox"), exc)
                det.update({"font_color_rgb": (0, 0, 0), "font_size": 24,
                            "font_weight": "normal", "font_style": "normal",
                            "font_family": "sans-serif"})

        inpainted = remove_text(image, detections, self.inpaint_radius, self.bbox_padding)
        final, rendered, skipped = render_all(
            inpainted, detections, self.font_path, self.font_size_max, self.font_size_min)
        if rendered == 0:
            reasons = "; ".join(f"{r} (text={t!r:.40})" for r, t, _ in skipped) or "unknown"
            raise RuntimeError(f"Rendering failed for all {len(detections)} regions: {reasons}.")

        if final.size != (orig_w, orig_h):
            final = final.resize((orig_w, orig_h), Image.LANCZOS)
        if ImageChops.difference(final, image).getbbox() is None:
            raise RuntimeError("Rendering produced no visible change — output identical to input.")
        return final, [{"text": d["text"], "translated_text": d.get("translated_text", ""),
                        "bbox": d["bbox"]} for d in detections]


def main() -> None:
    ap = argparse.ArgumentParser(description="Translate all text in an image.")
    ap.add_argument("input", help="Input image path")
    ap.add_argument("output", help="Output image path (PNG)")
    ap.add_argument("--source", default="English")
    ap.add_argument("--target", default="French")
    ap.add_argument("--base-url", required=True, help="e.g. http://34.63.203.19:11434/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="qwen2.5-vl:7b")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=3500)
    ap.add_argument("--font", default=None, help="TTF path (falls back to system/Pillow fonts)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    tr = ImageTranslator(base_url=args.base_url, api_key=args.api_key, model=args.model,
                         timeout=args.timeout, max_tokens=args.max_tokens, font_path=args.font)
    out, dets = tr.translate_image(Image.open(args.input), args.source, args.target)
    out.save(args.output, format="PNG")
    print(f"Saved {args.output} with {len(dets)} regions")
    for d in dets:
        print(f"  {d['text']!r} -> {d['translated_text']!r} {d['bbox']}")


if __name__ == "__main__":
    main()
