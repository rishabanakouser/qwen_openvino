"""
services/inpaint_service.py

InpaintService removes the original text from the image using OpenCV's
Telea inpainting algorithm.  Each detected bounding-box is expanded by a
configurable padding, converted to a binary mask, and inpainted so the
surrounding background is reconstructed without the text.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

import config

logger = logging.getLogger(__name__)


class InpaintService:
    """
    Removes text from an image based on OCR bounding boxes.

    Configuration is read from config.py (INPAINT_RADIUS, BBOX_PADDING).
    """

    def __init__(
        self,
        inpaint_radius: int = config.INPAINT_RADIUS,
        bbox_padding: int = config.BBOX_PADDING,
        flat_bg_std_threshold: float = 18.0,
    ):
        self.inpaint_radius = inpaint_radius
        self.bbox_padding = bbox_padding
        # Solid-fill is only safe on flat backgrounds. Above this grayscale
        # std the region is gradient/code-block/photo -> must use TELEA.
        self.flat_bg_std_threshold = flat_bg_std_threshold

    # Public API
    def remove_text(
        self,
        image: Image.Image,
        detections: list[dict],
    ) -> Image.Image:
        """
        Remove text from *image* at every bbox listed in *detections*.

        Strategy per box:
          - Flat background (gray std < threshold) + known bg color
            -> fast solid fill (clean, no blur).
          - Otherwise (gradient, code-block, dark UI, photo)
            -> TELEA inpaint with dilated mask (reconstructs texture).
        """
        if not detections:
            logger.info("Inpaint | no detections, returning original image")
            return image

        img_np = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        mask = np.zeros((h, w), dtype=np.uint8)
        pad = self.bbox_padding

        for det in detections:
            try:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                bg_color = det.get("background_color_rgb")

                px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
                px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
                if px2 <= px1 or py2 <= py1:
                    continue
                region_std = float(gray[py1:py2, px1:px2].std())
                is_flat = region_std < self.flat_bg_std_threshold

                if bg_color and is_flat:
                    bgr = (int(bg_color[2]), int(bg_color[1]), int(bg_color[0]))
                    cv2.rectangle(img_bgr, (px1, py1), (px2, py2), bgr, -1)
                else:
                    mask[py1:py2, px1:px2] = 255
            except Exception as exc:
                logger.warning("Inpaint | skipping bad bbox %s: %s", det.get("bbox"), exc)

        # Dilate mask 3px so anti-aliased text edges don't leave halos
        # (this was visible as ghost outlines on your dark screenshot).
        if cv2.countNonZero(mask) > 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            inpainted_bgr = cv2.inpaint(
                img_bgr,
                mask,
                inpaintRadius=self.inpaint_radius,
                flags=cv2.INPAINT_TELEA,
            )
        else:
            inpainted_bgr = img_bgr

        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(inpainted_rgb)
