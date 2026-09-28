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
    ):
        self.inpaint_radius = inpaint_radius
        self.bbox_padding = bbox_padding

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def remove_text(
        self,
        image: Image.Image,
        detections: list[dict],
    ) -> Image.Image:
        """
        Remove text from *image* at every bbox listed in *detections*.

        Args:
            image:      PIL RGB image.
            detections: List of {"text": ..., "bbox": [x1, y1, x2, y2]}.

        Returns:
            New PIL RGB image with text regions inpainted.
        """
        if not detections:
            logger.info("Inpaint | no detections, returning original image")
            return image

        img_np = np.array(image.convert("RGB"))          # H×W×3 uint8
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Build a single combined mask covering all text regions
        mask = self._build_mask(img_bgr.shape[:2], detections)

        # Inpaint
        inpainted_bgr = cv2.inpaint(
            img_bgr,
            mask,
            inpaintRadius=self.inpaint_radius,
            flags=cv2.INPAINT_TELEA,
        )

        inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(inpainted_rgb)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _build_mask(
        self,
        shape: tuple[int, int],
        detections: list[dict],
    ) -> np.ndarray:
        """
        Create a uint8 binary mask (255 = inpaint, 0 = keep).

        Each bbox is expanded by self.bbox_padding pixels on every side
        and clipped to image boundaries.
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        pad = self.bbox_padding

        for det in detections:
            try:
                x1, y1, x2, y2 = det["bbox"]
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                mask[y1:y2, x1:x2] = 255
            except Exception as exc:
                logger.warning("Inpaint | skipping bad bbox %s: %s", det.get("bbox"), exc)

        return mask
