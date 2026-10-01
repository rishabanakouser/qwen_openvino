"""
config.py - Centralized configuration for the OCR Translation Service.
All settings are read from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.resolve()

MODEL_PATH: str = os.getenv(
    "MODEL_PATH",
    str(BASE_DIR / "models" / "Qwen3-VL-4B-Instruct-int4-ov"),
)

FONT_PATH: str = os.getenv(
    "FONT_PATH",
    str(BASE_DIR / "fonts" / "DejaVuSans.ttf"),
)

UPLOADS_DIR: Path = BASE_DIR / "uploads"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# OpenVINO / Model
OPENVINO_DEVICE: str = os.getenv("OPENVINO_DEVICE", "GPU")

# Qwen bbox coordinate convention: "auto" | "normalized" | "pixels".
# "auto" treats all-values-<=1000 responses as 0-1000 normalized and scales
# to pixels; larger values pass through as pixels (legacy behaviour).
OCR_COORD_MODE: str = os.getenv("OCR_COORD_MODE", "auto")

# Inpainting
INPAINT_RADIUS: int = int(os.getenv("INPAINT_RADIUS", "5"))
BBOX_PADDING: int = int(os.getenv("BBOX_PADDING", "2"))

# Text Rendering
FONT_SIZE_MAX: int = int(os.getenv("FONT_SIZE_MAX", "40"))
FONT_SIZE_MIN: int = int(os.getenv("FONT_SIZE_MIN", "6"))

# Model metadata (informational)
MODEL_NAME: str = "Qwen3-VL-4B-Instruct"
