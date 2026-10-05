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

# OpenVINO / Model (local backend only)
OPENVINO_DEVICE: str = os.getenv("OPENVINO_DEVICE", "GPU")

# OCR backend: "remote" (Ollama/OpenAI-compatible endpoint) or "local" (OpenVINO)
OCR_BACKEND: str = os.getenv("OCR_BACKEND", "remote").lower()

# Remote VLM endpoint (Ollama OpenAI-compatible API)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://34.63.203.19:11434/v1")
OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "EMPTY")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5-vl:7b")
OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))
OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "3500"))

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
