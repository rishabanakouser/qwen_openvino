# Qwen-VL OCR Translation Service

A **backend-only** image OCR and translation API built with FastAPI and
Qwen-VL running on OpenVINO. Upload a screenshot → get back the same-size
image with translated text rendered in place, plus JSON detections.

Default backend is a **remote Ollama endpoint** (`qwen2.5-vl:7b` via an
OpenAI-compatible API) — the image is PNG-encoded to base64 and posted to
`/v1/chat/completions` with a maximum-completeness OCR prompt. A local
OpenVINO model (Qwen3-VL-4B INT4, with Qwen2.5-VL-7B INT8 as backup) is
switchable via `OCR_BACKEND=local`. See `models/README.md`.

---

## Architecture

Qwen does OCR **and** translation in a single pass. Font/color analysis is
local OpenCV (no second model).

```
POST /translate
       │
       ▼
   FastAPI (main.py)
       │
       ▼
   OCRService            ← Qwen-VL, remote Ollama endpoint (default)
                           or local OpenVINO (openvino_genai.VLMPipeline)
       │  [{text, translated_text, bbox}]   (bboxes scaled to pixels)
       ▼
   FontStyleService      ← OpenCV: font/background color, size, weight
       │  enriched detections
       ▼
   InpaintService        ← solid fill on flat bg, else TELEA inpaint
       │  clean image (original text erased)
       ▼
   TextRenderer          ← Pillow: auto-fit + wrap + centered TTF text
       │  final image (same dimensions as input)
       ▼
   JSON response + GET /outputs/{filename}
```

---

## Project Structure

```
qwen_openvino/
│
├── main.py                     # FastAPI app, lifespan, endpoints
├── config.py                   # All settings via environment variables
├── requirements.txt
├── .env                        # Active config (copy from .env.example)
├── .env.example                # Documented defaults
├── download_model.py           # Qwen2.5-VL INT8 conversion script
├── download_font.py            # DejaVuSans download script
│
├── services/
│   ├── __init__.py
│   ├── ocr_service.py          # Qwen inference + robust JSON parsing + bbox validation
│   ├── font_style_service.py   # Local font/color/size estimation (OpenCV)
│   ├── inpaint_service.py      # Erase original text (fill or TELEA)
│   └── text_renderer.py        # Auto-fit TTF rendering inside bboxes
│
├── models/
│   ├── README.md               # Both models: download / convert / switch
│   ├── Qwen3-VL-4B-Instruct-int4-ov/
│   └── Qwen2.5-VL-7B-Instruct-int8-ov/
│
├── fonts/
│   └── DejaVuSans.ttf          # Fallback render font (run download_font.py)
│
├── uploads/                    # Unused staging dir (kept for compatibility)
└── outputs/                    # Generated translated images
```

---

## Requirements

| Component      | Requirement                              |
|----------------|------------------------------------------|
| Python         | 3.10 – 3.12                              |
| RAM            | ≥ 16 GB (32 GB for INT8 model conversion)|
| GPU (optional) | Intel Arc / Intel iGPU with OpenVINO GPU |
| Disk           | ≥ 5 GB (INT4) / ≥ 10 GB (INT8)           |

---

## Installation

```bash
cd qwen_openvino

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
python download_font.py       # fonts/DejaVuSans.ttf
```

Get a model (see `models/README.md` for all options):

```bash
huggingface-cli download \
    OpenVINO/Qwen3-VL-4B-Instruct-int4-ov \
    --local-dir ./models/Qwen3-VL-4B-Instruct-int4-ov
```

```bash
copy .env.example .env        # Windows
# cp .env.example .env         # Linux / macOS
```

---

## Starting the service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Startup loads the VLM once (compilation takes minutes on first run).

Switch models via `.env` and restart:

```env
MODEL_PATH=./models/Qwen3-VL-4B-Instruct-int4-ov
# MODEL_PATH=./models/Qwen2.5-VL-7B-Instruct-int8-ov
```

---

## API Reference

### `POST /translate`

**Request** (`multipart/form-data`):

| Field             | Type   | Required | Description                        |
|-------------------|--------|----------|------------------------------------|
| `image`           | file   | ✅       | Image file (PNG, JPG, BMP, etc.)   |
| `source_language` | string | ✅       | Language of the original text      |
| `target_language` | string | ✅       | Language to translate into         |

**Response** (`application/json`):

```json
{
  "success": true,
  "source_language": "English",
  "target_language": "French",
  "detections": [
    {
      "text": "Settings",
      "translated_text": "Paramètres",
      "bbox": [42, 21, 74, 40]
    }
  ],
  "image_url": "/outputs/3f2a1b4c8e9d0f1a2b3c4d5e6f7a8b9c.png",
  "timing": {
    "ocr_seconds": 100.79,
    "translation_seconds": 0.0,
    "inpaint_seconds": 0.03,
    "render_seconds": 0.01,
    "total_seconds": 100.9
  }
}
```

> `translation_seconds` is `0.0` because translation is bundled into the
> single Qwen pass. Output image pixels always equal input pixels.

### `GET /outputs/{filename}` — translated image (PNG).

### `GET /health` — `{"status": "ok"|"loading", "model", "device"}`.

### `GET /info` — model path, device, OpenVINO version, font, inpaint settings.

### curl (Windows)

```cmd
curl -X POST "http://localhost:8000/translate" ^
     -F "image=@input.png" ^
     -F "source_language=English" ^
     -F "target_language=French"
```

---

## Pipeline Details

### 1 — OCR + Translation (OCRService, single Qwen pass)

- The prompt asks for **normalized 0-1000 bboxes** (`0,0` top-left) because
  Qwen's vision preprocessor resizes internally — raw "pixels" come back in
  an unknown resized space. `OCR_COORD_MODE=auto` scales `0-1000 → pixels`
  deterministically; legacy pixel responses pass through untouched.
- Tiny boxes (`<8px`) are dropped; overlapping fragments (`IoU > 0.3`) are
  merged into their union box with concatenated text; output is sorted in
  reading order.
- The JSON parser recovers from markdown fences, trailing prose, trailing
  commas, control characters, and salvages valid `{...}` objects
  individually — one corrupt entry never discards the rest. Parse failures
  log `msg + char position + snippet`.

### 2 — Font/style (FontStyleService, no model, ms per box)

- Background = median of bbox border pixels; font color = Otsu text cluster
  (median + dark/bright core so anti-aliasing doesn't wash it gray).
- Size ≈ `box_h × 0.8` clamped to `FONT_SIZE_MIN/MAX`; bold if text-pixel
  density > 0.35.

### 3 — Inpainting (InpaintService)

- Flat background (gray std < 18) + known color → fast solid fill.
- Otherwise (gradients, code blocks, dark UI) → TELEA inpaint with a
  3px-dilated mask so glyph edges don't leave halos.

### 4 — Rendering (TextRenderer)

- Fits against `box_w × 1.2` (room for EN→FR expansion), starting from the
  estimated size and shrinking until the wrapped block fits.
- Long unbroken tokens (commands, URLs) are hard-split character-wise.
- Text is centered in the original box using the system font
  (`arial/bold/italic` on Windows) or `DejaVuSans.ttf` fallback.

---

## Configuration Reference

| Variable          | Default                                         | Description                        |
|-------------------|-------------------------------------------------|------------------------------------|
| `OCR_BACKEND`     | `remote`                                        | `remote` (Ollama endpoint) / `local` (OpenVINO) |
| `OLLAMA_BASE_URL` | `http://34.63.203.19:11434/v1`                 | OpenAI-compatible endpoint         |
| `OLLAMA_API_KEY`  | `EMPTY`                                         | Endpoint API key                   |
| `OLLAMA_MODEL`    | `qwen2.5-vl:7b`                                 | Remote model id                    |
| `OLLAMA_TIMEOUT`  | `300`                                           | Request timeout (seconds)          |
| `OLLAMA_MAX_TOKENS` | `3500`                                        | Max output tokens per OCR request  |
| `MODEL_PATH`      | `./models/Qwen3-VL-4B-Instruct-int4-ov`         | OV model dir (local backend only)  |
| `OPENVINO_DEVICE` | `GPU`                                           | `GPU` / `CPU` / `AUTO` / `NPU` (local only) |
| `OCR_COORD_MODE`  | `auto`                                          | `auto` / `normalized` / `pixels`   |
| `INPAINT_RADIUS`  | `5`                                             | OpenCV inpaint brush radius        |
| `BBOX_PADDING`    | `2`                                             | Extra pixels around each bbox      |
| `FONT_PATH`       | `./fonts/DejaVuSans.ttf`                        | Fallback TrueType font             |
| `FONT_SIZE_MAX`   | `40`                                            | Largest font size tried            |
| `FONT_SIZE_MIN`   | `6`                                             | Smallest font size before giving up|

Set `OPENVINO_DEVICE=AUTO` if `GPU` fails at load time.

---

## Logging

Every request logs one line per stage (`main`, `services.ocr_service`).
Raw model JSON is logged at `DEBUG` level only. No per-request `print()`
noise — console output is the startup banner plus the `INFO` lines.

---

## Error Handling

| Error                        | HTTP Status | Response                              |
|------------------------------|-------------|---------------------------------------|
| Missing image                | 422         | `"Uploaded image is empty"`           |
| Invalid image format         | 422         | `"Cannot open image: ..."`            |
| Missing source_language      | 422         | `"source_language is required"`       |
| Missing target_language      | 422         | `"target_language is required"`       |
| Model not found at startup   | crash       | Fix `MODEL_PATH`, see `models/README` |
| OCR JSON unparseable         | 200 + `[]`  | Partial salvage attempted first       |
| No text regions detected     | 422         | Nothing to translate — input returned as error, never as output |
| Nothing rendered (all skipped) | 500       | Per-region reasons in `detail` (empty-translation, degenerate-bbox, render-error) |
| Output identical to input    | 500         | Safety net — inpaint/render left no visible change |
| Invalid bbox                 | 200 (skip)  | Bad box skipped, others processed     |
| Output file not found        | 404         | Standard 404                          |
| Services not initialised     | 503         | `"Services not initialised"`          |
