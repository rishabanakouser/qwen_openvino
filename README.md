# Qwen2.5-VL OCR Translation Service

A **backend-only** image OCR and translation API built with FastAPI and
Qwen2.5-VL-7B-Instruct running on OpenVINO (INT8).

---

## Architecture

```
POST /translate
       │
       ▼
   FastAPI (main.py)
       │
       ▼
   OCRService          ← Qwen2.5-VL-7B INT8 via OpenVINO
       │  [text, bbox] list
       ▼
   TranslationService  ← deep-translator / Google Translate
       │  [text, translated_text, bbox] list
       ▼
   InpaintService      ← OpenCV INPAINT_TELEA
       │  clean image (original text erased)
       ▼
   TextRenderer        ← Pillow + TrueType font
       │  final image (translated text rendered)
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
├── download_model.py           # One-click model conversion script
├── download_font.py            # One-click font download script
│
├── services/
│   ├── __init__.py
│   ├── ocr_service.py          # Qwen2.5-VL inference + JSON validation
│   ├── translation_service.py  # Modular translation backend
│   ├── inpaint_service.py      # OpenCV inpainting
│   └── text_renderer.py        # Auto-fit TTF text rendering
│
├── models/
│   └── README.md               # Model download instructions
│
├── fonts/
│   └── README.md               # Font download instructions
│
├── uploads/                    # Temporary uploaded images
└── outputs/                    # Generated translated images
```

---

## Requirements

| Component      | Requirement                              |
|----------------|------------------------------------------|
| Python         | 3.10 or 3.11                             |
| RAM            | ≥ 16 GB (32 GB for model conversion)    |
| GPU (optional) | Intel Arc / Intel iGPU with OpenVINO GPU |
| Disk           | ≥ 10 GB for the INT8 OV model            |

---

## Installation

### 1. Clone / navigate to the project

```bash
cd qwen_openvino
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the font

```bash
python download_font.py
```

### 5. Convert / download the model

**Option A — automatic conversion from HuggingFace (recommended):**

```bash
python download_model.py
```

This downloads Qwen2.5-VL-7B-Instruct (~15 GB) and converts it to
OpenVINO INT8 format (~8 GB).  One-time operation, takes 10–30 minutes.

**Option B — manual optimum-cli:**

```bash
optimum-cli export openvino \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --weight-format int8 \
    ./models/Qwen2.5-VL-7B-Instruct-int8-ov
```

**Option C — HuggingFace Hub pre-converted:**

```bash
huggingface-cli download \
    OpenVINO/Qwen2.5-VL-7B-Instruct-int8-ov \
    --local-dir ./models/Qwen2.5-VL-7B-Instruct-int8-ov
```

### 6. Configure `.env`

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux / macOS
```

Edit `.env` to match your setup:

```env
MODEL_PATH=./models/Qwen2.5-VL-7B-Instruct-int8-ov
OPENVINO_DEVICE=GPU       # GPU | CPU | AUTO
INPAINT_RADIUS=5
BBOX_PADDING=2
FONT_PATH=./fonts/DejaVuSans.ttf
FONT_SIZE_MAX=40
FONT_SIZE_MIN=6
```

---

## Starting the service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Startup output:

```
============================================================
Available OpenVINO devices:
  CPU
  GPU
Selected device: GPU
============================================================
Loading model from: ./models/Qwen2.5-VL-7B-Instruct-int8-ov
This may take a few minutes on first run (model compilation)...
Model loaded successfully.
============================================================
```

---

## API Reference

### `POST /translate`

Translates all text in an uploaded image.

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
      "text": "Open",
      "translated_text": "Ouvrir",
      "bbox": [99, 88, 177, 154]
    },
    {
      "text": "Settings",
      "translated_text": "Paramètres",
      "bbox": [300, 200, 450, 250]
    }
  ],
  "image_url": "/outputs/3f2a1b4c8e9d0f1a2b3c4d5e6f7a8b9c.png",
  "timing": {
    "ocr_seconds": 4.21,
    "translation_seconds": 0.85,
    "inpaint_seconds": 0.03,
    "render_seconds": 0.01,
    "total_seconds": 5.10
  }
}
```

---

### `GET /outputs/{filename}`

Returns the generated translated image file.

```
GET /outputs/3f2a1b4c8e9d0f1a2b3c4d5e6f7a8b9c.png
```

---

### `GET /health`

```json
{
  "status": "ok",
  "model": "Qwen2.5-VL-7B-Instruct",
  "device": "GPU"
}
```

---

### `GET /info`

```json
{
  "model": "Qwen2.5-VL-7B-Instruct",
  "model_path": "./models/Qwen2.5-VL-7B-Instruct-int8-ov",
  "device": "GPU",
  "openvino_version": "2024.3.0",
  "available_devices": ["CPU", "GPU"],
  "font_path": "./fonts/DejaVuSans.ttf",
  "inpaint_radius": 5,
  "bbox_padding": 2
}
```

---

## Example Requests

### curl (Windows)

```cmd
curl -X POST "http://localhost:8000/translate" ^
     -F "image=@input.png" ^
     -F "source_language=English" ^
     -F "target_language=French"
```

### curl (Linux / macOS)

```bash
curl -X POST "http://localhost:8000/translate" \
     -F "image=@input.png" \
     -F "source_language=English" \
     -F "target_language=French"
```

### Python `requests`

```python
import requests

files = {"image": open("input.png", "rb")}
data  = {"source_language": "English", "target_language": "French"}

response = requests.post(
    "http://localhost:8000/translate",
    files=files,
    data=data,
)
result = response.json()
print(result)

# Download the translated image
img_url = "http://localhost:8000" + result["image_url"]
img_response = requests.get(img_url)
with open("output.png", "wb") as f:
    f.write(img_response.content)
print("Saved output.png")
```

---

## GPU Usage Verification

### Check at startup

The service prints the selected device on startup:

```
Selected device: GPU
```

### Check via /info

```bash
curl http://localhost:8000/info
```

Look for `"device": "GPU"`.

### Verify GPU is being used with Intel GPU tools

```bash
# Intel GPU activity monitor (Linux)
intel_gpu_top

# Or via OpenVINO benchmark tool
benchmark_app -m ./models/.../openvino_language_model.xml -d GPU
```

### Fallback behaviour

If `OPENVINO_DEVICE=GPU` is set but no GPU is available, OpenVINO will
raise an error at model load time.  Set `OPENVINO_DEVICE=AUTO` to let
OpenVINO automatically pick the best available device.

---

## Pipeline Details

### 1 — OCR (OCRService)

- The image is sent to Qwen2.5-VL with a carefully engineered prompt
  that instructs the model to return **only** a JSON array of
  `{"text", "bbox"}` objects.
- If the model wraps the JSON in markdown fences or adds prose,
  the parser strips them automatically via regex fallback.
- Every detection is validated: non-empty text, exactly 4 numeric bbox
  values, `x1 < x2`, `y1 < y2`, coordinates clipped to image bounds.
- Invalid detections are skipped with a warning — they never crash the request.

### 2 — Translation (TranslationService)

- Receives `text`, `source_language`, `target_language` per detection.
- The backend is **deep-translator** (Google Translate).
- Language names ("English", "French") are mapped to ISO 639-1 codes
  via an internal table covering 80+ languages.
- To swap the translation engine, replace `_translate_backend()` in
  `services/translation_service.py`.

### 3 — Inpainting (InpaintService)

- A binary mask is built from all OCR bboxes (optionally expanded by
  `BBOX_PADDING` pixels).
- `cv2.inpaint()` with `INPAINT_TELEA` reconstructs the background.
- Unlike blurring, Telea inpainting propagates surrounding texture and
  colour to fill the erased regions.

### 4 — Text Rendering (TextRenderer)

- For each detection, the auto-fitter tries font sizes from `FONT_SIZE_MAX`
  down to `FONT_SIZE_MIN`.
- Text is word-wrapped to fit the box width.
- The final text block is centred both horizontally and vertically inside
  the original bbox.
- If no TTF font is found, Pillow's built-in bitmap font is used.

---

## Configuration Reference

| Variable          | Default                                         | Description                        |
|-------------------|-------------------------------------------------|------------------------------------|
| `MODEL_PATH`      | `./models/Qwen2.5-VL-7B-Instruct-int8-ov`      | Path to OV INT8 model directory    |
| `OPENVINO_DEVICE` | `GPU`                                           | `GPU` / `CPU` / `AUTO`             |
| `INPAINT_RADIUS`  | `5`                                             | OpenCV inpaint brush radius        |
| `BBOX_PADDING`    | `2`                                             | Extra pixels around each bbox      |
| `FONT_PATH`       | `./fonts/DejaVuSans.ttf`                        | TrueType font for rendering        |
| `FONT_SIZE_MAX`   | `40`                                            | Largest font size tried            |
| `FONT_SIZE_MIN`   | `6`                                             | Smallest font size before giving up|

---

## Supported Languages

The TranslationService maps language names to ISO codes for 80+ languages
including: English, French, German, Spanish, Italian, Portuguese, Russian,
Chinese (Simplified/Traditional), Japanese, Korean, Arabic, Hindi, and many more.

See `services/translation_service.py` → `_LANGUAGE_MAP` for the full list.

---

## Logging

Every request produces structured log lines:

```
2026-09-25 18:00:01 | INFO     | main | Request received | source=English | target=French | file=input.png
2026-09-25 18:00:01 | INFO     | main | Image dimensions: 1920x1080
2026-09-25 18:00:06 | INFO     | services.ocr_service | OCR | 15 valid detections
2026-09-25 18:00:06 | INFO     | main | OCR detections: 15 | time: 4.21s
2026-09-25 18:00:07 | INFO     | main | Translation time: 0.85s
2026-09-25 18:00:07 | INFO     | main | Inpainting time: 0.03s
2026-09-25 18:00:07 | INFO     | main | Rendering time: 0.01s
2026-09-25 18:00:07 | INFO     | main | SUMMARY | detections=15 | ocr=4.21s | translation=0.85s | inpaint=0.03s | render=0.01s | total=5.10s
```

---

## Error Handling

| Error                        | HTTP Status | Response                              |
|------------------------------|-------------|---------------------------------------|
| Missing image                | 422         | `"Uploaded image is empty"`           |
| Invalid image format         | 422         | `"Cannot open image: ..."`            |
| Missing source_language      | 422         | `"source_language is required"`       |
| Missing target_language      | 422         | `"target_language is required"`       |
| Model not found at startup   | 500 (crash) | Server won't start                    |
| Device unavailable           | 500 (crash) | Server won't start                    |
| OCR JSON parse failure       | 200 + `[]`  | Returns empty detections, not a crash |
| Invalid bbox                 | 200 (skip)  | Bad bbox skipped, others processed    |
| Translation failure          | 200 (orig)  | Returns original text, not a crash    |
| Output file not found        | 404         | Standard 404                          |
| Services not initialised     | 503         | `"Services not initialised"`          |
