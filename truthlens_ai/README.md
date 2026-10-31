# TruthLens AI — Image Authenticity Assessment

A runnable full-stack research prototype based on the uploaded TruthLens HTML. It combines a FastAPI backend, a Hugging Face image-classification model, file metadata inspection, and a responsive browser UI.

## Important accuracy note

No image detector can prove that an image is real or AI-generated. This project **does not combine EXIF, dimensions, compression or pixel noise into a fake probability**. The classifier output is shown as a model score, not as a calibrated probability of authenticity. Metadata is displayed as separate clues. C2PA/JUMBF marker presence is only a marker check; this project does **not** cryptographically validate C2PA manifests.

The default model is `aaronkantrowitz/ai-human-generated-image-detection`. Its model card says it was fine-tuned on the CIFAKE dataset and warns that performance on newer generators (including Flux, Midjourney v6, DALL-E 3 and Stable Diffusion 3) may vary. See the model card before using it: https://huggingface.co/aaronkantrowitz/ai-human-generated-image-detection . Treat this as a baseline classifier, not forensic-grade detection. You can set `HF_MODEL_ID` to another compatible Hugging Face image-classification model after checking its labels, training data, license and limitations.

## Requirements

- Python 3.10 or 3.11 recommended (Python 3.12 may also work depending on PyTorch wheels).
- 8 GB RAM minimum recommended; 16 GB is more comfortable.
- Internet connection for the first model download. The model is cached locally afterward.
- CPU works but inference may be slower. A supported NVIDIA GPU can accelerate inference.

## Windows setup

1. Extract the ZIP to a folder such as `C:\Projects\truthlens_ai`.
2. Install Python 3.11 and ensure the Python launcher `py` is available.
3. Double-click `run_windows.bat` or run these commands in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

4. Open **`http://127.0.0.1:8000/`** in Chrome or Edge. The FastAPI server now serves the frontend and API from the same origin, preventing the browser `Failed to fetch` / CORS issue.
5. Keep the backend terminal window open while using the app. You can also visit `http://127.0.0.1:8000/docs` to inspect the API.

**Important:** Do not double-click `frontend/index.html` for normal use. Open the application through `http://127.0.0.1:8000/` so the frontend and backend share the same origin.

## macOS / Linux

```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

Open `http://127.0.0.1:8000/` in your browser. The backend serves the frontend from the same origin.

## API

- `GET /api/health` — backend health and configured model ID.
- `POST /api/analyze` — multipart image upload and analysis.
- `POST /detect` — compatibility alias for the original uploaded HTML endpoint.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze -F "file=@sample.jpg"
```

## Configure a different model

Set the environment variable before starting the backend:

```powershell
$env:HF_MODEL_ID="your-org/your-image-classification-model"
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

The model must be compatible with the Transformers `image-classification` pipeline and expose recognizable class labels (for example `Real`/`Fake`, `human`/`AI-generated`). Unknown labels are reported as inconclusive.

## Project structure

```text
truthlens_ai/
├── backend/
│   └── main.py
├── frontend/
│   └── index.html
├── tests/
├── .env.example
├── requirements.txt
├── run_windows.bat
├── run_mac_linux.sh
└── README.md
```

## Current scope and known limitations

- Single-image classification only; no video/audio deepfake analysis.
- No user accounts, database, history storage or feedback-learning pipeline.
- No cryptographic C2PA manifest validation.
- No claim of forensic-grade accuracy or universal detector performance.
- Uploaded images are processed in memory and are not written to disk by the application. Avoid exposing this development server publicly without authentication, rate limiting, HTTPS, and privacy/security review.
- The default model's output can be unreliable on image types outside its training distribution. Evaluate it on a held-out dataset representing your intended use before making performance claims.

## Model attribution

Default model: `aaronkantrowitz/ai-human-generated-image-detection` (Hugging Face). Review and comply with the model's license and attribution requirements. This repository does not redistribute model weights.
