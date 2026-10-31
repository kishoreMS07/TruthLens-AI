from __future__ import annotations

import io
import os
import re
import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ExifTags, UnidentifiedImageError

APP_NAME = "TruthLens AI"
MODEL_ID = os.getenv("HF_MODEL_ID", "aaronkantrowitz/ai-human-generated-image-detection")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))
MAX_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff", "image/bmp"}

app = FastAPI(title=APP_NAME, version="1.0.0", description="Image authenticity assessment with an optional Hugging Face vision classifier.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


def _model_pipeline():
    """Load the configured image classifier once. Model download occurs on first analysis."""
    try:
        from transformers import pipeline
        import torch
    except Exception as exc:
        raise RuntimeError("Model dependencies are not installed. Install requirements.txt first.") from exc
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-classification", model=MODEL_ID, device=device)


@lru_cache(maxsize=1)
def get_detector():
    return _model_pipeline()


def _safe_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (label or "").lower()).strip()


def _map_prediction(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Map common model label conventions without pretending scores are calibrated probabilities."""
    ai_terms = ("ai", "fake", "synthetic", "generated", "artificial")
    human_terms = ("real", "human", "authentic", "natural")
    normalized = []
    for item in predictions:
        label = str(item.get("label", "unknown"))
        score = float(item.get("score", 0.0))
        normalized.append({"label": label, "score": max(0.0, min(1.0, score))})
    ai_score = sum(x["score"] for x in normalized if any(t in _safe_label(x["label"]) for t in ai_terms))
    human_score = sum(x["score"] for x in normalized if any(t in _safe_label(x["label"]) for t in human_terms))
    if ai_score == 0 and human_score == 0:
        return {"status": "inconclusive", "label": "Model output could not be interpreted", "score": None, "raw": normalized}
    if ai_score >= human_score:
        return {"status": "ai_indicated", "label": "AI-generated indicators detected", "score": round(ai_score, 4), "raw": normalized}
    return {"status": "human_indicated", "label": "Human-created indicators detected", "score": round(human_score, 4), "raw": normalized}


def _metadata(image: Image.Image, raw: bytes) -> dict[str, Any]:
    exif_data: dict[str, Any] = {}
    try:
        exif = image.getexif()
        for key, value in exif.items():
            name = ExifTags.TAGS.get(key, str(key))
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")[:200]
            else:
                value = str(value)[:200]
            exif_data[name] = value
    except Exception:
        pass
    info = {str(k): str(v)[:200] for k, v in image.info.items() if isinstance(k, str)}
    generator_patterns = ["midjourney", "stable diffusion", "dall-e", "dalle", "firefly", "comfyui", "novelai", "leonardo", "imagen"]
    searchable = " ".join(list(exif_data.values()) + list(info.values()) + [raw[:300000].decode("latin-1", "ignore")])[:500000].lower()
    generator_hints = sorted({p for p in generator_patterns if p in searchable})
    camera_fields = [k for k in ("Make", "Model", "LensModel", "DateTimeOriginal") if k in exif_data]
    c2pa_hint = b"c2pa" in raw[:2_000_000].lower() or b"jumbf" in raw[:2_000_000].lower()
    return {
        "exif": exif_data,
        "container_metadata": info,
        "camera_metadata_fields": camera_fields,
        "generator_metadata_hints": generator_hints,
        "c2pa_manifest_marker_present": c2pa_hint,
        "warning": "Metadata can be removed, edited or copied. A marker is not cryptographic C2PA validation."
    }


def _forensic_observations(image: Image.Image) -> dict[str, Any]:
    # Deliberately descriptive only: these weak signals do not contribute to the model result.
    width, height = image.size
    return {
        "dimensions": {"width": width, "height": height},
        "format": image.info.get("_source_format", image.format),
        "mode": image.mode,
        "aspect_ratio": round(width / height, 4) if height else None,
        "observations": [
            "Image dimensions and compression characteristics are not reliable proof of origin.",
            "A model classification can fail on unseen generators, edits, screenshots and recompressed images."
        ]
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": APP_NAME, "version": "1.0.0", "model_id": MODEL_ID, "model_loaded": get_detector.cache_info().currsize > 0}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported image type. Upload JPG, PNG, WebP, TIFF or BMP.")
    raw = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {MAX_UPLOAD_MB} MB upload limit.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        source_format = Image.open(io.BytesIO(raw)).format
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image.info["_source_format"] = source_format or "unknown"
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The file could not be decoded as a supported image.")
    if image.width * image.height > MAX_PIXELS:
        raise HTTPException(status_code=413, detail=f"Image dimensions exceed the {MAX_PIXELS:,}-pixel safety limit.")

    metadata = _metadata(image, raw)
    observations = _forensic_observations(image)
    try:
        detector = get_detector()
        model_output = _map_prediction(detector(image, top_k=None))
        model_error = None
    except Exception as exc:
        model_output = {"status": "unavailable", "label": "Classifier unavailable", "score": None, "raw": []}
        model_error = str(exc)[:500]

    digest = hashlib.sha256(raw).hexdigest()
    if model_output["status"] == "ai_indicated":
        verdict = "AI-generated indicators detected"
    elif model_output["status"] == "human_indicated":
        verdict = "Human-created indicators detected"
    else:
        verdict = "Inconclusive"

    return {
        "filename": file.filename,
        "sha256": digest,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "model": {"id": MODEL_ID, **model_output, "error": model_error},
        "metadata": metadata,
        "image": observations,
        "interpretation": "This is a model-assisted assessment, not proof of authenticity. The displayed model score is the classifier's output score, not a calibrated probability that the image is real or AI-generated.",
        "limitations": [
            "Performance depends on the model's training data and may be poor on newer image generators or out-of-domain images.",
            "Editing, resizing, screenshots and social-media recompression can change model behavior.",
            "Metadata and C2PA marker presence are clues only; this application does not cryptographically validate C2PA manifests.",
            "Do not use this result alone for moderation, journalism, legal, employment or other high-impact decisions."
        ]
    }

# Backwards-compatible endpoint used by the original uploaded HTML.
@app.post("/detect")
async def detect_legacy(file: UploadFile = File(...)):
    return await analyze(file)


# Serve the frontend from the same origin as the API. This avoids browser CORS / file:// fetch failures.
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
