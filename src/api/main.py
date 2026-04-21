"""
FastAPI backend for the brain tumor MRI classifier.

Endpoints:
    GET  /              — health check
    POST /predict       — returns predicted class + confidence scores
    POST /explain       — returns prediction + Grad-CAM heatmap as base64 PNG

Usage:
    uvicorn main:app --reload --port 8000
"""

import base64
import io
import random
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

# Make packages/ml/ importable from apps/api/
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "packages" / "ml"
sys.path.insert(0, str(MODEL_DIR))

from dataset import CLASSES                          # noqa: E402
from gradcam import build_heatmap_only, load_model, preprocess, run_gradcam  # noqa: E402

CHECKPOINT = MODEL_DIR / "checkpoints" / "best_model.pth"

# ---------------------------------------------------------------------------
# App state — model loaded once at startup, shared across all requests
# ---------------------------------------------------------------------------
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {CHECKPOINT} on {device} ...")
    state["model"] = load_model(str(CHECKPOINT), device)
    state["device"] = device
    print("Model ready.")
    yield
    state.clear()


app = FastAPI(
    title="Brain Tumor Classifier API",
    description="EfficientNetB0 classifier with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the React dev server (port 3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Sample-Class"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


def _validate_image(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Upload a JPEG or PNG.",
        )


async def _read_image(file: UploadFile):
    """Read uploaded bytes → (PIL Image, float RGB array [H,W,3])."""
    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    return pil_img


def _pil_to_tensor_and_rgb(pil_img: Image.Image):
    """Save PIL image to a temp buffer, then run the same preprocess as gradcam.py."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    # Write to a real temp file because preprocess() uses Image.open(path)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(buf.read())
        tmp_path = tmp.name

    try:
        tensor, rgb = preprocess(tmp_path)
    finally:
        os.unlink(tmp_path)

    return tensor, rgb


def _array_to_base64(img_array: np.ndarray) -> str:
    """Convert uint8 numpy array [H,W,3] → base64-encoded PNG string."""
    pil = Image.fromarray(img_array.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_confidence_dict(probs: np.ndarray) -> dict:
    return {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "model": "EfficientNetB0", "classes": CLASSES}


@app.get("/random-sample")
async def random_sample():
    """
    Returns a random image from the Testing dataset.

    Response headers:
        X-Sample-Class: the ground-truth class folder name (e.g. "glioma")
    """
    dataset_dir = MODEL_DIR.parent.parent / "data" / "Testing"
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Testing dataset not found on this server.")

    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise HTTPException(status_code=404, detail="No class folders found in Testing dataset.")

    chosen_class = random.choice(class_dirs)
    images = (
        list(chosen_class.glob("*.jpg"))
        + list(chosen_class.glob("*.jpeg"))
        + list(chosen_class.glob("*.png"))
    )
    if not images:
        raise HTTPException(status_code=404, detail=f"No images in {chosen_class.name}.")

    chosen = random.choice(images)
    media_type = "image/jpeg" if chosen.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    return FileResponse(
        path=str(chosen),
        media_type=media_type,
        headers={
            "X-Sample-Class": chosen_class.name,
            "Cache-Control": "no-store",
        },
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Returns the predicted class and confidence score for each of the 4 classes.

    Response:
        {
            "predicted_class": "glioma",
            "confidence": 0.712,
            "scores": {
                "glioma": 0.712,
                "meningioma": 0.183,
                "notumor": 0.061,
                "pituitary": 0.044
            }
        }
    """
    _validate_image(file)
    pil_img = await _read_image(file)
    tensor, _ = _pil_to_tensor_and_rgb(pil_img)

    pred_class, probs, _ = run_gradcam(state["model"], tensor, state["device"])

    return {
        "predicted_class": CLASSES[pred_class],
        "confidence": round(float(probs[pred_class]), 4),
        "scores": _build_confidence_dict(probs),
    }


@app.post("/explain")
async def explain(file: UploadFile = File(...)):
    """
    Returns prediction + a base64-encoded PNG of the Grad-CAM heatmap overlay.

    Response:
        {
            "predicted_class": "glioma",
            "confidence": 0.712,
            "scores": { ... },
            "heatmap_base64": "<base64 PNG string>"
        }
    """
    _validate_image(file)
    pil_img = await _read_image(file)
    tensor, rgb = _pil_to_tensor_and_rgb(pil_img)

    pred_class, probs, grayscale_cam = run_gradcam(state["model"], tensor, state["device"])
    heatmap = build_heatmap_only(rgb, grayscale_cam)
    heatmap_b64 = _array_to_base64(heatmap)

    return {
        "predicted_class": CLASSES[pred_class],
        "confidence": round(float(probs[pred_class]), 4),
        "scores": _build_confidence_dict(probs),
        "heatmap_base64": heatmap_b64,
    }
