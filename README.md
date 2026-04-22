---
title: Neuroscan Ml
emoji: 🔥
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
short_description: MRI Classifier for Biology Class
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# NeuroScan ML

A brain tumor MRI classification app built with a deep convolutional neural network. Users can pull a random MRI scan from the test set, get a prediction, and see a Grad-CAM heatmap highlighting the regions that influenced the result.

Built as a biology class presentation project.

## Model

- Architecture: EfficientNetB0 (transfer learning)
- Dataset: Brain Tumor MRI Dataset — 7,000+ images, 4 classes (glioma, meningioma, pituitary, no tumor)
- Test accuracy: 95%

## Stack

- Model: PyTorch
- Backend: FastAPI, deployed on Hugging Face Spaces
- Frontend: React + Vite, deployed on Vercel

## Running locally

Backend:
```
cd src/api
../../venv/Scripts/uvicorn main:app --reload --port 8000
```

Frontend:
```
npm run dev
```

Set `VITE_API_URL=http://localhost:8000` in `.env`.
