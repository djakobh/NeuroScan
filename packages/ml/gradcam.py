"""
Generate Grad-CAM heatmap overlays for brain tumor MRI images.

Usage:
    # Single image
    python gradcam.py --image ../archive/Testing/glioma/Te-gl_0010.jpg

    # Entire folder (saves one output image per input)
    python gradcam.py --folder ../archive/Testing/glioma --output_dir checkpoints/gradcam_samples
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from dataset import CLASSES, IMG_SIZE
from model import build_model

# Same normalization used during training
EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint: str, device: torch.device):
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    return model


def preprocess(image_path: str):
    """Returns (input_tensor [1,3,H,W], rgb_float_array [H,W,3] in [0,1])."""
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    rgb = np.array(img_resized, dtype=np.float32) / 255.0  # needed for overlay

    tensor = EVAL_TRANSFORMS(img).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor, rgb


def run_gradcam(model, tensor, device, target_class=None):
    """
    Runs Grad-CAM and returns:
      - predicted class index
      - confidence scores (softmax)
      - grayscale CAM array [H, W] in [0, 1]
    """
    target_layer = [model.features[-1]]

    with GradCAM(model=model, target_layers=target_layer) as cam:
        tensor = tensor.to(device)
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().detach().numpy()
        pred_class = int(np.argmax(probs))

        target = target_class if target_class is not None else pred_class
        grayscale_cam = cam(
            input_tensor=tensor,
            targets=[ClassifierOutputTarget(target)],
        )[0]  # shape [H, W]

    return pred_class, probs, grayscale_cam


def build_heatmap_only(rgb_float, grayscale_cam):
    """Returns just the Grad-CAM overlay as a uint8 RGB array [H, W, 3]."""
    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)


def build_output_image(rgb_float, grayscale_cam, pred_class, probs):
    """
    Builds a side-by-side image:
      Left:  original MRI
      Right: MRI with Grad-CAM heatmap overlay + prediction label
    """
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # Draw prediction text onto the overlay
    label = f"{CLASSES[pred_class]} ({probs[pred_class]*100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.putText(overlay_bgr, label, (8, 24), font, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(overlay_bgr, label, (8, 24), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    # Original MRI (convert float [0,1] back to uint8)
    original = (rgb_float * 255).astype(np.uint8)

    # Side-by-side
    combined = np.concatenate([original, overlay], axis=1)
    return combined


def process_image(image_path: str, model, device, output_path: str = None):
    tensor, rgb = preprocess(image_path)
    pred_class, probs, grayscale_cam = run_gradcam(model, tensor, device)
    combined = build_output_image(rgb, grayscale_cam, pred_class, probs)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(combined).save(out)
        print(f"Saved: {out}")

    print(f"  Image : {Path(image_path).name}")
    print(f"  Prediction : {CLASSES[pred_class]} ({probs[pred_class]*100:.1f}%)")
    print(f"  Confidences: " + " | ".join(f"{c}: {p*100:.1f}%" for c, p in zip(CLASSES, probs)))

    return combined, pred_class, probs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="Path to a single MRI image")
    p.add_argument("--folder", help="Path to a folder of MRI images")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--output_dir", default="checkpoints/gradcam_samples")
    p.add_argument("--limit", type=int, default=10,
                   help="Max images to process when using --folder")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    if args.image:
        stem = Path(args.image).stem
        out = str(Path(args.output_dir) / f"{stem}_gradcam.png")
        process_image(args.image, model, device, output_path=out)

    elif args.folder:
        folder = Path(args.folder)
        image_paths = [
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ][:args.limit]

        if not image_paths:
            print(f"No images found in {folder}")
            return

        print(f"Processing {len(image_paths)} images from {folder}\n")
        for img_path in image_paths:
            out = str(Path(args.output_dir) / f"{img_path.stem}_gradcam.png")
            process_image(str(img_path), model, device, output_path=out)
            print()

    else:
        print("Provide --image <path> or --folder <path>")


if __name__ == "__main__":
    main()
