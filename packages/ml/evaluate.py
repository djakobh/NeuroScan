"""
Evaluate the saved model on the held-out test set.

Usage:
    python evaluate.py --data_root ../archive --checkpoint checkpoints/best_model.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from dataset import get_dataloaders, CLASSES
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="../archive")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", default="checkpoints")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    cm = confusion_matrix(all_labels, all_preds)
    output_dir = Path(args.output_dir)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()

    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
