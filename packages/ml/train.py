"""
Train the brain tumor classifier.

Usage:
    python train.py --data_root ../archive --epochs 20 --batch_size 32 --lr 1e-4
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="../archive")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze EfficientNet backbone; only train classifier head")
    return p.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    model = build_model(freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining for {args.epochs} epochs\n{'='*50}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, training=False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")

    torch.save(model.state_dict(), checkpoint_dir / "last_model.pth")
    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir.resolve()}")

    _save_training_plot(history, checkpoint_dir / "training_curves.png")


def _save_training_plot(history, path):
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        epochs = range(1, len(history["train_loss"]) + 1)

        ax1.plot(epochs, history["train_loss"], label="Train")
        ax1.plot(epochs, history["val_loss"], label="Val")
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax1.legend()

        ax2.plot(epochs, history["train_acc"], label="Train")
        ax2.plot(epochs, history["val_acc"], label="Val")
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(path)
        print(f"Training curves saved to: {path}")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    main()
