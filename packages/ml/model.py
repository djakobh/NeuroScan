import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4


def build_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = False) -> nn.Module:
    """
    EfficientNetB0 pretrained on ImageNet with a custom classifier head.

    freeze_backbone=True: only the classifier head trains (fast, less data needed)
    freeze_backbone=False: full fine-tune (better accuracy with sufficient data)
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


if __name__ == "__main__":
    model = build_model()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape:     {out.shape}")
