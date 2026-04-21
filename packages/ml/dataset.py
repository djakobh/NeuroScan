from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(data_root: str, batch_size: int = 32, val_split: float = 0.15, num_workers: int = 4):
    """
    Returns train, val, and test DataLoaders.
    Expects data_root to contain Training/ and Testing/ subdirectories,
    each with one folder per class.
    """
    data_root = Path(data_root)
    train_dir = data_root / "Training"
    test_dir = data_root / "Testing"

    # Load the folder twice so each split can have its own transform
    train_base = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORMS)
    val_base = datasets.ImageFolder(train_dir, transform=EVAL_TRANSFORMS)

    val_size = int(len(train_base) * val_split)
    train_size = len(train_base) - val_size

    # Use the same random split indices for both bases so they don't overlap
    import torch
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(len(train_base)), [train_size, val_size], generator=generator)

    from torch.utils.data import Subset
    train_dataset = Subset(train_base, list(train_indices))
    val_dataset = Subset(val_base, list(val_indices))

    test_dataset = datasets.ImageFolder(test_dir, transform=EVAL_TRANSFORMS)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
