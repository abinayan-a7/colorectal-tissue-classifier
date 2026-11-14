"""Data loading and preprocessing pipelines for NCT-CRC-HE histopathology patches."""

import os
import random
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as tv_datasets
import torchvision.transforms as T


def official_classes() -> list[str]:
    """Return the 9 colorectal tissue classes in exact index order (0-8).

    0: ADI  - Adipose tissue
    1: BACK - Background / clear glass
    2: DEB  - Debris
    3: LYM  - Lymphocytes
    4: MUC  - Mucus
    5: MUS  - Smooth muscle
    6: NORM - Normal colon mucosa
    7: STR  - Cancer-associated stroma
    8: TUM  - Colorectal adenocarcinoma epithelium
    """
    return ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def make_transforms(train: bool) -> T.Compose:
    """Build torchvision transform pipeline for 64x64 H&E patches.

    Args:
        train: If True, applies random horizontal & vertical flips for data augmentation.
               If False, applies deterministic resize and normalization.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    if train:
        return T.Compose([
            T.Resize((64, 64)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class HFSubsetDataset(Dataset):
    """Wraps a list of Hugging Face {'image': PIL.Image, 'label': int} dictionaries."""

    def __init__(self, items: list[dict], transform: T.Compose | None = None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = item["image"]
        label = item["label"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def build_datasets(data_dir: str | None = None, per_class: int = 1000):
    """Build train and validation datasets.

    Uses local ImageFolder if data_dir (or CAME_DATA_DIR environment variable) is provided;
    otherwise streams a balanced subset from Hugging Face (`1aurent/NCT-CRC-HE`).

    Args:
        data_dir: Optional path to a local directory with 'train' and 'validation' folders.
        per_class: Number of samples per class when streaming from Hugging Face.

    Returns:
        (train_dataset, val_dataset) tuple.
    """
    data_dir = data_dir or os.environ.get("CAME_DATA_DIR")

    if data_dir:
        train_dataset = tv_datasets.ImageFolder(
            os.path.join(data_dir, "train"), transform=make_transforms(True)
        )
        val_dataset = tv_datasets.ImageFolder(
            os.path.join(data_dir, "validation"), transform=make_transforms(False)
        )
        return train_dataset, val_dataset

    from datasets import load_dataset

    num_classes = len(official_classes())
    ds = load_dataset(
        "1aurent/NCT-CRC-HE", split="NCT_CRC_HE_100K", streaming=True
    )

    class_buckets: dict[int, list[dict]] = {c: [] for c in range(num_classes)}
    for item in ds:
        label = item["label"]
        if len(class_buckets[label]) < per_class:
            class_buckets[label].append({"image": item["image"], "label": label})
        if all(len(v) >= per_class for v in class_buckets.values()):
            break

    all_items = []
    for c in range(num_classes):
        all_items.extend(class_buckets[c])

    random.seed(42)
    random.shuffle(all_items)

    split_idx = int(len(all_items) * 0.85)
    train_dataset = HFSubsetDataset(all_items[:split_idx], make_transforms(True))
    val_dataset = HFSubsetDataset(all_items[split_idx:], make_transforms(False))
    return train_dataset, val_dataset


def make_loaders(
    train_dataset,
    val_dataset,
    batch_size: int = 32,
    num_workers: int | None = None,
):
    """Return (train_loader, val_loader) PyTorch DataLoaders.

    Args:
        train_dataset: PyTorch Dataset for training.
        val_dataset: PyTorch Dataset for validation.
        batch_size: Batch size (default: 32).
        num_workers: Number of DataLoader worker subprocesses (default: env CAME_WORKERS or 0).

    Returns:
        (train_loader, val_loader) tuple.
    """
    if num_workers is None:
        num_workers = int(os.environ.get("CAME_WORKERS", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
