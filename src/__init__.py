"""HistoCNN: Lightweight Colorectal Histopathology Patch Classification.

A modular PyTorch implementation of a 4-block custom Convolutional Neural Network
for classifying 9 colorectal tissue types on the NCT-CRC-HE dataset.
"""

from src.dataset import (
    HFSubsetDataset,
    build_datasets,
    make_loaders,
    make_transforms,
    official_classes,
)
from src.evaluate import classification_report, load_pretrained, predict
from src.model import HistoCNN, build_model

__all__ = [
    "HistoCNN",
    "build_model",
    "official_classes",
    "make_transforms",
    "HFSubsetDataset",
    "build_datasets",
    "make_loaders",
    "predict",
    "classification_report",
    "load_pretrained",
]
