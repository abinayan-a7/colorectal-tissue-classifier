"""Training loop and checkpoint management for HistoCNN."""

import argparse
import copy
import json
import os
import torch
import torch.nn as nn
from src.dataset import build_datasets, make_loaders
from src.model import build_model


def train_model(
    train_loader,
    val_loader,
    epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
):
    """Train HistoCNN from scratch and return (best_model, history).

    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        epochs: Number of training epochs (default: 12).
        lr: Learning rate for AdamW (default: 1e-3).
        weight_decay: L2 regularization for AdamW (default: 1e-4).
        device: Target torch device (CPU or CUDA).

    Returns:
        (best_model, history) where history is a list of per-epoch metric dicts.
    """
    torch.manual_seed(42)
    model = build_model(9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []

    print(f"Beginning training on device: {device} | Total Epochs: {epochs}")
    print("-" * 65)

    for epoch in range(epochs):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            train_loss_sum += loss.item() * bs
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += bs

        train_loss = train_loss_sum / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                bs = labels.size(0)
                val_loss_sum += loss.item() * bs
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += bs

        val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        if val_acc > best_val_acc or (
            abs(val_acc - best_val_acc) < 1e-12 and val_loss < best_val_loss
        ):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        })

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc * 100:.2f}%"
        )

    print("-" * 65)
    print(f"Restoring best checkpoint (Val Acc: {best_val_acc * 100:.2f}%, Val Loss: {best_val_loss:.4f})")
    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HistoCNN from scratch on NCT-CRC-HE.")
    parser.add_argument("--data-dir", type=str, default=None, help="Local directory containing train/validation folders")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs (default: 12)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--save-path", type=str, default="weights/histocnn_best.pth", help="Output weights file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Preparing datasets...")
    train_ds, val_ds = build_datasets(data_dir=args.data_dir)
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size=args.batch_size)

    model, history = train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)

    history_path = os.path.join(os.path.dirname(args.save_path), "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved best model weights to: {args.save_path}")
    print(f"Saved training history to  : {history_path}")
