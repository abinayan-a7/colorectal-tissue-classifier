"""Training loop and checkpoint management for HistoCNN."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path if invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from src.dataset import build_datasets, make_loaders
from src.evaluate import classification_report
from src.model import build_model


def train_model(
    train_loader,
    val_loader,
    epochs: int = 16,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-4,
    use_cosine_lr: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """Train HistoCNN from scratch and return (best_model, history).

    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        epochs: Number of training epochs (default: 16).
        lr: Learning rate for AdamW (default: 1.2e-3).
        weight_decay: L2 regularization for AdamW (default: 1e-4).
        use_cosine_lr: Whether to use CosineAnnealingLR (default: True).
        device: Target torch device (CPU or CUDA).

    Returns:
        (best_model, history) where history is a list of per-epoch metric dicts.
    """
    torch.manual_seed(42)
    model = build_model(9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if use_cosine_lr
        else None
    )

    best_val_macro_recall = -1.0
    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    print(f"Beginning training on device: {device} | Total Epochs: {epochs}")
    print(f"Hyperparameters: lr={lr}, weight_decay={weight_decay}, cosine_lr={use_cosine_lr}")
    print("-" * 75)

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

        if scheduler is not None:
            scheduler.step()

        train_loss = train_loss_sum / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        model.eval()
        val_loss_sum, val_total = 0.0, 0
        all_val_preds, all_val_targets = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                bs = labels.size(0)
                val_loss_sum += loss.item() * bs
                val_total += bs
                all_val_preds.append(logits.argmax(dim=1).cpu())
                all_val_targets.append(labels.cpu())

        val_loss = val_loss_sum / val_total if val_total > 0 else 0.0
        y_pred = torch.cat(all_val_preds)
        y_true = torch.cat(all_val_targets)
        report = classification_report(y_true, y_pred)
        val_acc = report["accuracy"]
        val_macro_recall = report["macro_recall"]

        is_best = val_macro_recall > best_val_macro_recall
        if is_best:
            best_val_macro_recall = val_macro_recall
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            star = " ★ best"
        else:
            star = ""

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_macro_recall": float(val_macro_recall),
        })

        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc * 100:.2f}% MacroRecall: {val_macro_recall * 100:.2f}%{star}"
        )

    print("-" * 75)
    print(
        f"Restoring best checkpoint (Best Val Macro Recall: {best_val_macro_recall * 100:.2f}%, Val Acc: {best_val_acc * 100:.2f}%)"
    )
    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HistoCNN from scratch on NCT-CRC-HE.")
    parser.add_argument("--data-dir", type=str, default=None, help="Local directory containing train/validation folders")
    parser.add_argument("--epochs", type=int, default=16, help="Number of training epochs (default: 16)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Learning rate (default: 1.2e-3)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--use-cosine-lr", action="store_true", default=True, help="Use CosineAnnealingLR scheduler")
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
        weight_decay=args.weight_decay,
        use_cosine_lr=args.use_cosine_lr,
        device=device,
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)

    history_path = os.path.join(os.path.dirname(args.save_path), "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved best model weights to: {args.save_path}")
    print(f"Saved training history to  : {history_path}")
