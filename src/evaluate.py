"""Inference, evaluation metrics, and pretrained weight loading for HistoCNN."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path if invoked directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.dataset import build_datasets, make_loaders, official_classes
from src.model import build_model


def predict(model: torch.nn.Module, loader, device: torch.device) -> torch.Tensor:
    """Run inference and return a 1-D CPU torch.long tensor of predicted class indices."""
    model.eval()
    model.to(device)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            logits = model(images)
            all_preds.append(logits.argmax(dim=1).cpu())
    return torch.cat(all_preds).long()


def classification_report(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = 9) -> dict:
    """Compute overall accuracy, macro recall, and a (9, 9) confusion matrix.

    Args:
        y_true: 1-D torch.Tensor of ground truth class indices.
        y_pred: 1-D torch.Tensor of predicted class indices.
        num_classes: Number of distinct classes (default: 9).

    Returns:
        dict with:
            - 'accuracy': float
            - 'macro_recall': float
            - 'confusion': torch.Tensor of shape (num_classes, num_classes) and dtype torch.long
    """
    y_true = y_true.long()
    y_pred = y_pred.long()

    total = y_true.size(0)
    accuracy = (y_true == y_pred).sum().item() / total if total > 0 else 0.0

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        confusion[t.item(), p.item()] += 1

    recalls = []
    for c in range(num_classes):
        true_c = (y_true == c).sum().item()
        if true_c > 0:
            tp = ((y_true == c) & (y_pred == c)).sum().item()
            recalls.append(tp / true_c)
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "accuracy": float(accuracy),
        "macro_recall": float(macro_recall),
        "confusion": confusion,
    }


def load_pretrained(weights_path: str, device: torch.device = torch.device("cpu")) -> torch.nn.Module:
    """Load a saved state_dict (.pth) into HistoCNN.

    Args:
        weights_path: Path to the .pth state_dict file.
        device: Target torch device (CPU or CUDA).

    Returns:
        HistoCNN model in eval mode on the requested device.
    """
    model = build_model(len(official_classes()))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    # Cast float16 tensors back to float32 if compressed
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HistoCNN on validation data.")
    parser.add_argument("--weights", type=str, default="weights/histocnn_best.pth", help="Path to weights file")
    parser.add_argument("--data-dir", type=str, default=None, help="Local data directory with train/validation folders")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading weights from: {args.weights}")
    model = load_pretrained(args.weights, device)
    print("Pretrained HistoCNN loaded successfully!")

    print("Building validation dataset...")
    try:
        _, val_dataset = build_datasets(data_dir=args.data_dir)
        _, val_loader = make_loaders(val_dataset, val_dataset, batch_size=args.batch_size)
    except Exception as e:
        print(f"Could not load dataset ({e}). Specify local folder with --data-dir.")
        sys.exit(1)

    print("Running inference...")
    preds = predict(model, val_loader, device)
    y_true = torch.cat([labels for _, labels in val_loader])

    report = classification_report(y_true, preds)
    classes = official_classes()

    print("\n" + "=" * 55)
    print("HistoCNN Validation Performance")
    print("=" * 55)
    print(f"  Overall Accuracy: {report['accuracy'] * 100:.2f}%")
    print(f"  Macro Recall    : {report['macro_recall'] * 100:.2f}%")
    print("\nConfusion Matrix (Rows: True, Columns: Predicted):")
    header = "       " + " ".join(f"{c:>6}" for c in classes)
    print(header)
    for i, row in enumerate(report["confusion"]):
        row_str = f"{classes[i]:<6} " + " ".join(f"{val:>6d}" for val in row.tolist())
        print(row_str)
    print("=" * 55)
