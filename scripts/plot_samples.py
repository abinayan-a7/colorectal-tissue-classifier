"""Generate a 3x3 showcase grid of the 9 NCT-CRC-HE tissue phenotypes."""

import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image
from src.dataset import official_classes

CLASS_NAMES_FULL = {
    "ADI": "Adipose Tissue (ADI)",
    "BACK": "Background (BACK)",
    "DEB": "Debris (DEB)",
    "LYM": "Lymphocytes (LYM)",
    "MUC": "Mucus (MUC)",
    "MUS": "Smooth Muscle (MUS)",
    "NORM": "Normal Mucosa (NORM)",
    "STR": "Cancer Stroma (STR)",
    "TUM": "Adenocarcinoma (TUM)",
}


def fetch_one_per_class(data_dir: str | None = None) -> dict[int, Image.Image]:
    """Fetch 1 representative PIL image for each of the 9 classes (0-8)."""
    classes = official_classes()
    data_dir = data_dir or os.environ.get("CAME_DATA_DIR")
    samples = {}

    if data_dir:
        val_dir = os.path.join(data_dir, "validation")
        for idx, cls_name in enumerate(classes):
            cls_folder = os.path.join(val_dir, cls_name)
            if os.path.isdir(cls_folder):
                files = sorted(os.listdir(cls_folder))
                if files:
                    img_path = os.path.join(cls_folder, files[0])
                    samples[idx] = Image.open(img_path).convert("RGB")
        if len(samples) == len(classes):
            return samples

    from datasets import load_dataset

    ds = load_dataset("1aurent/NCT-CRC-HE", split="NCT_CRC_HE_100K", streaming=True)
    for item in ds:
        label = item["label"]
        if label not in samples:
            img = item["image"]
            samples[label] = img.convert("RGB") if img.mode != "RGB" else img
        if len(samples) == len(classes):
            break
    return samples


def plot_sample_grid(
    data_dir: str | None = None,
    save_path: str = "assets/sample_patches.png",
):
    """Render and save a 3x3 high-resolution grid of H&E tissue patches."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    classes = official_classes()
    samples = fetch_one_per_class(data_dir=data_dir)

    fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.5), dpi=300)
    fig.suptitle(
        "NCT-CRC-HE Colorectal Histopathology Phenotypes (H&E, 0.5 µm/px)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for idx, ax in enumerate(axes.flat):
        cls_code = classes[idx]
        ax.imshow(samples[idx])
        ax.set_title(
            f"[{idx}] {CLASS_NAMES_FULL[cls_code]}",
            fontsize=10,
            fontweight="bold",
            pad=6,
        )
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 3x3 sample patch grid to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot 3x3 grid of H&E tissue classes.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save-path", type=str, default="assets/sample_patches.png")
    args = parser.parse_args()

    plot_sample_grid(data_dir=args.data_dir, save_path=args.save_path)

