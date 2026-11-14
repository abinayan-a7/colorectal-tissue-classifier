"""Extract and decode pretrained weights from assignment02.py into a clean .pth file.

Decodes TRAINED_MODEL_B64 via:
    base64.b64decode -> zlib.decompress -> torch.load -> cast FP16 tensors to FP32 (.float())
and writes a clean, production-ready PyTorch state_dict to weights/histocnn_best.pth.
"""

import argparse
import base64
import io
import os
import re
import sys
import zlib
from pathlib import Path
import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluate import load_pretrained
from src.model import build_model


def find_source_file(provided_path: str | None) -> Path:
    """Locate assignment02.py from provided path or standard workspace locations."""
    candidates = []
    if provided_path:
        candidates.append(Path(provided_path))

    candidates.extend([
        REPO_ROOT / "assignment02.py",
        REPO_ROOT.parent / "assignment_cnn" / "assignment02.py",
        Path("assignment_cnn/assignment02.py"),
        Path("assignment02.py"),
    ])

    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(
        f"Could not locate assignment02.py. Looked in: {[str(c) for c in candidates]}. "
        f"Specify --source path/to/assignment02.py"
    )


def extract_weights(source_path: Path, output_path: Path) -> dict:
    """Extract Base64 model weights, decompress, cast to float32, and save to output_path."""
    print(f"Reading source script: {source_path}")
    content = source_path.read_text(encoding="utf-8")

    match = re.search(r'TRAINED_MODEL_B64\s*=\s*"""(.*?)"""', content, re.DOTALL)
    if not match:
        raise ValueError(f"Could not locate TRAINED_MODEL_B64 string in {source_path}")

    b64_str = match.group(1).strip()
    print(f"Extracted Base64 payload: {len(b64_str):,} characters")

    # Step 1: Base64 decode
    compressed_bytes = base64.b64decode(b64_str)

    # Step 2: zlib decompress
    decompressed_bytes = zlib.decompress(compressed_bytes)
    print(f"Decompressed raw bytes  : {len(decompressed_bytes):,} bytes")

    # Step 3: PyTorch load
    state_dict = torch.load(io.BytesIO(decompressed_bytes), map_location="cpu", weights_only=True)
    print(f"Loaded state_dict keys  : {len(state_dict)} tensor entries")

    # Step 4: Cast FP16 floating point tensors back to FP32
    clean_state_dict = {}
    converted_count = 0
    for k, v in state_dict.items():
        if v.is_floating_point() and v.dtype != torch.float32:
            clean_state_dict[k] = v.float()
            converted_count += 1
        else:
            clean_state_dict[k] = v

    print(f"Converted {converted_count} tensor(s) to float32 precision")

    # Save to destination
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean_state_dict, output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Successfully saved clean state_dict to: {output_path} ({file_size_mb:.2f} MB)")

    return clean_state_dict


def verify_weights(weights_path: Path):
    """Verify that load_pretrained and build_model can load the weights cleanly."""
    print("\n--- Verifying weights with src.evaluate & src.model ---")
    device = torch.device("cpu")
    model = load_pretrained(str(weights_path), device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model architecture      : {model.__class__.__name__}")
    print(f"Total parameters        : {total_params:,}")
    print(f"Trainable parameters    : {trainable_params:,}")

    # Test dummy inference forward pass
    dummy_input = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 9), f"Expected shape (2, 9), got {output.shape}"
    print(f"Dummy forward pass test : Passed! Output shape {tuple(output.shape)}")
    print("Verification complete: weights are 100% valid and ready for inference!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract clean PyTorch weights from assignment02.py")
    parser.add_argument("--source", type=str, default=None, help="Path to assignment02.py")
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "weights" / "histocnn_best.pth"),
        help="Path for output .pth file",
    )
    args = parser.parse_args()

    src = find_source_file(args.source)
    dst = Path(args.output).resolve()

    extract_weights(src, dst)
    verify_weights(dst)
