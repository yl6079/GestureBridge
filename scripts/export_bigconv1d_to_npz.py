"""Convert a BigConv1D PyTorch checkpoint into a single numpy .npz so the Pi
runtime can load it without PyTorch.

The npz stores each Module's parameters under flat keys (e.g.
`b1.c1.weight`) plus auxiliary tensors needed for BatchNorm eval-mode
inference (`running_mean`, `running_var`). It also stores
`__arch__='bigconv1d'` and `__num_classes__` for the loader.

Usage:
    python scripts/export_bigconv1d_to_npz.py \
      --ckpt artifacts/wlasl100_a100_conv1d/ckpts/best.pt \
      --out  artifacts/wlasl100_a100_conv1d/bigconv1d_s42.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_conv1d_a100 import BigConv1D  # type: ignore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num-classes", type=int, default=100)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"]

    # Build a model instance just to validate that all keys load cleanly.
    m = BigConv1D(num_classes=args.num_classes)
    m.load_state_dict(state)
    m.eval()

    weights: dict[str, np.ndarray] = {}
    for k, v in m.state_dict().items():
        # state_dict includes BN running_mean/running_var/num_batches_tracked
        weights[k] = v.detach().cpu().numpy().copy()
    weights["__arch__"] = np.array(["bigconv1d"], dtype="<U32")
    weights["__num_classes__"] = np.array([args.num_classes], dtype=np.int32)
    weights["__input_shape__"] = np.array([30, 63], dtype=np.int32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **weights)
    print(f"saved {args.out}  ({len(weights)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
