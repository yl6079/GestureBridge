"""Evaluate the A100 BigConv1D swarm individually + as a softmax ensemble.

Loads three PyTorch checkpoints and the existing Keras Conv1D + GRU
weights (npz, numpy forward) so we can quantify whether the A100 swarm
beats the prior production ensemble on the held-out 239-clip test set.

Usage:
    python scripts/eval_a100_ensemble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load BigConv1D class from the training script.
sys.path.insert(0, str(ROOT / "scripts"))
from train_conv1d_a100 import BigConv1D, WLASLSeqDataset  # type: ignore  # noqa: E402

from gesturebridge.pipelines.word_classifier import WordClassifier
from gesturebridge.pipelines.word_ensemble import GRUClassifier


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def topk_acc(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-probs, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def predict_pytorch(ckpt_path: Path, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Run a saved BigConv1D over X (CPU, batch). Returns softmax (N, C)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = BigConv1D(num_classes=n_classes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    out = []
    with torch.no_grad():
        # mini-batches of 256
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i+256]).float().to(device)
            logits = model(xb)
            out.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def predict_existing_conv1d(X: np.ndarray) -> np.ndarray:
    """Run the existing Keras-trained numpy Conv1D (artifacts/wlasl100/conv1d_small.npz)."""
    clf = WordClassifier(
        model_path=ROOT / "artifacts/wlasl100/conv1d_small.npz",
        labels_path=ROOT / "artifacts/wlasl100/labels.txt",
    )
    out = []
    for i in range(len(X)):
        logits = clf._forward(X[i])
        out.append(softmax_np(logits))
    return np.stack(out)


def predict_existing_gru(X: np.ndarray) -> np.ndarray:
    """Run the existing numpy GRU."""
    clf = GRUClassifier(
        model_path=ROOT / "artifacts/wlasl100/gru_small.npz",
        labels_path=ROOT / "artifacts/wlasl100/labels.txt",
    )
    out = []
    for i in range(len(X)):
        logits = clf.forward_logits(X[i])
        out.append(softmax_np(logits))
    return np.stack(out)


def main() -> int:
    d = np.load(ROOT / "data/wlasl100_kaggle/landmarks.npz", allow_pickle=True)
    labels = (ROOT / "data/wlasl100_kaggle/labels.txt").read_text(encoding="utf-8").splitlines()
    labels = [l.strip() for l in labels if l.strip()]
    n_classes = len(labels)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    sp = d["split"]
    test_mask = sp == 2
    Xt, yt = X[test_mask], y[test_mask]
    print(f"test set: {Xt.shape[0]} clips, {n_classes} classes")

    results = {}

    # individual A100 BigConv1D models
    a100_ckpts = [
        ("BigConv1D-s42", ROOT / "artifacts/wlasl100_a100_conv1d/ckpts/best.pt"),
        ("BigConv1D-s43", ROOT / "artifacts/wlasl100_a100_conv1d_s43/ckpts/best.pt"),
        ("BigConv1D-s1337", ROOT / "artifacts/wlasl100_a100_conv1d_s1337/ckpts/best.pt"),
    ]
    a100_probs = []
    for name, ckpt in a100_ckpts:
        p = predict_pytorch(ckpt, Xt, n_classes)
        a100_probs.append(p)
        results[name] = (topk_acc(p, yt, 1), topk_acc(p, yt, 5))

    # existing pipelines
    conv1d_p = predict_existing_conv1d(Xt)
    results["Conv1D (Keras, deployed)"] = (topk_acc(conv1d_p, yt, 1), topk_acc(conv1d_p, yt, 5))
    gru_p = predict_existing_gru(Xt)
    results["GRU (numpy, deployed)"] = (topk_acc(gru_p, yt, 1), topk_acc(gru_p, yt, 5))

    # existing 50/50 ensemble (deployed)
    p_existing_ens = 0.5 * conv1d_p + 0.5 * gru_p
    results["[Deployed] Conv1D+GRU (0.5/0.5)"] = (topk_acc(p_existing_ens, yt, 1), topk_acc(p_existing_ens, yt, 5))

    # A100 swarm ensemble
    p_a100_swarm = np.mean(a100_probs, axis=0)
    results["[A100] BigConv1D swarm (mean of 3)"] = (topk_acc(p_a100_swarm, yt, 1), topk_acc(p_a100_swarm, yt, 5))

    # A100 swarm + existing ensemble (5-way)
    p_combined = (p_existing_ens + p_a100_swarm) / 2.0
    results["[Combined] Existing-ens + A100-swarm (0.5/0.5)"] = (topk_acc(p_combined, yt, 1), topk_acc(p_combined, yt, 5))
    # weighted: trust A100 more
    p_combined_w = 0.3 * p_existing_ens + 0.7 * p_a100_swarm
    results["[Combined] 0.3 existing + 0.7 A100"] = (topk_acc(p_combined_w, yt, 1), topk_acc(p_combined_w, yt, 5))

    # 5-way mean (3 BigConv1D + Conv1D + GRU)
    p_5way = (sum(a100_probs) + conv1d_p + gru_p) / 5.0
    results["[5-way] 3xBigConv1D + Conv1D + GRU (mean)"] = (topk_acc(p_5way, yt, 1), topk_acc(p_5way, yt, 5))

    # nice table
    print()
    print(f"{'model':50s} {'top1':>8s} {'top5':>8s}")
    print("-" * 70)
    for name, (t1, t5) in results.items():
        print(f"{name:50s} {t1*100:>7.2f}% {t5*100:>7.2f}%")

    # save best-by-top1 ensemble's probs for downstream calibration
    best_name = max(results, key=lambda k: results[k][0])
    print()
    print(f"BEST: {best_name} top1={results[best_name][0]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
