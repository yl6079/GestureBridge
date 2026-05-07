"""ST-GCN training on WLASL-100 hand-landmark sequences (A100, PyTorch).

Inputs (already on disk on the A100 after `convert_kaggle_wlasl100_landmarks.py`):
    data/wlasl100_kaggle/landmarks.npz    # X: (N, 30, 63), y: (N,), split: (N,)
    data/wlasl100_kaggle/labels.txt       # 100 gloss names

Outputs (per-epoch + best, written to `--out-dir`):
    artifacts/wlasl100_a100/ckpts/epoch_NNN.pt   # rolling, last 3 kept
    artifacts/wlasl100_a100/ckpts/best.pt        # by val top-1
    artifacts/wlasl100_a100/eval.json            # rolling metrics
    artifacts/wlasl100_a100/train.log            # training log (also tee'd to stdout)

Architecture: minimal ST-GCN — 3 spatial-temporal graph conv blocks
(64 -> 128 -> 256 channels) over the 21-node hand skeleton, followed
by global average pool + linear classifier. Pure PyTorch, no extra
graph-conv libraries (the spatial conv is a single matmul against a
precomputed adjacency matrix).

Usage on A100:
    cd ~/shufeng/elen6908/GestureBridge
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate dd6908
    # smoke test (1 epoch on 200 clips)
    python scripts/train_stgcn_a100.py --smoke --epochs 1
    # full run
    python scripts/train_stgcn_a100.py --epochs 80
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Hand graph (21 MediaPipe Hands keypoints)
# -----------------------------------------------------------------------------
# Node order matches MediaPipe Hands convention:
#   0=WRIST, 1-4=THUMB, 5-8=INDEX, 9-12=MIDDLE, 13-16=RING, 17-20=PINKY
HAND_EDGES = [
    # thumb chain
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index chain
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle chain
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring chain
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky chain
    (0, 17), (17, 18), (18, 19), (19, 20),
    # palm rim — connect MCP joints so the hand isn't a star around the wrist
    (5, 9), (9, 13), (13, 17), (1, 5),
]
NUM_NODES = 21


def build_adjacency(num_nodes: int = NUM_NODES, edges=HAND_EDGES) -> np.ndarray:
    """Symmetrically-normalized adjacency D^{-1/2} (A + I) D^{-1/2}."""
    a = np.eye(num_nodes, dtype=np.float32)
    for (i, j) in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    deg = a.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-6))
    return (a * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]).astype(np.float32)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class WLASLPoseDataset(Dataset):
    """Wraps the (N, 30, 63) float32 landmarks tensor.

    Returns per-clip tensor of shape (C=3, T=30, V=21) and an int label.
    Augmentations done on numpy on the CPU, before tensor conversion.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        # (N, 30, 63) -> (N, 30, 21, 3)
        self.X = X.reshape(X.shape[0], X.shape[1], NUM_NODES, 3).astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return self.X.shape[0]

    def _augment_clip(self, x: np.ndarray) -> np.ndarray:
        # Conservative aug — matched to what worked for the existing
        # Conv1D-Small (temporal jitter + small spatial scale). The
        # earlier aggressive variant (rotation + noise + temporal warp +
        # frame dropout) caused under-fitting on a 11K-clip dataset.
        T, V, C = x.shape
        # 1. spatial scale 0.9 - 1.1 (same as Conv1D recipe)
        s = np.random.uniform(0.9, 1.1)
        x = x * s
        # 2. temporal jitter: shift ±2 frames (cheap and helps timing)
        shift = np.random.randint(-2, 3)
        if shift != 0:
            x = np.roll(x, shift=shift, axis=0)
        return x

    def __getitem__(self, i: int):
        x = self.X[i]  # (T, V, C)
        if self.augment:
            x = self._augment_clip(x)
        # torch wants (C, T, V)
        x = np.transpose(x, (2, 0, 1)).copy()
        return torch.from_numpy(x), int(self.y[i])


# -----------------------------------------------------------------------------
# ST-GCN model
# -----------------------------------------------------------------------------
class STGCNBlock(nn.Module):
    """One spatial graph-conv + one temporal 1D-conv block.

    spatial: y[c'] = sum_v A[v,v'] * x[c,v,t] then 1x1 conv over channels
    temporal: depthwise-style 1D conv along time axis (per-node).
    """
    def __init__(self, in_ch: int, out_ch: int, t_kernel: int = 9, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        # spatial: 1x1 conv after the adjacency matmul
        self.spatial_conv = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1))
        self.spatial_bn = nn.BatchNorm2d(out_ch)
        # temporal: conv over T axis, kernel size t_kernel, padding to keep T
        pad_t = (t_kernel - 1) // 2
        self.temporal_conv = nn.Conv2d(
            out_ch, out_ch, kernel_size=(t_kernel, 1), stride=(stride, 1), padding=(pad_t, 0)
        )
        self.temporal_bn = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout)
        # residual
        if in_ch != out_ch or stride != 1:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1))
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        # spatial: (B, C, T, V) @ (V, V) along V -> einsum
        residual = self.residual(x)
        y = torch.einsum("bctv,vw->bctw", x, A)
        y = self.spatial_conv(y)
        y = self.spatial_bn(y)
        y = F.relu(y, inplace=True)
        y = self.temporal_conv(y)
        y = self.temporal_bn(y)
        y = F.relu(y + residual, inplace=True)
        y = self.dropout(y)
        return y


class STGCN(nn.Module):
    """Small ST-GCN sized to match the Conv1D-Small (~50K params) so it
    doesn't overfit our 11K-clip dataset. Bigger initial config (64-128-256
    channels, kernel 9) gave train_top1=0.99 / test_top1=0.25 — classic
    over-parameterization regression. This variant: 2 blocks, kernel 5,
    32→64 channels, ~80K params total."""

    def __init__(self, num_classes: int, in_channels: int = 3, num_nodes: int = NUM_NODES,
                 width: int = 32, t_kernel: int = 5, dropout: float = 0.4):
        super().__init__()
        # data BatchNorm so absolute coord stats are stabilized
        self.input_bn = nn.BatchNorm1d(in_channels * num_nodes)
        self.block1 = STGCNBlock(in_channels, width, t_kernel=t_kernel, dropout=dropout)
        self.block2 = STGCNBlock(width, width * 2, t_kernel=t_kernel, stride=2, dropout=dropout)
        self.fc = nn.Linear(width * 2, num_classes)
        # adjacency stored as buffer so it moves to GPU with the model
        A = build_adjacency()
        self.register_buffer("A", torch.from_numpy(A))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        B, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.input_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()  # back to (B, C, T, V)
        x = self.block1(x, self.A)
        x = self.block2(x, self.A)
        # global avg pool over (T, V)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    data: Path = Path("data/wlasl100_kaggle/landmarks.npz")
    labels: Path = Path("data/wlasl100_kaggle/labels.txt")
    out_dir: Path = Path("artifacts/wlasl100_a100")
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    smoke: bool = False
    seed: int = 42
    workers: int = 4
    eval_every: int = 1
    keep_last_k: int = 3


def topk(probs: torch.Tensor, y: torch.Tensor, k: int) -> float:
    pred = probs.topk(k, dim=1).indices  # (B, k)
    correct = (pred == y[:, None]).any(dim=1).float().mean().item()
    return correct


def evaluate(model, loader, device) -> dict:
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            all_logits.append(logits)
            all_y.append(y)
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    probs = F.softmax(logits, dim=1)
    return {
        "n": int(y.numel()),
        "top1": topk(probs, y, 1),
        "top5": topk(probs, y, 5),
        "loss": float(F.cross_entropy(logits, y).item()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/wlasl100_kaggle/landmarks.npz"))
    p.add_argument("--labels", type=Path, default=Path("data/wlasl100_kaggle/labels.txt"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/wlasl100_a100"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--smoke", action="store_true", help="train on 200 clips for 1 epoch — pipeline sanity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu-mem-fraction", type=float, default=0.5,
                   help="cap CUDA memory share so we don't crowd the 6699 sweep")
    p.add_argument("--resume", type=Path, default=None, help="resume from a .pt checkpoint")
    p.add_argument("--keep-last-k", type=int, default=3,
                   help="how many recent epoch checkpoints to retain (best.pt is always kept)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stgcn] device={device}")
    if device.type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, device=0)
            print(f"[stgcn] capped CUDA memory fraction at {args.gpu_mem_fraction}")
        except Exception as exc:
            print(f"[stgcn] memory fraction cap not applied: {exc}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    print(f"[stgcn] loading {args.data}")
    d = np.load(args.data, allow_pickle=True)
    X, y, sp = d["X"].astype(np.float32), d["y"].astype(np.int64), d["split"]
    labels = [g.strip() for g in args.labels.read_text(encoding="utf-8").splitlines() if g.strip()]
    n_classes = len(labels)
    print(f"[stgcn] X={X.shape} classes={n_classes} train/val/test={(sp==0).sum()}/{(sp==1).sum()}/{(sp==2).sum()}")

    train_idx = np.where(sp == 0)[0]
    val_idx = np.where(sp == 1)[0]
    test_idx = np.where(sp == 2)[0]

    if args.smoke:
        # tiny subset to validate pipeline before the long run
        rng = np.random.default_rng(0)
        train_idx = rng.choice(train_idx, size=min(200, len(train_idx)), replace=False)
        val_idx = rng.choice(val_idx, size=min(50, len(val_idx)), replace=False)
        test_idx = rng.choice(test_idx, size=min(50, len(test_idx)), replace=False)
        args.epochs = 1
        print(f"[stgcn] SMOKE TEST — train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} epochs=1")

    train_ds = WLASLPoseDataset(X[train_idx], y[train_idx], augment=True)
    val_ds = WLASLPoseDataset(X[val_idx], y[val_idx], augment=False)
    test_ds = WLASLPoseDataset(X[test_idx], y[test_idx], augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = STGCN(num_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[stgcn] params={n_params/1e6:.2f}M")

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"[stgcn] resumed from {args.resume} at epoch {start_epoch}")
    else:
        start_epoch = 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_val_top1 = -1.0
    metrics_log: list[dict] = []
    log_path = args.out_dir / "train.log"
    eval_path = args.out_dir / "eval.json"

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"start training: epochs={args.epochs} batch={args.batch_size} lr={args.lr}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.monotonic()
        running_loss = 0.0
        running_top1 = 0.0
        n_seen = 0
        for x, y_b in train_loader:
            x = x.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * y_b.size(0)
            running_top1 += float((logits.argmax(1) == y_b).float().sum().item())
            n_seen += y_b.size(0)
        scheduler.step()
        train_loss = running_loss / max(1, n_seen)
        train_top1 = running_top1 / max(1, n_seen)

        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device) if (epoch % 5 == 0 or epoch == args.epochs) else None

        epoch_secs = time.monotonic() - t0
        msg = (
            f"epoch {epoch:03d}/{args.epochs}  "
            f"loss={train_loss:.4f} train_top1={train_top1:.3f}  "
            f"val_top1={val_metrics['top1']:.3f} val_top5={val_metrics['top5']:.3f}  "
            f"lr={scheduler.get_last_lr()[0]:.5f}  {epoch_secs:.1f}s"
        )
        if test_metrics is not None:
            msg += f"  TEST top1={test_metrics['top1']:.3f} top5={test_metrics['top5']:.3f}"
        log(msg)

        ckpt_payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_top1": val_metrics["top1"],
        }

        ckpt_path = args.out_dir / "ckpts" / f"epoch_{epoch:03d}.pt"
        torch.save(ckpt_payload, ckpt_path)

        # rotate: keep last K + best
        all_ckpts = sorted((args.out_dir / "ckpts").glob("epoch_*.pt"))
        if len(all_ckpts) > args.keep_last_k:
            for stale in all_ckpts[:-args.keep_last_k]:
                try:
                    stale.unlink()
                except Exception:
                    pass

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            torch.save(ckpt_payload, args.out_dir / "ckpts" / "best.pt")
            log(f"  -> best val_top1={best_val_top1:.3f}, saved best.pt")

        metrics_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "test": test_metrics,
            "lr": scheduler.get_last_lr()[0],
            "epoch_secs": epoch_secs,
        })
        eval_path.write_text(json.dumps(metrics_log, indent=2), encoding="utf-8")

    # final test
    final_test = evaluate(model, test_loader, device)
    log(f"FINAL TEST top1={final_test['top1']:.4f} top5={final_test['top5']:.4f} n={final_test['n']}")
    summary = {
        "best_val_top1": best_val_top1,
        "final_test_top1": final_test["top1"],
        "final_test_top5": final_test["top5"],
        "n_test": final_test["n"],
        "args": vars(args),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log(f"summary written to {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
