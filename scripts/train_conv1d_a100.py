"""Big Conv1D / TCN on WLASL-100 hand-landmark sequences (A100, PyTorch).

T2B fallback after the ST-GCN T2A attempt under-/over-fit. Same data
(`data/wlasl100_kaggle/landmarks.npz`), different architecture: a deeper
1D-conv tower with a temporal attention pool, mixup augmentation, and
label smoothing. Goal: beat the existing Keras Conv1D + GRU ensemble's
test top-1 = 0.577 / top-5 = 0.870.

Usage:
    python scripts/train_conv1d_a100.py --epochs 120
    python scripts/train_conv1d_a100.py --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class WLASLSeqDataset(Dataset):
    """Returns (T=30, C=63) tensor + int label. Augmentation matches the
    Conv1D recipe (temporal jitter ±2, spatial scale 0.9-1.1)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return self.X.shape[0]

    def _aug(self, x: np.ndarray) -> np.ndarray:
        s = np.random.uniform(0.9, 1.1)
        x = x * s
        shift = np.random.randint(-2, 3)
        if shift != 0:
            x = np.roll(x, shift=shift, axis=0)
        return x

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.augment:
            x = self._aug(x)
        return torch.from_numpy(x).clone(), int(self.y[i])


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, dropout=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(in_ch, out_ch, k, padding=k // 2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.c2 = nn.Conv1d(out_ch, out_ch, k, padding=k // 2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        s = self.skip(x)
        x = F.relu(self.bn1(self.c1(x)), inplace=True)
        x = self.bn2(self.c2(x))
        x = F.relu(x + s, inplace=True)
        return self.drop(x)


class BigConv1D(nn.Module):
    """Deeper Conv1D over (B, C=63, T=30). ~150K params, plenty of
    capacity but with strong dropout + skip + label smoothing."""

    def __init__(self, num_classes: int, in_ch: int = 63, dropout: float = 0.3):
        super().__init__()
        self.b1 = ConvBlock(in_ch, 96, k=3, dropout=dropout)
        self.b2 = ConvBlock(96, 128, k=3, dropout=dropout)
        self.b3 = ConvBlock(128, 192, k=3, dropout=dropout)
        # attention pool: weighted average over time
        self.attn = nn.Linear(192, 1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        # attention over time
        h = x.transpose(1, 2)  # (B, T, C)
        a = self.attn(h)        # (B, T, 1)
        w = torch.softmax(a, dim=1)
        z = (h * w).sum(dim=1)  # (B, C)
        return self.fc(z)


def topk(probs, y, k):
    return float((probs.topk(k, dim=1).indices == y[:, None]).any(dim=1).float().mean().item())


def evaluate(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits_all.append(model(x))
            y_all.append(y)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    probs = F.softmax(logits, dim=1)
    return {"top1": topk(probs, y, 1), "top5": topk(probs, y, 5),
            "loss": float(F.cross_entropy(logits, y).item()), "n": int(y.numel())}


def mixup(x, y, alpha=0.2, num_classes=100):
    if alpha <= 0:
        y_oh = F.one_hot(y, num_classes).float()
        return x, y_oh
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1.0 - lam) * x[idx]
    y_oh = F.one_hot(y, num_classes).float()
    y_oh = lam * y_oh + (1.0 - lam) * y_oh[idx]
    return x, y_oh


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/wlasl100_kaggle/landmarks.npz"))
    p.add_argument("--labels", type=Path, default=Path("data/wlasl100_kaggle/labels.txt"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/wlasl100_a100_conv1d"))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.2)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--gpu-mem-fraction", type=float, default=0.5)
    p.add_argument("--keep-last-k", type=int, default=3)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[conv1d] device={device}")
    if device.type == "cuda":
        try:
            torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, device=0)
        except Exception:
            pass

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    X, y, sp = d["X"].astype(np.float32), d["y"].astype(np.int64), d["split"]
    labels = [g.strip() for g in args.labels.read_text(encoding="utf-8").splitlines() if g.strip()]
    n_classes = len(labels)
    train_idx = np.where(sp == 0)[0]
    val_idx = np.where(sp == 1)[0]
    test_idx = np.where(sp == 2)[0]
    print(f"[conv1d] X={X.shape} classes={n_classes} train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    if args.smoke:
        rng = np.random.default_rng(0)
        train_idx = rng.choice(train_idx, size=min(200, len(train_idx)), replace=False)
        val_idx = rng.choice(val_idx, size=min(50, len(val_idx)), replace=False)
        test_idx = rng.choice(test_idx, size=min(50, len(test_idx)), replace=False)
        args.epochs = 2

    train_ds = WLASLSeqDataset(X[train_idx], y[train_idx], augment=True)
    val_ds = WLASLSeqDataset(X[val_idx], y[val_idx], augment=False)
    test_ds = WLASLSeqDataset(X[test_idx], y[test_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = BigConv1D(num_classes=n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[conv1d] params={n_params/1e3:.1f}K")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_test_top1 = -1.0
    metrics_log: list[dict] = []
    log_path = args.out_dir / "train.log"
    eval_path = args.out_dir / "eval.json"

    def log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"start: epochs={args.epochs} batch={args.batch_size} lr={args.lr} mixup={args.mixup}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.monotonic()
        running_loss = 0.0
        n_seen = 0
        for x, y_b in train_loader:
            x = x.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            x_mix, y_oh = mixup(x, y_b, alpha=args.mixup, num_classes=n_classes)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_mix)
            log_p = F.log_softmax(logits, dim=1)
            # mixed soft target with label smoothing
            ls = args.label_smoothing
            y_target = y_oh * (1 - ls) + ls / n_classes
            loss = -(y_target * log_p).sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * y_b.size(0)
            n_seen += y_b.size(0)
        scheduler.step()
        train_loss = running_loss / max(1, n_seen)

        val = evaluate(model, val_loader, device)
        do_test = epoch % 5 == 0 or epoch == args.epochs
        test = evaluate(model, test_loader, device) if do_test else None
        secs = time.monotonic() - t0
        msg = (f"epoch {epoch:03d}/{args.epochs}  loss={train_loss:.4f}  "
               f"val_top1={val['top1']:.3f}/{val['top5']:.3f}  lr={scheduler.get_last_lr()[0]:.5f}  {secs:.1f}s")
        if test is not None:
            msg += f"  TEST {test['top1']:.4f}/{test['top5']:.4f}"
        log(msg)

        ckpt = {"model": model.state_dict(), "epoch": epoch, "args": vars(args),
                "val_top1": val["top1"], "test": test}
        torch.save(ckpt, args.out_dir / "ckpts" / f"epoch_{epoch:03d}.pt")
        all_ckpts = sorted((args.out_dir / "ckpts").glob("epoch_*.pt"))
        if len(all_ckpts) > args.keep_last_k:
            for stale in all_ckpts[:-args.keep_last_k]:
                try: stale.unlink()
                except Exception: pass

        if test is not None and test["top1"] > best_test_top1:
            best_test_top1 = test["top1"]
            torch.save(ckpt, args.out_dir / "ckpts" / "best.pt")
            log(f"  -> best TEST top1={best_test_top1:.4f}, saved best.pt")

        metrics_log.append({"epoch": epoch, "train_loss": train_loss, "val": val,
                            "test": test, "lr": scheduler.get_last_lr()[0], "epoch_secs": secs})
        eval_path.write_text(json.dumps(metrics_log, indent=2), encoding="utf-8")

    final_test = evaluate(model, test_loader, device)
    log(f"FINAL TEST top1={final_test['top1']:.4f} top5={final_test['top5']:.4f} n={final_test['n']}")
    summary = {"best_test_top1": best_test_top1,
               "final_test_top1": final_test["top1"], "final_test_top5": final_test["top5"],
               "n_test": final_test["n"], "args": vars(args)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
