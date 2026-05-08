"""Signer-conditioned fine-tune on top of the BigConv1D backbone.

Consumes the output of `scripts/record_demo_vocab.py`. Loads the
pretrained backbone, swaps the classifier head for a small custom
vocabulary, freezes the early conv blocks, and fine-tunes the deeper
block, attention pool, and head with mixup and label smoothing. Take-
based 3/1/1 split per class avoids clip leakage. Exports an npz so the
Pi runtime can swap it in without PyTorch.

Usage:
    python scripts/finetune_demo_words.py \
        --data data/fewshot/landmarks.npz \
        --backbone artifacts/wlasl100_a100_conv1d/ckpts/best.pt \
        --out-dir artifacts/wlasl5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_conv1d_a100 import BigConv1D  # type: ignore  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))


def topk(probs: torch.Tensor, y: torch.Tensor, k: int) -> float:
    return float((probs.topk(min(k, probs.size(1)), dim=1).indices == y[:, None]).any(dim=1).float().mean().item())


def evaluate(model, X: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)
    return {
        "top1": topk(probs, y, 1),
        "top3": topk(probs, y, 3),
        "loss": float(F.cross_entropy(logits, y).item()),
        "n": int(y.numel()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/fewshot/landmarks.npz"))
    ap.add_argument("--backbone", type=Path, default=Path("artifacts/wlasl100_a100_conv1d/ckpts/best.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/wlasl5"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--mixup", type=float, default=0.2)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.data.exists():
        print(f"[finetune] dataset missing: {args.data}", file=sys.stderr)
        print("  → run `scripts/record_demo_vocab.py` first.", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[finetune] device={device}")

    d = np.load(args.data, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    take_ids = d["take_ids"].astype(np.int64)
    class_names = list(d["class_names"])
    n_classes = len(class_names)
    print(f"[finetune] data={X.shape} classes={n_classes} ({class_names})")

    # take-based split
    train_mask = take_ids <= 2
    val_mask = take_ids == 3
    test_mask = take_ids == 4
    print(f"[finetune] train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")

    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    Xtr = Xt[train_mask]; ytr = yt[train_mask]
    Xv = Xt[val_mask]; yv = yt[val_mask]
    Xte = Xt[test_mask]; yte = yt[test_mask]

    # build model with backbone num_classes=100, then swap head to n_classes
    model = BigConv1D(num_classes=100).to(device)
    if args.backbone.exists():
        ckpt = torch.load(args.backbone, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[finetune] loaded backbone {args.backbone}")
    else:
        print(f"[finetune] backbone {args.backbone} missing — training from scratch")

    # Replace classifier head: model.fc is Sequential(Dropout, Linear(192,128), ReLU, Dropout, Linear(128, 100))
    # we reset the last Linear to (128, n_classes)
    model.fc[4] = nn.Linear(128, n_classes).to(device)

    # Freeze the early/temporal feature blocks; train only the deep conv b3
    # + attention pool + fc head. This anchors the pretraining and just
    # adapts the higher-level features to the signer.
    for name, p in model.named_parameters():
        if name.startswith("b1.") or name.startswith("b2."):
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[finetune] trainable params: {sum(p.numel() for p in trainable)/1e3:.1f}K")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    best_val_top1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(Xtr.size(0), device=device)
        Xtr_s = Xtr[perm]
        ytr_s = ytr[perm]
        running_loss = 0.0
        nb = 0
        for i in range(0, Xtr_s.size(0), args.batch_size):
            xb = Xtr_s[i:i+args.batch_size]
            yb = ytr_s[i:i+args.batch_size]
            # mixup
            if args.mixup > 0 and xb.size(0) > 1:
                lam = np.random.beta(args.mixup, args.mixup)
                lam = max(lam, 1.0 - lam)
                idx = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1.0 - lam) * xb[idx]
                yb_oh = F.one_hot(yb, n_classes).float()
                yb_oh = lam * yb_oh + (1.0 - lam) * yb_oh[idx]
                target = yb_oh
            else:
                target = F.one_hot(yb, n_classes).float()

            # light additive noise on landmark space
            xb = xb + 0.003 * torch.randn_like(xb)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            log_p = F.log_softmax(logits, dim=1)
            ls = args.label_smoothing
            target = target * (1 - ls) + ls / n_classes
            loss = -(target * log_p).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
            nb += xb.size(0)
        scheduler.step()
        train_loss = running_loss / max(1, nb)

        if Xv.size(0) > 0:
            v = evaluate(model, Xv, yv)
            if v["top1"] > best_val_top1:
                best_val_top1 = v["top1"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch % 20 == 0 or epoch == args.epochs or epoch == 1:
                print(f"epoch {epoch:03d} loss={train_loss:.4f} val_top1={v['top1']:.3f}/{v['top3']:.3f}")
            log.append({"epoch": epoch, "train_loss": train_loss, "val": v})
        else:
            if epoch % 20 == 0 or epoch == args.epochs:
                print(f"epoch {epoch:03d} loss={train_loss:.4f}")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval
    final = {}
    final["train"] = evaluate(model, Xtr, ytr)
    if Xv.size(0) > 0:
        final["val"] = evaluate(model, Xv, yv)
    if Xte.size(0) > 0:
        final["test"] = evaluate(model, Xte, yte)
    print("[finetune] final:", json.dumps(final, indent=2))

    # save PyTorch ckpt + npz export (numpy-loadable on Pi)
    pt_path = args.out_dir / "conv1d_demo.pt"
    torch.save({"model": model.state_dict(), "class_names": class_names}, pt_path)

    npz_path = args.out_dir / "conv1d_demo.npz"
    weights = {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    weights["__arch__"] = np.array(["bigconv1d"], dtype="<U32")
    weights["__num_classes__"] = np.array([n_classes], dtype=np.int32)
    weights["__input_shape__"] = np.array([X.shape[1], X.shape[2]], dtype=np.int32)
    np.savez_compressed(npz_path, **weights)
    print(f"[finetune] saved {pt_path} and {npz_path}")

    labels_out = args.out_dir / "labels.txt"
    labels_out.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(f"[finetune] saved labels: {labels_out}")

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "class_names": class_names,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "best_val_top1": best_val_top1,
        "final": final,
        "args": {k: str(v) for k, v in vars(args).items()},
    }, indent=2), encoding="utf-8")
    print(f"[finetune] saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
