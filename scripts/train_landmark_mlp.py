"""Train the 63-d landmark MLP on extracted ASL29 landmarks.

Reads NPZ files produced by `scripts/extract_landmarks.py` and trains a
small MLP with sklearn. Exports weights to .npz for numpy inference on
the Pi (no TFLite runtime needed for this head).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load(npz_path: Path, drop_undetected: bool) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    X, y, det = data["X"], data["y"], data["detected"]
    if drop_undetected:
        mask = det.astype(bool)
        return X[mask], y[mask]
    return X, y


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _forward(X: np.ndarray, coefs: list, intercepts: list) -> np.ndarray:
    """ReLU hidden layers, softmax output."""
    h = X
    for W, b in zip(coefs[:-1], intercepts[:-1]):
        h = np.maximum(0.0, h @ W + b)
    logits = h @ coefs[-1] + intercepts[-1]
    return _softmax(logits)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks-dir", type=Path, default=Path("artifacts/asl29/landmarks"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/asl29/landmark_mlp"))
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelEncoder

    args.out_dir.mkdir(parents=True, exist_ok=True)

    Xt, yt = _load(args.landmarks_dir / "train.npz", drop_undetected=True)
    Xv, yv = _load(args.landmarks_dir / "val.npz", drop_undetected=True)
    Xs, ys = _load(args.landmarks_dir / "test.npz", drop_undetected=True)
    print(f"shapes: train {Xt.shape} val {Xv.shape} test {Xs.shape}")

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layers),
        activation="relu",
        solver="adam",
        learning_rate_init=args.lr,
        max_iter=args.max_iter,
        early_stopping=True,
        validation_fraction=0.0,
        n_iter_no_change=15,
        verbose=True,
        random_state=42,
    )
    # sklearn's early_stopping uses an internal val split; we monitor
    # manually instead using the pre-made val set.
    clf.early_stopping = False
    clf.max_iter = 1  # will loop manually
    clf._no_improvement_count = 0
    clf._best_val_acc = -1.0

    print("Training sklearn MLP …")
    best_val_acc = -1.0
    best_coefs = None
    best_intercepts = None
    no_improve = 0

    for epoch in range(args.max_iter):
        clf.partial_fit(Xt, yt, classes=np.arange(int(yt.max()) + 1))
        train_acc = clf.score(Xt, yt)
        val_acc = clf.score(Xv, yv)
        if epoch % 20 == 0 or epoch < 5:
            print(f"  epoch {epoch+1:3d}: train={train_acc:.4f} val={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_coefs = [c.copy() for c in clf.coefs_]
            best_intercepts = [b.copy() for b in clf.intercepts_]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 30:
                print(f"  early stop at epoch {epoch+1}, best_val={best_val_acc:.4f}")
                break

    # Restore best weights
    clf.coefs_ = best_coefs
    clf.intercepts_ = best_intercepts

    val_acc = clf.score(Xv, yv)
    test_acc = clf.score(Xs, ys)
    probs_test = _forward(Xs, clf.coefs_, clf.intercepts_)
    top3_acc = float((np.argsort(probs_test, axis=1)[:, -3:] == ys[:, None]).any(axis=1).mean())
    print(f"\nVal accuracy (detected only): {val_acc:.4f}")
    print(f"Test accuracy (detected only): {test_acc:.4f}")
    print(f"Test top-3 accuracy: {top3_acc:.4f}")

    # Save as .npz — LandmarkClassifier supports this format
    out_path = args.out_dir / "landmark_mlp.npz"
    np.savez_compressed(
        out_path,
        **{f"W{i}": c for i, c in enumerate(clf.coefs_)},
        **{f"b{i}": b for i, b in enumerate(clf.intercepts_)},
        n_layers=np.array(len(clf.coefs_)),
    )
    print(f"Saved weights to {out_path}")

    metrics = {
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy_on_detected": float(test_acc),
        "test_top3_accuracy": float(top3_acc),
        "n_train": int(len(yt)),
        "n_val": int(len(yv)),
        "n_test": int(len(ys)),
        "hidden_layers": list(args.hidden_layers),
        "npz_path": str(out_path),
        "npz_size_bytes": int(out_path.stat().st_size),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
