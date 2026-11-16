"""Mechanistic core: refusal direction extraction, layer localisation, probe transfer.

Nothing here is simulated. Directions are estimated from real activations of the
model under test; the probe is a plain logistic regression solved in numpy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def mean_difference_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Classic refusal direction: mean(refusals) - mean(compliances), L2-normalised."""
    if pos.size == 0 or neg.size == 0:
        raise ValueError("need both classes to estimate a direction")
    d = pos.mean(axis=0) - neg.mean(axis=0)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        return d.astype(np.float32)
    return (d / norm).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def layer_sweep(features_by_layer: Dict[int, np.ndarray], labels: np.ndarray) -> List[Dict[str, float]]:
    """For each layer, how separable are refusals from compliances?

    `features_by_layer[layer]` is (n_samples, d); `labels` is 1 for refusal.
    We use the mean-difference direction + its own projection as the score
    (a rank-1 linear probe), which needs no iterative optimisation.
    """
    out: List[Dict[str, float]] = []
    for layer in sorted(features_by_layer):
        X = features_by_layer[layer]
        pos, neg = X[labels == 1], X[labels == 0]
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            continue
        v = mean_difference_direction(pos, neg)
        proj = X @ v
        thr = (proj[labels == 1].mean() + proj[labels == 0].mean()) / 2.0
        pred = (proj > thr).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out.append({"layer": int(layer), "f1": round(f1, 4),
                    "precision": round(precision, 4), "recall": round(recall, 4),
                    "separation": round(abs(float(proj[labels == 1].mean() - proj[labels == 0].mean())), 4)})
    return out


def logistic_probe(X: np.ndarray, y: np.ndarray, epochs: int = 400, lr: float = 0.5,
                   l2: float = 1e-3) -> np.ndarray:
    """Plain L2-regularised logistic regression (numpy, full-batch gradient descent)."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1), dtype=np.float32), X.astype(np.float32)])
    w = np.zeros(d + 1, dtype=np.float64)
    y = y.astype(np.float64)
    for _ in range(epochs):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / n
        grad[1:] += l2 * w[1:]
        w -= lr * grad
    return w.astype(np.float32)


def probe_accuracy(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    Xb = np.hstack([np.ones((X.shape[0], 1), dtype=np.float32), X.astype(np.float32)])
    z = Xb @ w
    pred = (z > 0).astype(int)
    return float((pred == y.astype(int)).mean())


def direction_alignment(dirs: Dict[str, np.ndarray], reference: str = "en") -> Dict[str, float]:
    """Cosine of each language's direction against the reference-language direction."""
    if reference not in dirs:
        reference = sorted(dirs)[0]
    ref = dirs[reference]
    return {lang: round(cosine(v, ref), 4) for lang, v in sorted(dirs.items())}
