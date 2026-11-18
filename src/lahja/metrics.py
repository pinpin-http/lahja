"""Metrics that scikit-learn does not provide: calibration error and temperature scaling."""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """ECE over equal-width confidence bins: sum over bins of |accuracy - confidence|, weighted
    by the share of rows in the bin."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> list[dict]:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            out.append(
                {
                    "lo": lo,
                    "hi": hi,
                    "n": int(mask.sum()),
                    "conf": float(conf[mask].mean()),
                    "acc": float(correct[mask].mean()),
                }
            )
    return out


def nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(logits / T)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One scalar T > 0 that minimises validation NLL (Guo et al., 2017). It rescales
    confidence without changing any prediction."""
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 20.0), method="bounded")
    return float(res.x)
