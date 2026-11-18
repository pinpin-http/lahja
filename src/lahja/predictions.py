"""Predictions on validation, test and out-of-domain rows, one Parquet file per model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASSES, corpus_dirs


def save_predictions(
    corpus: str, model: str, rows: pd.DataFrame, probs: np.ndarray, logits: np.ndarray | None = None
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "row": rows.index.values,
            "split": rows.split.values,
            "set": rows.set.values,
            "label": rows.label.values,
        }
    )
    out["pred"] = [CLASSES[i] for i in probs.argmax(1)]
    for i, c in enumerate(CLASSES):
        out[f"p_{c}"] = probs[:, i].astype(np.float32)
    if logits is not None:
        for i, c in enumerate(CLASSES):
            out[f"logit_{c}"] = logits[:, i].astype(np.float32)
    path = corpus_dirs(corpus)["predictions"]
    path.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path / f"{model}.parquet", index=False)
    return out


def load_predictions(corpus: str, model: str) -> pd.DataFrame:
    return pd.read_parquet(corpus_dirs(corpus)["predictions"] / f"{model}.parquet")


def probs_of(df: pd.DataFrame, prefix: str = "p_") -> np.ndarray:
    return df[[f"{prefix}{c}" for c in CLASSES]].to_numpy()
