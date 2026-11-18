"""Reference model: character n-grams (2 to 5) + TF-IDF + logistic regression.

Fast on CPU, and its weights say which letter sequences mark each dialect."""

from __future__ import annotations

import json
import logging
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from .config import CLASSES, SEED, corpus_dirs
from .data import load_split
from .predictions import save_predictions

log = logging.getLogger(__name__)
C_GRID = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def make_pipeline(C: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, dtype=np.float32
                ),
            ),
            (
                "lr",
                LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED),
            ),
        ]
    )


def train(corpus: str) -> dict:
    d = corpus_dirs(corpus)
    df = load_split(corpus)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    y_tr = tr.label.map(CLASSES.index).values
    y_va = va.label.map(CLASSES.index).values

    scores = {}
    for C in C_GRID:
        pipe = make_pipeline(C).fit(tr.text_norm, y_tr)
        scores[C] = f1_score(y_va, pipe.predict(va.text_norm), average="macro")
        log.info("C=%s  val macro-F1 %.4f", C, scores[C])
    best_C = max(scores, key=scores.get)

    t0 = time.perf_counter()
    pipe = make_pipeline(best_C).fit(tr.text_norm, y_tr)
    fit_s = time.perf_counter() - t0

    out = d["models"] / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "model.pkl").open("wb") as f:
        pickle.dump(pipe, f)

    rows = df[df.split != "train"]
    probs = pipe.predict_proba(rows.text_norm)
    save_predictions(corpus, "baseline", rows, probs)

    info = {
        "model": "baseline",
        "C_grid": {str(k): round(v, 4) for k, v in scores.items()},
        "C": best_C,
        "fit_seconds": round(fit_s, 1),
        "n_features": len(pipe.named_steps["tfidf"].vocabulary_),
        "top_ngrams": top_ngrams(pipe),
    }
    d["results"].mkdir(parents=True, exist_ok=True)
    (d["results"] / "train_baseline.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return info


def top_ngrams(pipe: Pipeline, k: int = 15) -> dict[str, list[str]]:
    vocab = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    coef = pipe.named_steps["lr"].coef_
    out = {}
    for i, c in enumerate(CLASSES):
        order = np.argsort(-coef[i])
        # char_wb pads words with spaces; keep n-grams with at least two letters
        grams = [g for g in vocab[order] if len(g.strip()) >= 2]
        out[c] = grams[:k]
    return out


def load(corpus: str) -> Pipeline:
    with (corpus_dirs(corpus)["models"] / "baseline" / "model.pkl").open("rb") as f:
        return pickle.load(f)


def predict_frame(pipe: Pipeline, texts: pd.Series) -> np.ndarray:
    return pipe.predict_proba(texts)
