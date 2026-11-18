"""Evaluation: the three models on the test set and the out-of-domain sets, the cost of
quantisation, calibration, CPU latency, and the acceptance thresholds."""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .config import CLASSES, ENCODERS, SEED, corpus_dirs
from .data import load_split
from .metrics import expected_calibration_error, fit_temperature, reliability, softmax
from .predictions import load_predictions, probs_of, save_predictions

log = logging.getLogger(__name__)

THRESHOLDS = {"min_gain_points": 5.0, "max_quantization_loss_points": 1.0, "max_p95_ms": 300.0}
LATENCY_SAMPLE = 300
LATENCY_THREADS = 2  # a free Hugging Face CPU Space has 2 vCPUs


def score(df: pd.DataFrame, probs: np.ndarray, raw_probs: np.ndarray | None = None) -> dict:
    """Metrics over the classes present in the gold labels. An out-of-domain set without MSA
    is scored on its four dialects; predicting MSA there still counts as an error."""
    y = df.label.map(CLASSES.index).to_numpy()
    pred = probs.argmax(1)
    present = [i for i in range(len(CLASSES)) if (y == i).any()]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=present, zero_division=0)
    out = {
        "n": int(len(y)),
        "macro_f1": round(float(f1_score(y, pred, labels=present, average="macro", zero_division=0)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "per_class": {
            CLASSES[i]: {
                "precision": round(float(a), 4),
                "recall": round(float(b), 4),
                "f1": round(float(c), 4),
                "n": int(d),
            }
            for i, a, b, c, d in zip(present, p, r, f, s, strict=True)
        },
        "confusion": confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist(),
        "ece": round(expected_calibration_error(probs, y), 4),
    }
    if raw_probs is not None:
        out["ece_uncalibrated"] = round(expected_calibration_error(raw_probs, y), 4)
    return out


def eval_sets(df: pd.DataFrame) -> dict[str, pd.Series]:
    sets = {"test": (df.split == "test")}
    for name in sorted(df.loc[df.split == "ood", "set"].unique()):
        sets[name] = (df.split == "ood") & (df.set == name)
    return sets


def onnx_predictions(corpus: str, df: pd.DataFrame, onnx_file: str, name: str) -> pd.DataFrame:
    from .inference import DialectModel

    d = corpus_dirs(corpus)
    m = DialectModel(d["models"] / "onnx", onnx_file=onnx_file, threads=4)
    rows = df[df.split != "train"]
    texts = rows.text_norm.tolist()
    logits = np.concatenate([m.logits(texts[i : i + 64]) for i in range(0, len(texts), 64)])
    val = (rows.split == "val").to_numpy()
    T = fit_temperature(logits[val], rows.label[val].map(CLASSES.index).to_numpy())
    preds = save_predictions(corpus, name, rows, softmax(logits / T), logits)
    preds.attrs["temperature"] = T
    return preds


def latency(corpus: str, df: pd.DataFrame) -> dict:
    """Wall time per sentence on CPU, one request at a time, measured through the FastAPI app
    (validation, normalisation, tokenisation, ONNX, JSON), plus the model call alone."""
    from fastapi.testclient import TestClient

    from . import api
    from .inference import DialectModel

    d = corpus_dirs(corpus)
    model = DialectModel(d["models"] / "onnx", threads=LATENCY_THREADS)
    texts = df[df.split == "test"].text.sample(LATENCY_SAMPLE, random_state=SEED).tolist()
    api.app.dependency_overrides[api.get_model] = lambda: model
    client = TestClient(api.app)
    for t in texts[:20]:  # warm-up
        client.post("/predict", json={"text": t})

    def timed(fn):
        out = []
        for t in texts:
            t0 = time.perf_counter()
            fn(t)
            out.append((time.perf_counter() - t0) * 1000)
        a = np.array(out)
        return {
            k: round(float(v), 1)
            for k, v in {
                "p50": np.percentile(a, 50),
                "p95": np.percentile(a, 95),
                "p99": np.percentile(a, 99),
                "mean": a.mean(),
            }.items()
        }

    res = {
        "threads": LATENCY_THREADS,
        "n": len(texts),
        "api_predict_ms": timed(lambda t: client.post("/predict", json={"text": t})),
        "model_predict_ms": timed(model.predict),
        "api_predict_explain_ms": timed(lambda t: client.post("/predict", json={"text": t, "explain": True})),
    }
    api.app.dependency_overrides.clear()

    from . import baseline
    from .normalize import normalize

    pipe = baseline.load(corpus)
    res["baseline_predict_ms"] = timed(lambda t: pipe.predict_proba([normalize(t)]))
    return res


def run(corpus: str) -> dict:
    d = corpus_dirs(corpus)
    df = load_split(corpus)
    export_info = json.loads((d["results"] / "export.json").read_text())
    selected = export_info["selected"]

    preds = {"baseline": load_predictions(corpus, "baseline")}
    for key in ENCODERS:
        try:
            preds[key] = load_predictions(corpus, key)
        except FileNotFoundError:
            log.warning("no predictions for %s", key)
    temps = {}
    for name, f in (("onnx_fp32", "model.fp32.opt.onnx"), ("onnx_int8", "model.int8.onnx")):
        preds[name] = onnx_predictions(corpus, df, f, name)
        temps[name] = round(preds[name].attrs["temperature"], 4)

    # the served model uses the temperature fitted on its own (int8) validation logits
    cfg_path = d["models"] / "onnx" / "lahja.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["temperature"] = temps["onnx_int8"]
    cfg_path.write_text(json.dumps(cfg, indent=2))

    metrics = {"selected": selected, "temperatures": temps, "models": {}}
    for name, p in preds.items():
        metrics["models"][name] = {}
        for set_name, mask in eval_sets(df).items():
            rows = p[p.row.isin(df.index[mask])]
            raw = softmax(probs_of(rows, "logit_")) if f"logit_{CLASSES[0]}" in rows else None
            metrics["models"][name][set_name] = score(rows, probs_of(rows), raw)

    sel = metrics["models"][selected]["test"]
    metrics["reliability"] = {
        "uncalibrated": reliability(
            softmax(probs_of(preds[selected][preds[selected].split == "test"], "logit_")),
            preds[selected][preds[selected].split == "test"].label.map(CLASSES.index).to_numpy(),
        ),
        "calibrated": reliability(
            probs_of(preds[selected][preds[selected].split == "test"]),
            preds[selected][preds[selected].split == "test"].label.map(CLASSES.index).to_numpy(),
        ),
    }
    metrics["latency"] = latency(corpus, df)
    metrics["latency"]["cpu"] = cpu_name()
    metrics["by_source"] = by_source(df, preds)
    metrics["seeds"] = seed_spread(corpus, df)

    base_f1 = metrics["models"]["baseline"]["test"]["macro_f1"]
    gain = round((sel["macro_f1"] - base_f1) * 100, 2)
    qloss = round(
        (
            metrics["models"]["onnx_fp32"]["test"]["macro_f1"]
            - metrics["models"]["onnx_int8"]["test"]["macro_f1"]
        )
        * 100,
        2,
    )
    p95 = metrics["latency"]["api_predict_ms"]["p95"]
    gate = {
        "thresholds": THRESHOLDS,
        "gain_points": gain,
        "quantization_loss_points": qloss,
        "p95_ms": p95,
        "checks": {
            "gain": gain >= THRESHOLDS["min_gain_points"],
            "quantization": qloss <= THRESHOLDS["max_quantization_loss_points"],
            "latency": p95 < THRESHOLDS["max_p95_ms"],
        },
    }
    gate["passed"] = all(gate["checks"].values())
    metrics["gate"] = gate
    (d["results"] / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (d["results"] / "gate.json").write_text(json.dumps(gate, indent=2))
    log.info("gate: %s", json.dumps(gate))
    return metrics


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform

    return platform.processor() or platform.machine()


def seed_spread(corpus: str, df: pd.DataFrame) -> dict:
    """Test macro-F1 of each fine-tuned model over all the seeds that were trained."""
    path = corpus_dirs(corpus)["predictions"]
    out = {}
    for key in ENCODERS:
        files = sorted(path.glob(f"{key}.parquet")) + sorted(path.glob(f"{key}-s*.parquet"))
        if not files:
            continue
        scores = {}
        for f in files:
            p = pd.read_parquet(f)
            for set_name, mask in eval_sets(df).items():
                rows = p[p.row.isin(df.index[mask])]
                present = sorted(rows.label.unique())
                scores.setdefault(set_name, []).append(
                    float(f1_score(rows.label, rows.pred, labels=present, average="macro", zero_division=0))
                )
        out[key] = {
            s: {
                "n": len(v),
                "mean": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4),
                "values": [round(x, 4) for x in v],
            }
            for s, v in scores.items()
        }
    return out


BY_SOURCE_MIN_ROWS = 20


def by_source(df: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> dict:
    """Test macro-F1 split by corpus of origin (FLORES, SADA, Lahga), over the classes that have
    at least BY_SOURCE_MIN_ROWS test sentences in that corpus: SADA has 2 Maghrebi test sentences,
    and their F1 alone would move its macro-F1 by a quarter."""
    out = {}
    test = df[df.split == "test"]
    for name in ("baseline", *[k for k in ENCODERS if k in preds]):
        p = preds[name].set_index("row")
        out[name] = {}
        for src, g in test.groupby("source"):
            rows = p.loc[g.index]
            counts = rows.label.value_counts()
            present = sorted(counts[counts >= BY_SOURCE_MIN_ROWS].index)
            out[name][src] = {
                "n": int(len(rows)),
                "macro_f1": round(float(f1_score(rows.label, rows.pred, labels=present, average="macro")), 4),
                "accuracy": round(float((rows.label == rows.pred).mean()), 4),
                "classes": present,
            }
    return out
