"""The four data steps. Each one reads the previous Parquet file and writes its own, so any
step can be re-run alone: ingest -> normalize -> label -> split."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import ingest, labels
from .config import (
    CAP_PER_SOURCE_CLASS,
    CLASSES,
    CORPORA,
    MAX_WORDS,
    MIN_WORDS,
    N_FOLDS,
    OOD_SAMPLE_PER_CLASS,
    OOD_SETS,
    SEED,
    TEST_FOLD,
    VAL_FOLD,
    corpus_dirs,
)
from .normalize import normalize, word_count

log = logging.getLogger(__name__)


def _paths(corpus: str):
    d = corpus_dirs(corpus)
    d["interim"].mkdir(parents=True, exist_ok=True)
    d["results"].mkdir(parents=True, exist_ok=True)
    return d


def step_ingest(corpus: str) -> pd.DataFrame:
    d = _paths(corpus)
    frames = []
    for name in CORPORA[corpus]:
        frames.append(ingest.load_source(name).assign(set="corpus"))
    for name in OOD_SETS:
        if name in CORPORA[corpus]:
            continue
        if not ingest.available(name):
            log.warning("%s not available, skipped as an out-of-domain set", name)
            continue
        frames.append(ingest.load_source(name).assign(set=name))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(d["interim"] / "ingested.parquet", index=False)
    return df


def step_normalize(corpus: str) -> pd.DataFrame:
    d = _paths(corpus)
    df = pd.read_parquet(d["interim"] / "ingested.parquet")
    df["text_norm"] = df.text.map(normalize)
    df["n_words"] = df.text_norm.map(word_count)
    df.to_parquet(d["interim"] / "normalized.parquet", index=False)
    return df


def _class_of(row) -> str | None:
    src, v = row.source, row.variety
    if src == "flores200":
        return labels.FLORES.get(v, (None, None))[0]
    if src == "sada":
        return labels.SADA.get(v, (None, None))[0]
    if src == "lahga":
        return labels.LAHGA.get(v, (None, None))[0]
    if src == "tweets":
        return labels.country_label(v)
    if src == "madar":
        return labels.madar_label(v)[0]
    raise ValueError(src)


def step_label(corpus: str) -> pd.DataFrame:
    d = _paths(corpus)
    df = pd.read_parquet(d["interim"] / "normalized.parquet")
    counts = {"ingested": df.groupby(["set", "source"]).size().to_dict()}

    df["label"] = df.apply(_class_of, axis=1)
    df = df[df.label.notna()]
    counts["mapped"] = df.groupby(["set", "source"]).size().to_dict()

    # FLORES "Najdi" copies the MSA sentence verbatim in about 70 % of cases. Those copies
    # carry no dialect signal and would be labelled Gulf and MSA at once: drop them.
    fl = df[df.source == "flores200"]
    msa = fl[fl.variety == "arb_Arab"].set_index("group_id").text_norm
    ars = fl[fl.variety == "ars_Arab"]
    copies = ars.index[ars.text_norm.values == msa.reindex(ars.group_id).values]
    df = df.drop(index=copies)
    counts["flores_najdi_copies_dropped"] = int(len(copies))

    df = df[(df.n_words >= MIN_WORDS) & (df.n_words <= MAX_WORDS)]
    counts["length_ok"] = df.groupby(["set", "source"]).size().to_dict()

    # Same normalised text under two classes is ambiguous by construction: drop every copy.
    corpus_rows = df[df.set == "corpus"]
    n_labels = corpus_rows.groupby("text_norm").label.nunique()
    ambiguous = set(n_labels.index[n_labels > 1])
    df = df[~((df.set == "corpus") & df.text_norm.isin(ambiguous))]
    counts["ambiguous_texts_dropped"] = len(ambiguous)
    # Exact duplicates inside the corpus: keep the first.
    df = df[~((df.set == "corpus") & df.duplicated(["set", "text_norm"]))]

    rng = np.random.default_rng(SEED)
    parts = []
    for (set_, _source, _label), g in df.groupby(["set", "source", "label"], sort=True):
        cap = CAP_PER_SOURCE_CLASS if set_ == "corpus" else OOD_SAMPLE_PER_CLASS if set_ == "tweets" else None
        if cap is not None and len(g) > cap:
            g = spread_sample(g, cap, rng)
        parts.append(g)
    df = pd.concat(parts)

    # An out-of-domain sentence that also occurs in the training corpus is not out of domain.
    train_texts = set(df.loc[df.set == "corpus", "text_norm"])
    leak = (df.set != "corpus") & df.text_norm.isin(train_texts)
    counts["ood_overlap_dropped"] = int(leak.sum())
    df = df[~leak & ~((df.set != "corpus") & df.duplicated(["set", "text_norm"]))]

    df = df.reset_index(drop=True)
    df.to_parquet(d["interim"] / "labeled.parquet", index=False)
    (d["results"] / "data_steps.json").write_text(
        json.dumps({k: _jsonable(v) for k, v in counts.items()}, indent=2, ensure_ascii=False)
    )
    return df


def spread_sample(g: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw n rows taking groups in turn: one random row from every group, then a second, and
    so on. A few long TV shows would otherwise fill the sample, and with them a whole fold."""
    shuffled = g.iloc[rng.permutation(len(g))]
    rank = shuffled.groupby("group_id").cumcount().to_numpy()
    picked = shuffled.iloc[np.argsort(rank, kind="stable")[:n]]
    return picked.sort_index()


def _jsonable(v):
    if isinstance(v, dict):
        return {"/".join(k) if isinstance(k, tuple) else k: int(x) for k, x in v.items()}
    return v


def assign_splits(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Grouped, class-stratified 10-fold split: fold 0 is test, fold 1 validation, the rest train.

    Rows sharing a group_id (all translations of one source sentence, all segments of one TV
    show) always land in the same split.
    """
    split = pd.Series("train", index=df.index)
    y = df.label.map({c: i for i, c in enumerate(CLASSES)})
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for k, (_, idx) in enumerate(folds.split(df, y, groups=df.group_id)):
        if k == TEST_FOLD:
            split.iloc[idx] = "test"
        elif k == VAL_FOLD:
            split.iloc[idx] = "val"
    return split


def step_split(corpus: str) -> pd.DataFrame:
    d = _paths(corpus)
    df = pd.read_parquet(d["interim"] / "labeled.parquet")
    df["split"] = "ood"
    is_corpus = df.set == "corpus"
    parts = []
    # Split each source separately: their groups never overlap, and per-source folds keep the
    # FLORES / SADA mix similar across train, validation and test.
    for _, g in df[is_corpus].groupby("source"):
        parts.append(assign_splits(g))
    df.loc[is_corpus, "split"] = pd.concat(parts)
    df.to_parquet(d["interim"] / "split.parquet", index=False)
    fp = fingerprint(df)
    (d["results"] / "split_fingerprint.txt").write_text(fp + "\n")
    log.info("split fingerprint %s", fp)

    summary = (
        df.groupby(["set", "source", "split", "label"])
        .size()
        .rename("n")
        .reset_index()
        .to_dict(orient="records")
    )
    (d["results"] / "data_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("\n%s", df.groupby(["set", "split", "label"]).size().unstack(fill_value=0))
    return df


def fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 of the split's content (source, group, label, split, text), independent of the
    Parquet encoding: two machines that rebuild the same split print the same value."""
    import hashlib

    lines = sorted(df.source + "|" + df.group_id + "|" + df.label + "|" + df.split + "|" + df.text_norm)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def load_split(corpus: str) -> pd.DataFrame:
    return pd.read_parquet(corpus_dirs(corpus)["interim"] / "split.parquet")
