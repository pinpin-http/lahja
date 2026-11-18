"""Paths, classes and pinned data sources. Everything a run depends on is declared here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"

# v1 classes, in the order used for logits, tables and figures.
CLASSES: tuple[str, ...] = ("gulf", "levant", "egypt", "maghreb", "msa")

SEED = 13
MIN_WORDS, MAX_WORDS = 3, 60
VAL_FOLD, TEST_FOLD, N_FOLDS = 1, 0, 10  # 10 grouped folds: one for test, one for validation, 80 % train

# At most this many rows per (source, class) after filtering. SADA alone has ~190k Gulf
# segments; without a cap the Gulf class would be TV transcripts and little else.
CAP_PER_SOURCE_CLASS = 6000

# Corpus used for training: "open" (FLORES-200 + SADA + Lahga) or "madar" (needs the manual download).
CORPUS = os.environ.get("LAHJA_CORPUS", "open")

# Pre-trained encoders, pinned to a Hub revision.
ENCODERS = {
    "camelbert-da": ("CAMeL-Lab/bert-base-arabic-camelbert-da", "231698eab9ebf0ae7b518a64277b81b2fe829f2d"),
    "marbert": ("UBC-NLP/MARBERT", "88e1fa192dd723cf0b3563500aec46209762eb22"),
}

HF = "https://huggingface.co/datasets"


@dataclass(frozen=True)
class Source:
    name: str
    files: dict[str, tuple[str | None, str | None]]  # local file -> (url, sha256)
    license: str
    homepage: str
    role: str  # "train" (part of a training corpus) or "ood" (out-of-domain test only)
    notes: str = ""
    manual: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return RAW / self.name


SOURCES: dict[str, Source] = {
    "flores200": Source(
        name="flores200",
        files={
            "flores200_dataset.tar.gz": (
                "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
                "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6",
            )
        },
        license="CC BY-SA 4.0",
        homepage="https://github.com/facebookresearch/flores/tree/main/flores200",
        role="train",
    ),
    "sada": Source(
        name="sada",
        files={
            f"{split}.csv": (
                f"{HF}/khaledalganem/sada2022/resolve/f45d9ad8e60aa3613229322e0712eb5874ff94f6/{split}.csv",
                sha,
            )
            for split, sha in [
                ("train", "6cece5e67a40ebceb9dd5236721c37eb5357c4a928f680a2f94b75200deb4cf4"),
                ("valid", "6dbc7bf6f66932c64421e4efcf795d71eef9d235dab8d34590b31d97f861dfa2"),
                ("test", "af09e5a10f5bb33f60cf43c4587366e76a403e1419c6088a92de4bd1f705d622"),
            ]
        },
        license="CC BY-NC-SA 4.0",
        homepage="https://www.kaggle.com/datasets/sdaiancai/sada2022",
        role="train",
        notes="Transcript CSVs of SDAIA's SADA corpus, mirrored on the Hugging Face Hub (no audio).",
    ),
    "lahga": Source(
        name="lahga",
        files={
            name: (f"{HF}/lahga/arabic-dialects/resolve/3bacbff0dede7f42c6f9d08a7ca56b7ca6bcdaa8/{name}", sha)
            for name, sha in [
                ("examples.jsonl", "d57afb2ae7162237f40dd1b73b7889d5929a3c1c02a4852dc4cecf636335a287"),
                ("entries.jsonl", "99a3d0176253a2718ce59055eb590ad6e894b146f4f62977ab8f314b15b847d1"),
            ]
        },
        license="CC BY-SA 4.0",
        homepage="https://lahga.fyi",
        role="train",
        notes="Example sentences written in each dialect, and the MSA definition of each headword.",
    ),
    "tweets": Source(
        name="tweets",
        files={
            "train-00000-of-00001.parquet": (
                f"{HF}/amgadhasan/arabic_tweets_dialects/resolve/423e3b7f3809c91de94332da20cd2226a82b21d5"
                "/data/train-00000-of-00001.parquet",
                "b015d38234be452c2440082e2ceb06533338d38aeb14aa0fd27aaec1d541b264",
            )
        },
        license="tweets: X terms of service; labels: QADI (QCRI)",
        homepage="https://huggingface.co/datasets/amgadhasan/arabic_tweets_dialects",
        role="ood",
        notes="Country-labelled tweets (QADI label set). Metrics only: no tweet text is published.",
    ),
    "madar": Source(
        name="madar",
        files={},
        license="MADAR licence: internal research and evaluation only, no redistribution",
        homepage="https://camel.abudhabi.nyu.edu/madar-parallel-corpus/",
        role="train",
        manual=True,
        notes="Download through the CAMeL Lab form and unzip into data/raw/madar/.",
    ),
}

CORPORA = {"open": ("flores200", "sada", "lahga"), "madar": ("madar",)}
OOD_SETS = ("tweets", "madar")
OOD_SAMPLE_PER_CLASS = 1000  # tweets are subsampled to this many per class


def corpus_dirs(corpus: str = CORPUS) -> dict[str, Path]:
    return {
        "interim": DATA / "interim" / corpus,
        "models": MODELS / corpus,
        "results": RESULTS / corpus,
        "predictions": RESULTS / corpus / "predictions",
    }
