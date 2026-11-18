"""Publication: only when the acceptance thresholds are met, copy the quantised model to
models/release/ (what the API serves) and pick the demo's example sentences."""

from __future__ import annotations

import json
import logging
import shutil

import numpy as np

from .config import CLASSES, MODELS, SEED, corpus_dirs
from .data import load_split
from .ingest import sha256
from .predictions import load_predictions

log = logging.getLogger(__name__)
WEB_EXAMPLES = MODELS.parent / "src" / "lahja" / "web" / "examples.json"
PER_CLASS = 4


def pick_examples(corpus: str) -> dict[str, list[str]]:
    """Test sentences of 5 to 18 words that the served model classifies correctly, spread over
    the source corpora. Tweets are never used: their text cannot be republished."""
    df = load_split(corpus)
    p = load_predictions(corpus, "onnx_int8").set_index("row")
    test = df[(df.split == "test") & df.n_words.between(5, 18)]
    test = test[p.loc[test.index, "pred"].values == test.label.values]
    rng = np.random.default_rng(SEED)
    out = {}
    for c in CLASSES:
        g = test[test.label == c]
        g = g.iloc[rng.permutation(len(g))]
        # one sentence per source in turn, so a class is not shown through a single corpus
        g = g.assign(_r=g.groupby("source").cumcount()).sort_values("_r", kind="stable")
        out[c] = [t.strip() for t in g.text.head(PER_CLASS)]
    return out


def run(corpus: str) -> int:
    d = corpus_dirs(corpus)
    gate = json.loads((d["results"] / "gate.json").read_text())
    if not gate["passed"]:
        failed = [k for k, v in gate["checks"].items() if not v]
        log.error("acceptance thresholds not met (%s): nothing published", ", ".join(failed))
        return 1
    if corpus == "madar":
        log.warning(
            "MADAR licence: internal research and evaluation only. "
            "Results are kept, the model is not published."
        )
        return 0

    src = d["models"] / "onnx"
    release = MODELS / "release"
    if release.exists():
        shutil.rmtree(release)
    release.mkdir(parents=True)
    for name in ("model.int8.onnx", "tokenizer.json", "lahja.json"):
        shutil.copy2(src / name, release / name)
    (release / "SHA256SUMS").write_text(
        "".join(f"{sha256(release / n)}  {n}\n" for n in ("model.int8.onnx", "tokenizer.json", "lahja.json"))
    )
    WEB_EXAMPLES.write_text(json.dumps(pick_examples(corpus), ensure_ascii=False, indent=2) + "\n")
    log.info("published to %s", release)
    return 0
