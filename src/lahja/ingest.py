"""Download step: fetch each source, check its SHA-256, parse it into one common table.

Common columns: source, set, group_id, variety, country, text.
`group_id` is what the split keeps together: the source sentence for parallel corpora
(FLORES, MADAR), the TV show for SADA.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
from pathlib import Path

import pandas as pd
import requests

from . import labels
from .config import SOURCES, Source

log = logging.getLogger(__name__)
USER_AGENT = "lahja/0.1 (+https://github.com/pinpin-http/lahja)"


class ChecksumError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(source: Source) -> list[Path]:
    """Download missing files, then verify every file against its pinned hash."""
    source.dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname, (url, expected) in source.files.items():
        path = source.dir / fname
        if not path.exists():
            log.info("downloading %s", url)
            tmp = path.with_suffix(path.suffix + ".part")
            with requests.get(url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT}) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.rename(path)
        got = sha256(path)
        if expected and got != expected:
            raise ChecksumError(f"{path}: sha256 {got}, expected {expected}")
        paths.append(path)
    return paths


# --- parsers -------------------------------------------------------------------------------


def parse_flores(paths: list[Path]) -> pd.DataFrame:
    rows = []
    with tarfile.open(paths[0]) as tar:
        for split, offset in (("dev", 0), ("devtest", 997)):
            for code in labels.FLORES:
                member = tar.extractfile(f"./flores200_dataset/{split}/{code}.{split}")
                for i, line in enumerate(io.TextIOWrapper(member, encoding="utf-8")):
                    rows.append(("flores200", f"flores:{offset + i}", code, line.strip()))
    df = pd.DataFrame(rows, columns=["source", "group_id", "variety", "text"])
    df["country"] = df.variety.map(lambda v: labels.FLORES[v][1])
    return df


def parse_sada(paths: list[Path]) -> pd.DataFrame:
    cols = ["ShowName", "SpeakerDialect", "GroundTruthText", "ProcessedText"]
    df = pd.concat([pd.read_csv(p, usecols=cols, dtype=str) for p in paths], ignore_index=True)
    text = df.GroundTruthText.fillna(df.ProcessedText).fillna("")
    return pd.DataFrame(
        {
            "source": "sada",
            "group_id": "sada:" + df.ShowName.fillna("?"),
            "variety": df.SpeakerDialect.fillna("?"),
            "text": text,
            "country": df.SpeakerDialect.map(lambda v: labels.SADA.get(v, (None, None))[1]),
        }
    )


def parse_lahga(paths: list[Path]) -> pd.DataFrame:
    """Dialect example sentences, plus each headword's definition as an MSA sentence.

    Rows are grouped by headword, so a meaning's MSA definition and its dialect examples
    always share a split. Forms flagged `needs_review` are skipped."""
    by_name = {p.name: p for p in paths}
    examples = [json.loads(line) for line in by_name["examples.jsonl"].open(encoding="utf-8")]
    rows = [
        (f"lahga:{r['word_id']}", r["dialect"], r["text"], labels.LAHGA.get(r["dialect"], (None, None))[1])
        for r in examples
        if not r.get("needs_review")
    ]
    seen = set()
    for line in by_name["entries.jsonl"].open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("definition") and r["word_id"] not in seen:
            seen.add(r["word_id"])
            rows.append((f"lahga:{r['word_id']}", "msa_definition", r["definition"], None))
    df = pd.DataFrame(rows, columns=["group_id", "variety", "text", "country"])
    df.insert(0, "source", "lahga")
    return df


def parse_tweets(paths: list[Path]) -> pd.DataFrame:
    df = pd.read_parquet(paths[0])
    return pd.DataFrame(
        {
            "source": "tweets",
            "group_id": [f"tweets:{i}" for i in range(len(df))],
            "variety": df.dialect.str.upper(),
            "text": df.text.fillna(""),
            "country": df.dialect.str.upper(),
        }
    )


def find_madar_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.tsv") if "lexicon" not in p.name.lower())


def parse_madar(paths: list[Path]) -> pd.DataFrame:
    """MADAR corpus files: one TSV per city with columns sentID.BTEC, split, lang, sent.

    Column names are matched loosely; when the city column is missing, the city is read from
    the file name (MADAR.corpus.Cairo.tsv). English and French rows are skipped.
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep="\t", dtype=str, quoting=3).fillna("")
        cols = {c.lower(): c for c in df.columns}
        id_col = next((cols[c] for c in cols if c.startswith("sentid") or c == "id"), None)
        text_col = next((cols[c] for c in cols if c in ("sent", "sentence", "text")), None)
        lang_col = next((cols[c] for c in cols if c in ("lang", "city", "dialect")), None)
        if id_col is None or text_col is None:
            raise ValueError(f"{p}: no sentence id / text column in {list(df.columns)}")
        city = df[lang_col] if lang_col else pd.Series(p.stem.split(".")[-1], index=df.index)
        frames.append(
            pd.DataFrame(
                {"group_id": "madar:" + df[id_col], "variety": city.str.strip(), "text": df[text_col]}
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df = df[~df.variety.str.upper().isin(["EN", "ENG", "ENGLISH", "FR", "FRA", "FRENCH"])]
    df = df.drop_duplicates(["group_id", "variety"])
    df["source"] = "madar"
    df["country"] = df.variety.map(lambda v: labels.madar_label(v)[1])
    return df.reset_index(drop=True)


PARSERS = {
    "flores200": parse_flores,
    "sada": parse_sada,
    "lahga": parse_lahga,
    "tweets": parse_tweets,
    "madar": parse_madar,
}


def available(name: str) -> bool:
    src = SOURCES[name]
    return bool(find_madar_files(src.dir)) if src.manual else True


def load_source(name: str) -> pd.DataFrame:
    src = SOURCES[name]
    if src.manual:
        paths = find_madar_files(src.dir)
        if not paths:
            raise FileNotFoundError(
                f"No MADAR files under {src.dir}. Request the corpus at {src.homepage}, then unzip it there."
            )
        record_manual_checksums(src, paths)
    else:
        paths = fetch(src)
    df = PARSERS[name](paths)
    log.info("%s: %d rows", name, len(df))
    return df


def record_manual_checksums(src: Source, paths: list[Path]) -> None:
    """Manual downloads have no pinned hash: the first run writes one, later runs check it."""
    lock = src.dir / "SHA256SUMS"
    current = {str(p.relative_to(src.dir)): sha256(p) for p in paths}
    if lock.exists():
        known = dict(line.split("  ", 1)[::-1] for line in lock.read_text().splitlines() if line)
        for name, digest in current.items():
            if name in known and known[name] != digest:
                raise ChecksumError(f"{name}: sha256 {digest}, recorded {known[name]}")
    lock.write_text("".join(f"{d}  {n}\n" for n, d in sorted(current.items())))
