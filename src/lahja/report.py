"""Evaluation report and model card, in English and French, generated from the result files
so the numbers in the documents are always the ones the last run produced."""

from __future__ import annotations

import json
import logging

import pandas as pd

from .config import CLASSES, DOCS, ENCODERS, FIGURES, SOURCES, corpus_dirs
from .data import load_split
from .figures import confusion_grid, reliability_plot

log = logging.getLogger(__name__)

NAMES = {
    "en": {
        "baseline": "Char n-grams + LR",
        "camelbert-da": "CAMeLBERT-DA",
        "marbert": "MARBERT",
        "onnx_fp32": "ONNX fp32",
        "onnx_int8": "ONNX int8",
        "classes": {"gulf": "Gulf", "levant": "Levant", "egypt": "Egypt", "maghreb": "Maghreb", "msa": "MSA"},
        "sets": {
            "test": "test (in-domain)",
            "tweets": "tweets (out-of-domain)",
            "madar": "MADAR (out-of-domain)",
        },
        "sources": {
            "flores200": "FLORES-200",
            "sada": "SADA",
            "lahga": "Lahga",
            "tweets": "Tweets (QADI labels)",
            "madar": "MADAR",
        },
    },
    "fr": {
        "baseline": "N-grammes + RL",
        "camelbert-da": "CAMeLBERT-DA",
        "marbert": "MARBERT",
        "onnx_fp32": "ONNX fp32",
        "onnx_int8": "ONNX int8",
        "classes": {
            "gulf": "Golfe",
            "levant": "Levant",
            "egypt": "Égypte",
            "maghreb": "Maghreb",
            "msa": "Arabe standard",
        },
        "sets": {
            "test": "test (même domaine)",
            "tweets": "tweets (hors domaine)",
            "madar": "MADAR (hors domaine)",
        },
        "sources": {
            "flores200": "FLORES-200",
            "sada": "SADA",
            "lahga": "Lahga",
            "tweets": "Tweets (étiquettes QADI)",
            "madar": "MADAR",
        },
    },
}


def pct(x: float, lang: str, digits: int = 1) -> str:
    s = f"{x * 100:.{digits}f}"
    return s.replace(".", ",") if lang == "fr" else s


def num(x: float, lang: str, digits: int = 1) -> str:
    s = f"{x:,.{digits}f}" if digits else f"{x:,.0f}"
    if lang == "fr":
        s = s.replace(",", " ").replace(".", ",")
    return s


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def load_all(corpus: str) -> dict:
    d = corpus_dirs(corpus)
    r = d["results"]
    out = {
        "metrics": json.loads((r / "metrics.json").read_text()),
        "steps": json.loads((r / "data_steps.json").read_text()),
        "export": json.loads((r / "export.json").read_text()),
        "baseline": json.loads((r / "train_baseline.json").read_text()),
        "train": {},
    }
    for key in ENCODERS:
        p = r / f"train_{key}.json"
        if p.exists():
            out["train"][key] = json.loads(p.read_text())
    return out


def make_figures(corpus: str, R: dict) -> dict[str, str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    m = R["metrics"]
    files = {}
    for lang in ("en", "fr"):
        N = NAMES[lang]
        labels = [N["classes"][c] for c in CLASSES]
        models = ["baseline", *[k for k in ENCODERS if k in m["models"]]]
        panels = [(N[k], m["models"][k]["test"]["confusion"]) for k in models]
        f = FIGURES / f"confusion_test_{lang}.png"
        confusion_grid(panels, labels, f, *(("Predicted", "True") if lang == "en" else ("Prédit", "Réel")))
        files[f"confusion_test_{lang}"] = f.name
        sel = m["selected"]
        for set_name in [s for s in m["models"][sel] if s != "test"]:
            f = FIGURES / f"confusion_{set_name}_{lang}.png"
            panels = [(N[k], m["models"][k][set_name]["confusion"]) for k in models]
            confusion_grid(
                panels, labels, f, *(("Predicted", "True") if lang == "en" else ("Prédit", "Réel"))
            )
            files[f"confusion_{set_name}_{lang}"] = f.name
        f = FIGURES / f"reliability_{lang}.png"
        names = (
            ("Before temperature scaling", "After temperature scaling")
            if lang == "en"
            else ("Avant mise à l'échelle", "Après mise à l'échelle")
        )
        reliability_plot(
            m["reliability"]["uncalibrated"],
            m["reliability"]["calibrated"],
            names,
            f,
            *(("Confidence", "Accuracy") if lang == "en" else ("Confiance", "Exactitude")),
        )
        files[f"reliability_{lang}"] = f.name
    return files


def data_tables(df: pd.DataFrame, lang: str) -> tuple[str, str]:
    N = NAMES[lang]
    c = df[df.split != "ood"]
    counts = (
        c.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(CLASSES), fill_value=0)
    )
    split_names = {
        "en": {"train": "train", "val": "validation", "test": "test"},
        "fr": {"train": "entraînement", "val": "validation", "test": "test"},
    }[lang]
    rows = []
    for s in ("train", "val", "test"):
        r = counts.loc[s]
        rows.append([split_names[s], *[num(r[k], lang, 0) for k in CLASSES], num(r.sum(), lang, 0)])
    for name in sorted(df.loc[df.split == "ood", "set"].unique()):
        r = (
            df[(df.split == "ood") & (df.set == name)]
            .label.value_counts()
            .reindex(list(CLASSES), fill_value=0)
        )
        rows.append(
            [
                N["sets"].get(name, name),
                *[num(r[k], lang, 0) if r[k] else "–" for k in CLASSES],
                num(r.sum(), lang, 0),
            ]
        )
    head = ["" if lang == "en" else "", *[N["classes"][k] for k in CLASSES], "total"]
    t1 = table(head, rows)

    by_src = (
        c.groupby(["source", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(CLASSES), fill_value=0)
    )
    rows = []
    for src, r in by_src.iterrows():
        s = SOURCES[src]
        rows.append([N["sources"][src], s.license, *[num(r[k], lang, 0) if r[k] else "–" for k in CLASSES]])
    t2 = table(
        [
            ("source" if lang == "en" else "source"),
            ("licence" if lang == "fr" else "license"),
            *[N["classes"][k] for k in CLASSES],
        ],
        rows,
        "ll" + "r" * len(CLASSES),
    )
    return t1, t2


def country_table(df: pd.DataFrame, lang: str) -> str:
    c = df[(df.split != "ood") & (df.label == "gulf")]
    counts = c.country.fillna("?").value_counts()
    names = {
        "en": {
            "SA": "Saudi Arabia",
            "AE": "UAE",
            "QA": "Qatar",
            "KW": "Kuwait",
            "BH": "Bahrain",
            "OM": "Oman",
            "?": 'unspecified (Lahga "gulf")',
        },
        "fr": {
            "SA": "Arabie saoudite",
            "AE": "Émirats",
            "QA": "Qatar",
            "KW": "Koweït",
            "BH": "Bahreïn",
            "OM": "Oman",
            "?": "non précisé (Lahga « gulf »)",
        },
    }[lang]
    rows = [[names[k], num(counts.get(k, 0), lang, 0)] for k in ("SA", "AE", "QA", "KW", "BH", "OM", "?")]
    return table(["", "n"], rows)


def write_evaluation(corpus: str, R: dict, df: pd.DataFrame, lang: str) -> str:
    T = TEXT[lang]
    N = NAMES[lang]
    m = R["metrics"]
    sel = m["selected"]
    models = ["baseline", *[k for k in ENCODERS if k in m["models"]]]
    sets = list(m["models"][sel].keys())
    g = m["gate"]
    ok = {True: T["yes"], False: T["no"]}

    out = [
        f"# {T['eval_title']}",
        "",
        T["lang_link"],
        "",
        T["eval_intro"].format(
            corpus=corpus,
            sel=N[sel],
            f1=pct(m["models"][sel]["test"]["macro_f1"], lang),
            base=pct(m["models"]["baseline"]["test"]["macro_f1"], lang),
            gain=num(g["gain_points"], lang),
        ),
        "",
    ]

    out += [f"## {T['gate']}", "", T["gate_intro"], ""]
    out.append(
        table(
            [T["criterion"], T["threshold"], T["measured"], T["passed"]],
            [
                [T["gain_row"], T["gain_thr"], f"{num(g['gain_points'], lang)} pts", ok[g["checks"]["gain"]]],
                [
                    T["quant_row"],
                    T["quant_thr"],
                    f"{num(g['quantization_loss_points'], lang, 2)} pt",
                    ok[g["checks"]["quantization"]],
                ],
                [T["lat_row"], T["lat_thr"], f"{num(g['p95_ms'], lang)} ms", ok[g["checks"]["latency"]]],
            ],
            "llrl",
        )
    )
    out += ["", T["gate_pass"] if g["passed"] else T["gate_fail"], ""]

    out += [f"## {T['comparison']}", "", T["comparison_intro"], ""]
    head = [T["model"], *[f"macro-F1 {N['sets'].get(s, s)}" for s in sets], T["acc_test"], "ECE test"]
    rows = []
    for k in models + ["onnx_fp32", "onnx_int8"]:
        mm = m["models"][k]
        rows.append(
            [
                N[k] + (" ★" if k == sel else ""),
                *[pct(mm[s]["macro_f1"], lang) for s in sets],
                pct(mm["test"]["accuracy"], lang),
                num(mm["test"]["ece"], lang, 3),
            ]
        )
    out += [
        table(head, rows),
        "",
        T["star"].format(
            sel=N[sel],
            scores=", ".join(f"{N[k]} {pct(v, lang)}" for k, v in R["export"]["val_macro_f1"].items()),
        ),
        "",
    ]
    seeds = m.get("seeds", {})
    if seeds and max(v["test"]["n"] for v in seeds.values()) > 1:
        out += [T["seeds_intro"], ""]
        head = [T["model"], T["n_seeds"], *[f"macro-F1 {N['sets'].get(s, s)}" for s in sets]]
        rows = [
            [
                N[k],
                str(v["test"]["n"]),
                *[f"{pct(v[s]['mean'], lang)} ± {pct(v[s]['std'], lang)}" for s in sets],
            ]
            for k, v in seeds.items()
        ]
        out += [table(head, rows), ""]

    out += [f"### {T['per_class']}", ""]
    head = [T["class"]] + [f"{N[k]} P / R / F1" for k in models]
    rows = []
    for c in CLASSES:
        row = [N["classes"][c]]
        for k in models:
            pc = m["models"][k]["test"]["per_class"][c]
            row.append(
                f"{pct(pc['precision'], lang, 0)} / {pct(pc['recall'], lang, 0)} / {pct(pc['f1'], lang, 0)}"
            )
        rows.append(row)
    out += [
        table(head, rows),
        "",
        f"![{T['cm_alt']}](figures/confusion_test_{lang}.png)",
        "",
        T["cm_note"],
        "",
    ]

    out += [f"### {T['by_source']}", "", T["by_source_intro"], ""]
    srcs = sorted(m["by_source"]["baseline"])
    head = [
        T["model"],
        *[f"{N['sources'][s]} (n = {num(m['by_source']['baseline'][s]['n'], lang, 0)})" for s in srcs],
    ]
    rows = [[N[k], *[pct(m["by_source"][k][s]["macro_f1"], lang) for s in srcs]] for k in models]
    out += [table(head, rows), "", T["by_source_note"], ""]

    out += [f"## {T['ood']}", "", T["ood_intro"], ""]
    for s in [x for x in sets if x != "test"]:
        mm = {k: m["models"][k][s] for k in models}
        present = list(mm[sel]["per_class"])
        head = [T["model"], "macro-F1", *[f"F1 {N['classes'][c]}" for c in present]]
        rows = [
            [N[k], pct(mm[k]["macro_f1"], lang), *[pct(mm[k]["per_class"][c]["f1"], lang) for c in present]]
            for k in models
        ]
        out += [
            f"### {N['sets'][s]}",
            "",
            T.get(f"ood_{s}", ""),
            "",
            table(head, rows),
            "",
            f"![{T['cm_alt']}](figures/confusion_{s}_{lang}.png)",
            "",
        ]
    if "madar" not in sets:
        out += [T["madar_missing"], ""]

    temps = m["temperatures"]
    ms = m["models"][sel]["test"]
    out += [
        f"## {T['calibration']}",
        "",
        T["calibration_intro"].format(
            before=num(ms.get("ece_uncalibrated", float("nan")), lang, 3),
            after=num(ms["ece"], lang, 3),
            t=num(R["train"][sel]["temperature"], lang, 2),
            t8=num(temps["onnx_int8"], lang, 2),
        ),
        "",
        f"![{T['rel_alt']}](figures/reliability_{lang}.png)",
        "",
    ]

    q32, q8 = m["models"]["onnx_fp32"]["test"]["macro_f1"], m["models"]["onnx_int8"]["test"]["macro_f1"]
    sizes = R["export"]["onnx_mb"]
    out += [
        f"## {T['quant']}",
        "",
        T["quant_intro"].format(
            f32=pct(q32, lang, 2),
            f8=pct(q8, lang, 2),
            loss=num(g["quantization_loss_points"], lang, 2),
            mb32=num(sizes.get("model.fp32.opt.onnx", 0), lang, 0),
            mb8=num(sizes.get("model.int8.onnx", 0), lang, 0),
        ),
        "",
    ]

    L = m["latency"]
    out += [
        f"## {T['latency']}",
        "",
        T["latency_intro"].format(n=L["n"], threads=L["threads"], cpu=L.get("cpu", "?")),
        "",
    ]
    rows = [
        [T[k], num(L[k]["p50"], lang), num(L[k]["p95"], lang)]
        for k in ("api_predict_ms", "model_predict_ms", "api_predict_explain_ms", "baseline_predict_ms")
    ]
    out += [table([T["path"], "p50 (ms)", "p95 (ms)"], rows), "", T["latency_note"], ""]

    out += [f"## {T['ngrams']}", "", T["ngrams_intro"], ""]
    grams = R["baseline"]["top_ngrams"]
    rows = [[N["classes"][c], " ".join(f"`{x.strip()}`" for x in grams[c][:10])] for c in CLASSES]
    out += [table([T["class"], T["ngrams_col"]], rows, "ll"), ""]

    out += [f"## {T['training']}", ""]
    rows = []
    for k, t in R["train"].items():
        hist = ", ".join(pct(h["val_macro_f1"], lang) for h in t["history"])
        rows.append(
            [N[k], t["device"], num(t["train_seconds"] / 60, lang, 0) + " min", str(t["best_epoch"]), hist]
        )
    out += [
        table([T["model"], T["device"], T["duration"], T["best_epoch"], T["val_history"]], rows, "lllrl"),
        "",
        T["training_note"].format(C=R["baseline"]["C"]),
        "",
    ]
    return "\n".join(out)


def write_model_card(corpus: str, R: dict, df: pd.DataFrame, lang: str) -> str:
    T = TEXT[lang]
    N = NAMES[lang]
    m = R["metrics"]
    sel = m["selected"]
    ms = m["models"]["onnx_int8"]
    t = R["train"][sel]
    rows = [
        [
            N["classes"][c],
            pct(ms["test"]["per_class"][c]["precision"], lang),
            pct(ms["test"]["per_class"][c]["recall"], lang),
            pct(ms["test"]["per_class"][c]["f1"], lang),
            num(ms["test"]["per_class"][c]["n"], lang, 0),
        ]
        for c in CLASSES
    ]
    ood = [s for s in ms if s != "test"]
    d1, _ = data_tables(df, lang)
    body = T["card"].format(
        sel=N[sel],
        base=t["base"],
        revision=(t["revision"] or "local")[:7],
        f1=pct(ms["test"]["macro_f1"], lang),
        acc=pct(ms["test"]["accuracy"], lang),
        ece=num(ms["test"]["ece"], lang, 3),
        per_class=table([T["class"], T["precision"], T["recall"], "F1", "n"], rows),
        ood_lines="\n".join(
            f"- {N['sets'][s]} : macro-F1 {pct(ms[s]['macro_f1'], lang)}"
            if lang == "fr"
            else f"- {N['sets'][s]}: macro-F1 {pct(ms[s]['macro_f1'], lang)}"
            for s in ood
        ),
        data_table=d1,
        countries=country_table(df, lang),
        p95=num(m["latency"]["api_predict_ms"]["p95"], lang),
        n_train=num(t["n_train"], lang, 0),
    )
    return body


def run(corpus: str) -> None:
    R = load_all(corpus)
    df = load_split(corpus)
    make_figures(corpus, R)
    DOCS.mkdir(exist_ok=True)
    suffix = "" if corpus == "open" else f"_{corpus}"
    for lang in ("en", "fr"):
        ext = ".md" if lang == "en" else ".fr.md"
        (DOCS / f"evaluation{suffix}{ext}").write_text(write_evaluation(corpus, R, df, lang) + "\n")
        if corpus == "open":
            (DOCS / f"model_card{ext}").write_text(write_model_card(corpus, R, df, lang) + "\n")
    log.info("report written to %s", DOCS)


from .report_text import TEXT  # noqa: E402
