"""Fine-tune a pre-trained Arabic encoder (CAMeLBERT-DA or MARBERT) on the five classes.

Meant for a GPU (a cloud instance; Colab or Kaggle work too). Standard recipe: full
fine-tuning with AdamW, linear warm-up then decay, mixed precision (bf16 where the GPU
supports it, fp16 otherwise), cross-entropy weighted by inverse class frequency.
`--seed` trains extra runs whose test scores give the spread between seeds; only the main
seed keeps its weights.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import nullcontext

import numpy as np
from sklearn.metrics import f1_score

from .config import CLASSES, ENCODERS, SEED, corpus_dirs
from .data import load_split
from .metrics import fit_temperature, softmax
from .predictions import save_predictions

log = logging.getLogger(__name__)

MAX_LEN = 128
BATCH = 32
EVAL_BATCH = 128
EPOCHS = 3
LR = 3e-5
WARMUP = 0.06
WEIGHT_DECAY = 0.01
LOG_EVERY = 100


def seed_everything(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def length_batches(lengths: np.ndarray, batch: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle, sort by length inside chunks of 50 batches, cut, then shuffle the batches:
    rows of similar length end up together and padding stays short."""
    order = rng.permutation(len(lengths))
    chunk = batch * 50
    batches = []
    for i in range(0, len(order), chunk):
        part = order[i : i + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[j : j + batch] for j in range(0, len(part), batch)]
    rng.shuffle(batches)
    return batches


def collate(ids: list[list[int]], idx: np.ndarray, pad_id: int, device):
    import torch

    L = max(len(ids[i]) for i in idx)
    x = np.full((len(idx), L), pad_id, dtype=np.int64)
    m = np.zeros((len(idx), L), dtype=np.int64)
    for r, i in enumerate(idx):
        x[r, : len(ids[i])] = ids[i]
        m[r, : len(ids[i])] = 1
    return torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)


def autocast_for(device):
    import torch

    if device.type != "cuda":
        return None, nullcontext
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return dtype, lambda: torch.autocast("cuda", dtype=dtype)


def predict_logits(model, ids: list[list[int]], pad_id: int, device) -> np.ndarray:
    import torch

    _, ctx = autocast_for(device)
    model.eval()
    order = np.argsort([len(t) for t in ids], kind="stable")
    out = np.zeros((len(ids), len(CLASSES)), dtype=np.float32)
    with torch.inference_mode(), ctx():
        for j in range(0, len(order), EVAL_BATCH):
            idx = order[j : j + EVAL_BATCH]
            x, m = collate(ids, idx, pad_id, device)
            out[idx] = model(input_ids=x, attention_mask=m).logits.float().cpu().numpy()
    return out


def train(
    corpus: str, key: str, epochs: int | None = None, limit: int | None = None, seed: int = SEED
) -> dict:
    import torch
    import transformers
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    epochs = epochs or EPOCHS
    seed_everything(seed)
    d = corpus_dirs(corpus)
    name, revision = ENCODERS[key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log.warning("no GPU found: fine-tuning on CPU will take hours")
    main_run = seed == SEED
    tag = key if main_run else f"{key}-s{seed}"
    out = d["models"] / key if main_run else d["models"] / "seeds" / tag

    df = load_split(corpus)
    tr = df[df.split == "train"]
    if limit:
        tr = tr.sample(min(limit, len(tr)), random_state=seed)
    va = df[df.split == "val"]
    rest = df[df.split != "train"]

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        name,
        revision=revision,
        num_labels=len(CLASSES),
        id2label=dict(enumerate(CLASSES)),
        label2id={c: i for i, c in enumerate(CLASSES)},
        use_safetensors=False
        if revision
        else None,  # pinned revision: its own .bin, not a later conversion PR
    ).to(device)

    def encode(texts):
        return tok(list(texts), truncation=True, max_length=MAX_LEN)["input_ids"]

    ids_tr, ids_rest = encode(tr.text_norm), encode(rest.text_norm)
    y_tr = torch.tensor(tr.label.map(CLASSES.index).values, device=device)
    va_pos = np.flatnonzero((rest.split == "val").values)
    y_va = va.label.map(CLASSES.index).values

    counts = np.bincount(y_tr.cpu().numpy(), minlength=len(CLASSES))
    weights = torch.tensor(len(y_tr) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(groups, lr=LR)
    lengths = np.array([len(t) for t in ids_tr])
    rng = np.random.default_rng(seed)
    steps = epochs * int(np.ceil(len(ids_tr) / BATCH))
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    amp_dtype, ctx = autocast_for(device)
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    history, best = [], (-1.0, 0)
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, seen, running = time.perf_counter(), 0, 0.0
        for b, idx in enumerate(length_batches(lengths, BATCH, rng), 1):
            x, m = collate(ids_tr, idx, tok.pad_token_id, device)
            with ctx():
                logits = model(input_ids=x, attention_mask=m).logits
            loss = loss_fn(logits.float(), y_tr[idx])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            seen += len(idx)
            running += loss.item()
            if b % LOG_EVERY == 0:
                rate = seen / (time.perf_counter() - t0)
                log.info(
                    "%s epoch %d batch %d  loss %.4f  %.0f rows/s", tag, epoch, b, running / LOG_EVERY, rate
                )
                running = 0.0
        logits_va = predict_logits(model, [ids_rest[i] for i in va_pos], tok.pad_token_id, device)
        f1 = f1_score(y_va, logits_va.argmax(1), average="macro")
        history.append(
            {"epoch": epoch, "val_macro_f1": round(float(f1), 4), "seconds": round(time.perf_counter() - t0)}
        )
        log.info("%s epoch %d  val macro-F1 %.4f", tag, epoch, f1)
        if f1 > best[0]:
            best = (f1, epoch)
            model.save_pretrained(out, safe_serialization=True)
            tok.save_pretrained(out)
    train_seconds = time.perf_counter() - t_start

    model = AutoModelForSequenceClassification.from_pretrained(out).to(device)
    logits = predict_logits(model, ids_rest, tok.pad_token_id, device)
    T = fit_temperature(logits[va_pos], y_va)
    save_predictions(corpus, tag, rest, softmax(logits / T), logits)

    meta = {
        "model": key,
        "base": name,
        "revision": revision,
        "classes": list(CLASSES),
        "max_len": MAX_LEN,
        "temperature": round(float(T), 4),
        "normalization": "lahja.normalize v1",
    }
    info = {
        **meta,
        "seed": seed,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "precision": str(amp_dtype).replace("torch.", "") if amp_dtype else "float32",
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "n_train": len(ids_tr),
        "epochs": epochs,
        "best_epoch": best[1],
        "history": history,
        "train_seconds": round(train_seconds),
        "hyperparameters": {
            "batch": BATCH,
            "lr": LR,
            "warmup": WARMUP,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "loss": "cross-entropy with inverse-frequency class weights",
        },
    }
    d["results"].mkdir(parents=True, exist_ok=True)
    if main_run:
        (out / "lahja.json").write_text(json.dumps(meta, indent=2))
        (d["results"] / f"train_{key}.json").write_text(json.dumps(info, indent=2))
    else:
        (d["results"] / "seeds").mkdir(exist_ok=True)
        (d["results"] / "seeds" / f"train_{tag}.json").write_text(json.dumps(info, indent=2))
        import shutil

        shutil.rmtree(out)  # only the predictions of extra seeds are kept
    return info
