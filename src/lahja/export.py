"""Pick the better fine-tuned model on the validation set, export it to ONNX, quantise it to
8-bit integers. The test set plays no part in the choice."""

from __future__ import annotations

import json
import logging
import shutil

from sklearn.metrics import f1_score

from .config import ENCODERS, corpus_dirs
from .predictions import load_predictions

log = logging.getLogger(__name__)
OPSET = 17


def choose(corpus: str) -> tuple[str, dict]:
    scores = {}
    for key in ENCODERS:
        try:
            p = load_predictions(corpus, key)
        except FileNotFoundError:
            continue
        v = p[p.split == "val"]
        scores[key] = round(float(f1_score(v.label, v.pred, average="macro")), 4)
    if not scores:
        raise FileNotFoundError("no fine-tuned model found: run `make finetune` first")
    return max(scores, key=scores.get), scores


def run(corpus: str) -> dict:
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.transformers import optimizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    d = corpus_dirs(corpus)
    key, val_scores = choose(corpus)
    log.info("validation macro-F1 %s -> %s", val_scores, key)
    src = d["models"] / key
    out = d["models"] / "onnx"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    model.config.return_dict = False
    tok = AutoTokenizer.from_pretrained(src)
    sample = tok(["هذا مثال قصير", "جملة ثانية أطول قليلا من الأولى"], padding=True, return_tensors="pt")
    fp32 = out / "model.fp32.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
    # fuse attention / layer norm / GELU before quantising: faster, and fewer ops to quantise
    opt = optimizer.optimize_model(
        str(fp32),
        model_type="bert",
        num_heads=model.config.num_attention_heads,
        hidden_size=model.config.hidden_size,
    )
    fused = out / "model.fp32.opt.onnx"
    opt.save_model_to_file(str(fused))
    int8 = out / "model.int8.onnx"
    # fused ops live in the com.microsoft domain, where ONNX shape inference gives no types
    quantize_dynamic(
        str(fused),
        str(int8),
        weight_type=QuantType.QInt8,
        per_channel=False,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )

    tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    meta = json.loads((src / "lahja.json").read_text())
    meta.update({"onnx": "model.int8.onnx", "selected_on": "validation macro-F1", "val_macro_f1": val_scores})
    (out / "lahja.json").write_text(json.dumps(meta, indent=2))
    sizes = {p.name: round(p.stat().st_size / 2**20, 1) for p in out.glob("*.onnx")}
    info = {"selected": key, "val_macro_f1": val_scores, "onnx_mb": sizes, "opset": OPSET}
    (d["results"] / "export.json").write_text(json.dumps(info, indent=2))
    log.info("%s", info)
    return info
