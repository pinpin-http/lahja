"""CPU inference with ONNX Runtime: no PyTorch, no Transformers. Used by the API, the demo and
the evaluation, so the numbers in the report come from the code that serves the demo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .metrics import softmax
from .normalize import normalize


class NoArabicText(ValueError):
    pass


class DialectModel:
    def __init__(self, model_dir: str | Path, onnx_file: str | None = None, threads: int | None = None):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "lahja.json").read_text())
        self.classes: list[str] = self.config["classes"]
        self.temperature: float = float(self.config.get("temperature", 1.0))
        self.max_len: int = int(self.config.get("max_len", 128))
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(self.max_len)
        self.pad_id = self.tok.token_to_id("[PAD]") or 0

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or int(os.environ.get("LAHJA_THREADS", "2"))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        path = model_dir / (onnx_file or self.config.get("onnx", "model.int8.onnx"))
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def logits(self, normalized: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(normalized)
        L = max(len(e.ids) for e in enc)
        ids = np.full((len(enc), L), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(enc), L), dtype=np.int64)
        for i, e in enumerate(enc):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        return self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0]

    def proba(self, normalized: list[str]) -> np.ndarray:
        return softmax(self.logits(normalized) / self.temperature)

    def predict(self, text: str) -> dict:
        norm = normalize(text)
        if not norm:
            raise NoArabicText(text)
        p = self.proba([norm])[0]
        return self._result(norm, p)

    def _result(self, norm: str, p: np.ndarray) -> dict:
        return {
            "label": self.classes[int(p.argmax())],
            "probabilities": {c: round(float(x), 4) for c, x in zip(self.classes, p, strict=True)},
            "n_words": len(norm.split()),
        }

    def explain(self, text: str, max_words: int = 60) -> tuple[dict, list[dict]]:
        """Prediction plus, for each word of the input, how much the predicted class loses when
        that word is removed (occlusion). One batched forward pass for all the variants."""
        norm = normalize(text)
        if not norm:
            raise NoArabicText(text)
        raw = text.split()
        # map every normalised word back to the raw token it came from
        owner = [i for i, tok in enumerate(raw) for _ in normalize(tok).split()]
        words = norm.split()
        if len(owner) != len(words):  # normalisation crossed a token boundary: show normalised words
            raw, owner = words, list(range(len(words)))
        n = min(len(words), max_words)
        variants = [norm] + [" ".join(words[:j] + words[j + 1 :]) or "" for j in range(n)]
        probs = self.proba([v if v else norm for v in variants])
        base, k = probs[0], int(probs[0].argmax())
        drop = base[k] - probs[1:, k]
        if len(words) == 1:
            drop[:] = base[k]
        weights = np.zeros(len(raw))
        for j in range(n):
            weights[owner[j]] += drop[j]
        tokens = [{"word": w, "weight": round(float(x), 4)} for w, x in zip(raw, weights, strict=True)]
        return self._result(norm, base), tokens
