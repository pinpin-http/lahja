# Lahja

[Version française](README.fr.md) · **Demo: [lahja.soheib.fr](https://lahja.soheib.fr)** · Model: [Hugging Face](https://huggingface.co/pinpin-http/lahja)

Lahja (لهجة, "dialect") reads an Arabic sentence and says which region it comes from: the Gulf, the Levant, Egypt, the Maghreb, or Modern Standard Arabic. It is a MARBERT encoder fine-tuned on three openly licensed corpora and quantised to 8-bit integers. In the demo it runs in the browser with ONNX Runtime Web, so the sentence never leaves the device.

| Model | test macro-F1 | tweets macro-F1 (out of domain) |
|---|---:|---:|
| char n-grams + logistic regression | 70.8 | 58.7 |
| CAMeLBERT-DA | 78.1 | 70.8 |
| MARBERT | 80.8 | 78.7 |
| MARBERT int8, in the browser | 80.1 | 78.7 |

The test set holds 2,749 sentences (FLORES-200, SADA, Lahga) that share no source sentence, TV show or headword with training; the tweets come from a source never seen. The weak spot is the Saudi TV dialogue of SADA, where 45 % of the Levantine sentences are taken for Gulf.

More: [evaluation report](docs/evaluation.md), [model card](docs/model_card.md), [full description](docs/details.md) (data, choices, limits).

## Reproduce

```bash
make install TRAIN=train-rocm   # AMD GPU; `make install` for NVIDIA
make all                        # data, baseline, fine-tuning, ONNX export, evaluation, report, publication
make serve                      # local demo: http://127.0.0.1:8000
make test
```

Fine-tuning the six models takes 18 min on a Radeon RX 9060 XT; everything else runs on a CPU. `make hub` publishes the model on Hugging Face, `make site deploy-site` puts the page online on Cloudflare.

## Licence

Code under MIT, model under CC BY-NC-SA 4.0 (SADA is non-commercial). The corpora are downloaded from their source, never redistributed here.

---

A project by [Soheib](https://soheib.fr).
