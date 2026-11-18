# Lahja

[Version française](README.fr.md)

Lahja (لهجة, "dialect") reads a short Arabic text and says which region it comes from: the Gulf, the Levant, Egypt, the Maghreb, or Modern Standard Arabic (MSA). The repository holds the data pipeline, three models (a character n-gram baseline and two fine-tuned Arabic BERT encoders), their evaluation, a FastAPI service and a minimal web demo in English and French.

Status on 18 November 2026: the data pipeline, the baseline, the export, the evaluation, the API, the demo and the tests are written and pass an end-to-end run. The two fine-tuned models still have to be trained on a GPU machine. Until then, the only measured model is the baseline.

## Task

Input: one sentence or message in Arabic script, 3 to 60 words. Output: one of five classes, with a probability for each.

| Class | Countries |
|---|---|
| Gulf | Saudi Arabia, United Arab Emirates, Qatar, Kuwait, Bahrain, Oman |
| Levant | Syria, Lebanon, Jordan, Palestine |
| Egypt | Egypt |
| Maghreb | Morocco, Algeria, Tunisia, Libya |
| MSA | Modern Standard Arabic (press, official texts) |

Iraq, Yemen, Sudan and Mauritania are left out of this first version, as are Arabizi (Arabic in Latin letters) and texts that mix dialects.

## Data

The plan was to build on MADAR (CAMeL Lab, NYU Abu Dhabi). Its licence, checked on 18 November 2026, allows "internal research and evaluation purposes" only, forbids redistribution and modification, and the corpus comes through a request form. A public demo cannot run on a model trained on it. The public model is therefore trained on three corpora with open licences, and MADAR stays an optional research track (`CORPUS=madar`, see [Reproduce](#reproduce)).

| Corpus | Content | Licence | Rows kept |
|---|---|---|---:|
| [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200) (Meta AI) | Wikipedia sentences translated into MSA, Najdi, North and South Levantine, Egyptian, Moroccan and Tunisian | CC BY-SA 4.0 | 12,466 |
| [SADA](https://www.kaggle.com/datasets/sdaiancai/sada2022) (SDAIA, Saudi Broadcasting Authority) | transcripts of Saudi TV shows, labelled with the speaker's dialect | CC BY-NC-SA 4.0 | 8,389 |
| [Lahga](https://lahga.fyi) | a collaborative dictionary: example sentences in each dialect, MSA definitions | CC BY-SA 4.0 | 3,235 |

Every file is downloaded from a pinned revision and checked against its SHA-256 before parsing (`src/lahja/config.py`). Nothing from these corpora is stored in the repository.

Each source needed a decision:

- In FLORES, the "Najdi" translation is the MSA sentence copied word for word in 1,444 of 2,009 cases. Those copies are dropped, so FLORES gives only a few hundred Gulf sentences.
- In SADA, the "ModernStandardArabic" label describes the speaker, and many of those segments are plainly dialectal, so the label is not used. SADA has about 130,000 usable Gulf segments; the pipeline keeps 6,000, drawn one show at a time so that a few long shows do not fill the sample.
- Lahga's maintainer says more than half of its forms were first drafted by a language model and then read by a person. Its example sentences are the only conversational text for the Levant and the Maghreb, so they are kept, with forms flagged `needs_review` left out. Each headword's definition is used as an MSA sentence.

Normalisation (`src/lahja/normalize.py`) unifies alif variants, removes diacritics and tatweel, folds ta marbuta into ha and alif maqsura into ya, and turns everything that is not an Arabic letter into a space. Punctuation and digits go too: subtitles have none and Wikipedia has plenty, so they would tell a model which corpus a sentence came from. The API calls the same function.

The split is 80/10/10, grouped and stratified by class. A group is a FLORES source sentence (all its translations), a SADA show, or a Lahga headword, and a group never spans two splits. Tests check that train and test share no group and no sentence.

| | Gulf | Levant | Egypt | Maghreb | MSA | total |
|---|---:|---:|---:|---:|---:|---:|
| train | 5,498 | 3,991 | 3,043 | 3,542 | 2,849 | 18,923 |
| validation | 656 | 650 | 316 | 440 | 356 | 2,418 |
| test | 652 | 764 | 534 | 443 | 356 | 2,749 |
| tweets (out of domain) | | 1,000 | 1,000 | 1,000 | | 3,000 |

The out-of-domain set is made of tweets with QADI country labels (QCRI). The only copy I found with the text included covers Egypt, Lebanon, Libya, Morocco and Sudan, so it has no Gulf class (Sudan is dropped). Tweets are used for scores only; their text is never written to the repository or shown in the demo. When MADAR is downloaded, it is added as a second out-of-domain set.

## Models

| Model | Role | Pre-training |
|---|---|---|
| character n-grams (2 to 5) + TF-IDF + logistic regression | baseline, fast on CPU, shows which letter sequences mark each dialect | none |
| [CAMeLBERT-DA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da) (CAMeL Lab) | main candidate | dialectal Arabic, about 5.8 billion words |
| [MARBERT](https://huggingface.co/UBC-NLP/MARBERT) (UBC-NLP) | comparison | MSA and dialectal tweets, about 15.6 billion words |

Both encoders are fine-tuned the same way: full model, 3 epochs, AdamW at 3e-5, mixed precision, cross-entropy weighted by inverse class frequency. Seed 13 gives the main model; seeds 14 and 15 are extra runs that measure the spread. The demo uses whichever of the two scores higher on validation; the test set plays no part in that choice. The chosen model is exported to ONNX, fused, quantised to 8-bit integers, and served by ONNX Runtime on CPU, with probabilities calibrated by temperature scaling.

## Evaluation

The main metric is macro-F1 on the test set. The report also gives precision and recall per class, confusion matrices, calibration error (ECE) before and after temperature scaling, out-of-domain scores, the macro-F1 lost to quantisation, and CPU latency measured through the API with 2 threads, as on a free Hugging Face Space. `make evaluate` writes everything to `results/` and `make report` turns it into [docs/evaluation.md](docs/evaluation.md) and the [model card](docs/model_card.md), in English and French.

`make publish` copies the model to `models/release/` only if all three thresholds hold:

| Threshold | Value |
|---|---|
| gain of the fine-tuned model over the baseline | at least 5 points of test macro-F1 |
| loss from quantisation | at most 1 point of macro-F1 |
| demo latency, 95th percentile | under 300 ms per sentence on CPU |

### Measured so far

The baseline (C = 10, chosen on validation) reaches a test macro-F1 of 70.9 % (accuracy 69.6 %) and 58.8 % on the tweets. Its errors lean one way: it sends many casual sentences to Gulf, including 431 of the 1,000 Maghrebi tweets. Most Gulf training sentences are TV dialogue, while most Maghrebi and MSA ones are translated Wikipedia sentences, so register is a strong cue for this model. The fine-tuned models will show how much of that they inherit.

## Demo and API

```bash
make serve   # http://127.0.0.1:8000, serves models/release/
```

The page has a text field, one example button per dialect, the predicted class with its probability, a bar for each class, and the words that weighed most on the decision. Word weights come from occlusion: each word is removed in turn and the drop in the predicted class's probability is its weight. The interface follows the browser language and switches between English and French.

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"text": "شو عم تعمل هلق؟", "explain": true}'
```

```json
{"label": "levant", "probabilities": {"gulf": 0.02, "levant": 0.95, "egypt": 0.01, "maghreb": 0.01, "msa": 0.01},
 "n_words": 4, "in_scope": true, "words": [{"word": "شو", "weight": 0.31}, ...]}
```

(Example of the response format; the numbers are illustrative.) A text with no Arabic letters gets a 422. `in_scope` is false outside 3 to 60 words.

`make space` assembles a folder ready to push to a Hugging Face Docker Space (`space/Dockerfile`).

## Reproduce

```bash
make install   # uv, Python 3.12, pinned dependencies, PyTorch included
make all       # data, baseline, fine-tuning, extra seeds, ONNX export, evaluation, report, publication
```

Fine-tuning needs an NVIDIA GPU; the pinned PyTorch wheel is built for CUDA 13.2, so the driver must support it. Everything else runs on a laptop CPU. Each step can be run alone (`make data`, `make baseline`, `make finetune`, `make seeds`, `make export`, `make evaluate`, `make report`, `make publish`), and each data step writes a Parquet file in `data/interim/`. `results/open/split_fingerprint.txt` holds a hash of the split's content: a machine that prints the same value has rebuilt the same split.

For the MADAR track, request the corpus through the [CAMeL Lab page](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/), unzip it into `data/raw/madar/`, then run `make all CORPUS=madar`. The licence allows research results but not the model, so `make publish` stops there. With the files in place, `make evaluate` on the open corpus also scores the public model on MADAR.

`make test` runs the tests: normalisation rules, label mappings, no shared group or sentence between train and test, metrics, the MADAR parser, API responses. GitHub Actions runs them with the linter on every push, without the training dependencies.

## Limitations

- Gulf mostly means Saudi Arabia. Nearly every Gulf training sentence is Saudi, and Emirati, Qatari, Kuwaiti, Bahraini and Omani speech is barely there. The per-country version 2 will need other data.
- Each corpus has its own register, so the models can learn register instead of dialect, as the baseline already does.
- The out-of-domain tweets have no Gulf class, and their labels are the author's country rather than a reading of the text.
- Iraqi, Yemeni and Sudanese text still receives one of the five labels.
- Under three words, a prediction is close to a guess.

## Code layout

```
src/lahja/
  config.py      paths, classes, pinned sources and hashes
  ingest.py      download, SHA-256 check, one parser per corpus
  normalize.py   normalisation shared by training and the API
  labels.py      source labels -> five classes
  data.py        ingest -> normalize -> label -> split, one Parquet file per step
  baseline.py    n-grams + TF-IDF + logistic regression
  finetune.py    fine-tuning (PyTorch, Transformers)
  export.py      model choice on validation, ONNX export, int8 quantisation
  inference.py   ONNX Runtime inference and word influence, without PyTorch
  evaluate.py    metrics, calibration, latency, thresholds
  report.py      evaluation report and model card (EN/FR), figures
  publish.py     release only if the thresholds hold
  api.py         FastAPI: POST /predict, demo page
  web/           demo (HTML, CSS, JS)
tests/           pytest
space/           Hugging Face Space files
```

## Licence

The code is under MIT. The trained model will be released under CC BY-NC-SA 4.0, because SADA is non-commercial and share-alike. The corpora keep their own licences and are downloaded from their sources, never redistributed here.
