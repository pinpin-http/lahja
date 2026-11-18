# Lahja: the full description

[Version française](details.fr.md) · [Back to the README](../README.md)

Lahja (لهجة, "dialect") reads a short Arabic text and says which region it comes from: the Gulf, the Levant, Egypt, the Maghreb, or Modern Standard Arabic (MSA). The repository holds the data pipeline, three models (a character n-gram baseline and two fine-tuned Arabic BERT encoders), their evaluation, a FastAPI service and a minimal web demo in English and French.

Status on 18 November 2026: the whole pipeline has run, from the downloads to the publication. Both encoders were fine-tuned on an AMD GPU, MARBERT was picked on validation, and its quantised version meets the three publication thresholds (test macro-F1 of 80.4 %, against 70.8 % for the baseline). The demo is online at [lahja.soheib.fr](https://lahja.soheib.fr) and the model on [Hugging Face](https://huggingface.co/pinpin-http/lahja).

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

The main metric is macro-F1 on the test set. The report also gives precision and recall per class, confusion matrices, calibration error (ECE) before and after temperature scaling, out-of-domain scores, the macro-F1 lost to quantisation, and CPU latency measured through the API with 2 threads, as on a free Hugging Face Space. `make evaluate` writes everything to `results/` and `make report` turns it into [docs/evaluation.md](evaluation.md) and the [model card](model_card.md), in English and French.

`make publish` copies the model to `models/release/` only if all three thresholds hold:

| Threshold | Value |
|---|---|
| gain of the fine-tuned model over the baseline | at least 5 points of test macro-F1 |
| loss from quantisation | at most 1 point of macro-F1 |
| demo latency, 95th percentile | under 300 ms per sentence on CPU |

### Results

Test set (2,749 sentences) and out-of-domain tweets (3,000), macro-F1 and accuracy in %:

| Model | test macro-F1 | tweets macro-F1 | test accuracy |
|---|---:|---:|---:|
| char n-grams + logistic regression (C = 10) | 70.8 | 58.7 | 69.6 |
| CAMeLBERT-DA | 78.1 | 70.8 | 76.7 |
| MARBERT | 80.8 | 78.7 | 79.6 |
| MARBERT, ONNX int8 (served by the demo) | 80.4 | 78.5 | 79.2 |

MARBERT wins on validation (84.0 against 81.3 for CAMeLBERT-DA), so it is the model the demo serves. All three thresholds hold: a 10.0-point gain over the baseline, 0.32 point lost to quantisation, and 24 ms at the 95th percentile through the API on 2 threads of a Ryzen 5 5600G. Over three seeds, MARBERT scores 80.7 ± 0.3 on the test set and 76.3 ± 2.4 on the tweets. Seed 13, the published model, is the best of the three on the tweets, so the mean is the fairer guide to what to expect out of domain.

The errors sit in one place. MARBERT reaches a macro-F1 of 84.7 on FLORES but 64.2 on the SADA TV dialogue, where it labels 45 % of the Levantine sentences and 29 % of the Egyptian ones as Gulf. Short sentences play a part: 4 in 10 Levantine or Egyptian SADA sentences have 5 words or fewer. But above 20 words, a quarter still go to Gulf, against none in FLORES. Three in four SADA training sentences are Gulf, and the model has learned that dialogue from a Saudi TV show is usually Gulf. The same bias reaches the Maghrebi tweets: MARBERT sends 285 of 1,000 to Gulf, the baseline 434. In FLORES the most frequent confusion is of another kind: 23 of the 56 Najdi test sentences are taken for MSA, because those translations stay close to the source sentence.

The details (per-class scores, confusion matrices, calibration, latency) are in the [evaluation report](evaluation.md).

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
{"label": "levant", "probabilities": {"gulf": 0.0343, "levant": 0.9108, "egypt": 0.0382, "maghreb": 0.0089, "msa": 0.0079},
 "n_words": 4, "in_scope": true,
 "words": [{"word": "شو", "weight": -0.014}, {"word": "عم", "weight": 0.0577}, {"word": "تعمل", "weight": 0.0056}, {"word": "هلق؟", "weight": 0.2743}]}
```

A text with no Arabic letters gets a 422. `in_scope` is false outside 3 to 60 words. When several words point to the same dialect, removing one of them barely moves the probability, so a typical word can still get a small weight. The computation takes one model pass per word: on 2 threads, a request with `explain` takes 100 ms at the median and 0.6 s at the 95th percentile, against 12 and 24 ms without.

The public page, [lahja.soheib.fr](https://lahja.soheib.fr), is the same web app without a server: the browser downloads the model from the [Hugging Face repo](https://huggingface.co/pinpin-http/lahja) (156 MB, pinned to a revision, then kept in its cache) and runs it with ONNX Runtime Web, so the text never leaves the device. `src/lahja/web/engine.js` ports the normalisation, the tokeniser and the word influence; on the 27,098 texts of the corpus it gives the same tokens as Python, and the model scores a test macro-F1 of 80.1 % and 78.7 % on the tweets in the browser (80.4 % and 78.5 % with the native ONNX Runtime). `make hub` publishes the model repo, `make site deploy-site` builds and deploys the page on Cloudflare. `space/Dockerfile` builds an image of the API, which fetches the model from the Hub; a Hugging Face Docker Space now needs a PRO account (`make deploy-space`).

## Reproduce

```bash
make install                   # uv, Python 3.12, pinned dependencies, PyTorch for an NVIDIA GPU (CUDA 13.2)
make install TRAIN=train-rocm  # or PyTorch for an AMD GPU (ROCm 7.2)
make all                       # data, baseline, fine-tuning, extra seeds, ONNX export, evaluation, report, publication
```

Fine-tuning needs a GPU. With an NVIDIA card, the driver must support CUDA 13.2, which the pinned PyTorch wheel is built for. With an AMD card, the `train-rocm` group takes the ROCm 7.2 wheel of PyTorch 2.14.1 (the ROCm index has no 2.15.0); PyTorch exposes the card as a `cuda` device, and nothing else changes. The published models were fine-tuned this way, on a Radeon RX 9060 XT (16 GB) in bf16: 2 min 20 s for CAMeLBERT-DA, 3 min 30 s for MARBERT, 18 min for all six runs including the extra seeds. Everything else runs on a laptop CPU. Each step can be run alone (`make data`, `make baseline`, `make finetune`, `make seeds`, `make export`, `make evaluate`, `make report`, `make publish`), and each data step writes a Parquet file in `data/interim/`. `results/open/split_fingerprint.txt` holds a hash of the split's content: a machine that prints the same value has rebuilt the same split. The fine-tuning machine got the same value as the laptop the pipeline was written on; the baseline moved there by a tenth of a point (70.8 instead of 70.9 on the test set), a floating-point difference from one CPU to another.

For the MADAR track, request the corpus through the [CAMeL Lab page](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/), unzip it into `data/raw/madar/`, then run `make all CORPUS=madar`. The licence allows research results but not the model, so `make publish` stops there. With the files in place, `make evaluate` on the open corpus also scores the public model on MADAR.

`make test` runs the tests: normalisation rules, label mappings, no shared group or sentence between train and test, metrics, the MADAR parser, API responses. GitHub Actions runs them with the linter on every push, without the training dependencies.

## Limitations

- Gulf mostly means Saudi Arabia. Nearly every Gulf training sentence is Saudi, and Emirati, Qatari, Kuwaiti, Bahraini and Omani speech is barely there. The per-country version 2 will need other data.
- Each corpus has its own register, so the models learn part of the register instead of the dialect: the baseline a lot, MARBERT less, but still visibly on the SADA dialogue and the Maghrebi tweets (see [Results](#results)).
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
  web/           demo (HTML, CSS, JS); engine.js runs the model in the browser
tests/           pytest
space/           Hugging Face model repo builder, API Docker image
site/            Cloudflare config of the public page
```

## Licence

The code is under MIT. The trained model will be released under CC BY-NC-SA 4.0, because SADA is non-commercial and share-alike. The corpora keep their own licences and are downloaded from their sources, never redistributed here.
