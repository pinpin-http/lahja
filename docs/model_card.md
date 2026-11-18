# Model card: Lahja

[Version française](model_card.fr.md)

Lahja reads a short Arabic text (a sentence, a message) and says which of five groups it belongs to: Gulf, Levant, Egypt, Maghreb, or Modern Standard Arabic (MSA). This card covers the model behind the demo.

## Model

- **Architecture**: MARBERT, a BERT-base encoder pre-trained by its authors (`UBC-NLP/MARBERT`, revision `88e1fa1`), with a classification head fine-tuned here.
- **Served as**: ONNX, weights quantised to 8-bit integers, run by ONNX Runtime on CPU.
- **Input**: Arabic script, normalised by `lahja.normalize` (alif variants unified, diacritics and tatweel removed, ta marbuta and alif maqsura folded, everything but Arabic letters dropped). It works best between 3 and 60 words.
- **Output**: a probability for each class, calibrated by temperature scaling on the validation set.
- **Licence**: CC BY-NC-SA 4.0 for the model, because one training corpus (SADA) is non-commercial and share-alike. The code is under MIT.

## Intended use

A demo and a portfolio project: guess the broad region a written sentence comes from. Do not use it to make decisions about people, such as origin, nationality, asylum or hiring. A dialect says little about where someone comes from, and nothing about who they are.

## Training data

18,923 training sentences from three corpora with open licences, split 80/10/10 so that no source sentence, TV show or dictionary headword appears on both sides.

|  | Gulf | Levant | Egypt | Maghreb | MSA | total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 5,498 | 3,991 | 3,043 | 3,542 | 2,849 | 18,923 |
| validation | 656 | 650 | 316 | 440 | 356 | 2,418 |
| test | 652 | 764 | 534 | 443 | 356 | 2,749 |
| tweets (out-of-domain) | – | 1,000 | 1,000 | 1,000 | – | 3,000 |

- **FLORES-200** (Meta AI): Wikipedia sentences translated by professionals into MSA and into dialects. Its "Najdi" version copies the MSA sentence word for word in most cases; those copies are removed, which leaves only a few hundred Gulf sentences from this corpus.
- **SADA** (SDAIA and the Saudi Broadcasting Authority): transcripts of Saudi TV shows, labelled with the speaker's dialect. It supplies most of the Gulf class, and some Egyptian and Levantine speech. Its "Modern Standard Arabic" label describes the speaker rather than the sentence, so it is not used.
- **Lahga** (lahga.fyi): a collaborative dictionary. Its example sentences give everyday speech in each dialect, its definitions give short MSA sentences. According to its maintainer, more than half of the forms were first drafted by a language model, then read by a person.

Gulf sentences by country:

|  | n |
| :--- | ---: |
| Saudi Arabia | 6,679 |
| UAE | 1 |
| Qatar | 0 |
| Kuwait | 0 |
| Bahrain | 0 |
| Oman | 0 |
| unspecified (Lahga "gulf") | 126 |

## Results

Test set, int8 model as served: macro-F1 80.4 %, accuracy 79.2 %, calibration error (ECE) 0.074.

| class | precision | recall | F1 | n |
| :--- | ---: | ---: | ---: | ---: |
| Gulf | 67.5 | 80.5 | 73.4 | 652 |
| Levant | 88.0 | 67.4 | 76.3 | 764 |
| Egypt | 78.5 | 71.7 | 75.0 | 534 |
| Maghreb | 88.5 | 91.9 | 90.1 | 443 |
| MSA | 79.0 | 97.2 | 87.2 | 356 |

Out of domain:

- tweets (out-of-domain): macro-F1 78.5

The full comparison with the other two models, the confusion matrices and the calibration curve are in the [evaluation report](evaluation.md).

## Known limitations

- **Gulf mostly means Saudi Arabia.** Almost all Gulf training sentences are Saudi (table above). Emirati, Qatari, Kuwaiti, Bahraini and Omani speech is barely present, and the per-country version 2 will need other data.
- **Iraq, Yemen, Sudan and Mauritania are out of scope.** Their dialects still get one of the five labels, and no test set says which.
- **Register can stand in for dialect.** Each corpus has its own register: Gulf comes mostly from TV dialogue, Maghreb mostly from translated Wikipedia sentences. The model learns part of the register instead of the dialect: in the SADA TV transcripts, many Levantine and Egyptian sentences are labelled Gulf, and so are many Maghrebi tweets (see the scores by corpus and the out-of-domain tests in the evaluation report).
- **Short texts.** Under three words the answer is close to a guess, and the demo says so.
- **Not supported**: mixed dialects, Arabizi (Arabic in Latin letters), code-switching. Latin letters are removed before the model sees the text.
- **Formal Gulf and MSA** are hard to tell apart when the sentence has few dialect words.

## Latency

24.0 ms at the 95th percentile per sentence through the API, on 2 CPU threads.

## Sources

FLORES-200 (Meta AI, CC BY-SA 4.0), SADA (SDAIA and SBA, CC BY-NC-SA 4.0), Lahga (lahga.fyi, CC BY-SA 4.0), CAMeLBERT-DA (CAMeL Lab, Apache 2.0), MARBERT (UBC-NLP). Tweets serve for evaluation only (QADI labels, QCRI); no tweet text is published.
