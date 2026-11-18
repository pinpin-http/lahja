# Lahja

[English version](README.md)

Lahja (لهجة, « dialecte ») lit un texte arabe court et dit de quelle région il vient : Golfe, Levant, Égypte, Maghreb ou arabe standard moderne. Le dépôt contient le pipeline de données, trois modèles (une référence en n-grammes de caractères et deux encodeurs BERT arabes affinés), leur évaluation, une API FastAPI et une démo web minimale en anglais et en français.

État au 18 novembre 2026 : le pipeline a tourné en entier, des téléchargements à la publication. Les deux encodeurs ont été affinés sur un GPU AMD, MARBERT a été retenu en validation et sa version quantifiée tient les trois seuils de publication (80,4 % de macro-F1 en test, contre 70,8 % pour la référence). Il reste à mettre la démo en ligne sur un Space Hugging Face.

## La tâche

En entrée, une phrase ou un message en écriture arabe, de 3 à 60 mots. En sortie, une classe parmi cinq, avec une probabilité pour chacune.

| Classe | Pays |
|---|---|
| Golfe | Arabie saoudite, Émirats arabes unis, Qatar, Koweït, Bahreïn, Oman |
| Levant | Syrie, Liban, Jordanie, Palestine |
| Égypte | Égypte |
| Maghreb | Maroc, Algérie, Tunisie, Libye |
| Arabe standard | arabe standard moderne (presse, textes officiels) |

L'Irak, le Yémen, le Soudan et la Mauritanie sont hors de cette première version, comme l'arabizi (l'arabe en lettres latines) et les textes qui mélangent plusieurs dialectes.

## Données

Le plan initial reposait sur MADAR (CAMeL Lab, NYU Abu Dhabi). Sa licence, vérifiée le 18 novembre 2026, n'autorise que « internal research and evaluation purposes », interdit la redistribution comme la modification, et le corpus s'obtient par un formulaire. Une démo publique ne peut donc pas tourner sur un modèle entraîné dessus. Le modèle public apprend sur trois corpus sous licence ouverte, et MADAR reste une piste de recherche facultative (`CORPUS=madar`, voir [Relancer](#relancer)).

| Corpus | Contenu | Licence | Lignes gardées |
|---|---|---|---:|
| [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200) (Meta AI) | phrases de Wikipédia traduites en arabe standard, najdi, levantin du nord et du sud, égyptien, marocain et tunisien | CC BY-SA 4.0 | 12 466 |
| [SADA](https://www.kaggle.com/datasets/sdaiancai/sada2022) (SDAIA, autorité saoudienne de l'audiovisuel) | transcriptions d'émissions de télévision saoudiennes, étiquetées avec le dialecte du locuteur | CC BY-NC-SA 4.0 | 8 389 |
| [Lahga](https://lahga.fyi) | dictionnaire collaboratif : phrases d'exemple dans chaque dialecte, définitions en arabe standard | CC BY-SA 4.0 | 3 235 |

Chaque fichier est téléchargé à une révision figée et comparé à son empreinte SHA-256 avant lecture (`src/lahja/config.py`). Rien de ces corpus n'est stocké dans le dépôt.

Chaque source a demandé un arbitrage.

- Dans FLORES, la traduction « najdi » recopie mot pour mot la phrase en arabe standard dans 1 444 cas sur 2 009. Ces copies sont retirées, si bien que FLORES n'apporte que quelques centaines de phrases du Golfe.
- Dans SADA, l'étiquette « ModernStandardArabic » décrit le locuteur, et une bonne partie de ces segments sont franchement dialectaux : elle n'est pas utilisée. SADA compte environ 130 000 segments du Golfe exploitables ; le pipeline en garde 6 000, tirés une émission après l'autre pour que quelques émissions très longues ne remplissent pas l'échantillon.
- D'après son mainteneur, plus de la moitié des formes de Lahga ont d'abord été rédigées par un modèle de langue, puis relues par une personne. Ses phrases d'exemple sont pourtant le seul texte familier dont je dispose pour le Levant et le Maghreb, alors je les garde, sans les formes marquées `needs_review`. La définition de chaque mot-vedette sert de phrase en arabe standard.

La normalisation (`src/lahja/normalize.py`) unifie les variantes de l'alif, retire les diacritiques et le tatwil, ramène le ta marbuta au ha et l'alif maqsura au ya, puis remplace par une espace tout ce qui n'est pas une lettre arabe. La ponctuation et les chiffres partent avec : les sous-titres n'en ont pas, Wikipédia en est plein, et un modèle s'en servirait pour reconnaître le corpus d'une phrase. L'API appelle la même fonction.

Le découpage suit la règle 80/10/10, par groupes et stratifié par classe. Un groupe, c'est une phrase source de FLORES (avec toutes ses traductions), une émission de SADA ou un mot-vedette de Lahga, et il ne se retrouve jamais dans deux partitions. Des tests vérifient que l'entraînement et le test ne partagent ni groupe ni phrase.

| | Golfe | Levant | Égypte | Maghreb | Arabe standard | total |
|---|---:|---:|---:|---:|---:|---:|
| entraînement | 5 498 | 3 991 | 3 043 | 3 542 | 2 849 | 18 923 |
| validation | 656 | 650 | 316 | 440 | 356 | 2 418 |
| test | 652 | 764 | 534 | 443 | 356 | 2 749 |
| tweets (hors domaine) | | 1 000 | 1 000 | 1 000 | | 3 000 |

Le jeu hors domaine est fait de tweets étiquetés par pays selon QADI (QCRI). La seule copie que j'ai trouvée avec le texte couvre l'Égypte, le Liban, la Libye, le Maroc et le Soudan : pas de classe Golfe, donc (le Soudan est écarté). Les tweets ne servent qu'aux scores ; leur texte n'est jamais écrit dans le dépôt ni affiché dans la démo. Quand MADAR sera téléchargé, il viendra s'ajouter comme second jeu hors domaine.

## Modèles

| Modèle | Rôle | Préentraînement |
|---|---|---|
| n-grammes de caractères (2 à 5) + TF-IDF + régression logistique | référence, rapide sur CPU, montre les suites de lettres propres à chaque dialecte | aucun |
| [CAMeLBERT-DA](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-da) (CAMeL Lab) | candidat principal | arabe dialectal, environ 5,8 milliards de mots |
| [MARBERT](https://huggingface.co/UBC-NLP/MARBERT) (UBC-NLP) | comparaison | tweets en arabe standard et dialectal, environ 15,6 milliards de mots |

Les deux encodeurs sont affinés de la même façon : modèle complet, 3 époques, AdamW à 3e-5, précision mixte, entropie croisée pondérée par l'inverse de la fréquence des classes. La graine 13 donne le modèle principal ; les graines 14 et 15 sont des exécutions supplémentaires qui mesurent la dispersion. La démo prend celui des deux qui fait le mieux en validation, sans que le test entre dans ce choix. Le modèle retenu est exporté en ONNX, fusionné, quantifié en entiers 8 bits et servi par ONNX Runtime sur CPU, avec des probabilités calibrées par mise à l'échelle de la température.

## Évaluation

La mesure principale est la macro-F1 sur le jeu de test. Le rapport donne aussi la précision et le rappel par classe, les matrices de confusion, l'erreur de calibration (ECE) avant et après mise à l'échelle, les scores hors domaine, la macro-F1 perdue à la quantification et la latence sur CPU, mesurée à travers l'API avec 2 threads, comme sur un Space Hugging Face gratuit. `make evaluate` écrit tout dans `results/` et `make report` en tire le [rapport d'évaluation](docs/evaluation.fr.md) et la [fiche du modèle](docs/model_card.fr.md), en anglais et en français.

`make publish` ne copie le modèle dans `models/release/` que si les trois seuils sont tenus :

| Seuil | Valeur |
|---|---|
| gain du modèle affiné sur la référence | au moins 5 points de macro-F1 en test |
| perte due à la quantification | au plus 1 point de macro-F1 |
| latence de la démo, 95e centile | moins de 300 ms par phrase sur CPU |

### Résultats

Jeu de test (2 749 phrases) et tweets hors domaine (3 000), macro-F1 et exactitude en % :

| Modèle | macro-F1 test | macro-F1 tweets | exactitude test |
|---|---:|---:|---:|
| n-grammes + régression logistique (C = 10) | 70,8 | 58,7 | 69,6 |
| CAMeLBERT-DA | 78,1 | 70,8 | 76,7 |
| MARBERT | 80,8 | 78,7 | 79,6 |
| MARBERT, ONNX int8 (servi par la démo) | 80,4 | 78,5 | 79,2 |

MARBERT l'emporte en validation (84,0 contre 81,3 pour CAMeLBERT-DA), c'est donc lui que sert la démo. Les trois seuils sont tenus : 10,0 points de gain sur la référence, 0,32 point perdu à la quantification et 24 ms au 95e centile à travers l'API, sur 2 threads d'un Ryzen 5 5600G. Sur trois graines, MARBERT fait 80,7 ± 0,3 en test et 76,3 ± 2,4 sur les tweets. La graine 13, celle du modèle publié, est la meilleure des trois sur les tweets : la moyenne donne une idée plus juste de ce qu'on peut attendre hors domaine.

Les erreurs se concentrent à un endroit. MARBERT atteint 84,7 de macro-F1 sur FLORES, mais 64,2 sur les dialogues télévisés de SADA, où il classe Golfe 45 % des phrases levantines et 29 % des égyptiennes. Les phrases courtes y sont pour quelque chose : 4 phrases levantines ou égyptiennes de SADA sur 10 font 5 mots ou moins. Mais au-delà de 20 mots, le quart part encore vers le Golfe, contre aucune dans FLORES. À l'entraînement, trois phrases de SADA sur quatre sont du Golfe, et le modèle a appris qu'un dialogue d'émission saoudienne vient le plus souvent de là. Le même biais touche les tweets maghrébins : MARBERT en envoie 285 sur 1 000 vers le Golfe, la référence 434. Dans FLORES, la confusion la plus fréquente est d'une autre nature : 23 des 56 phrases najdies du test sont prises pour de l'arabe standard, parce que ces traductions restent proches de la phrase source.

Le détail (scores par classe, matrices de confusion, calibration, latence) est dans le [rapport d'évaluation](docs/evaluation.fr.md).

## Démo et API

```bash
make serve   # http://127.0.0.1:8000, sert models/release/
```

La page a un champ de texte, un bouton d'exemple par dialecte, la classe prédite avec sa probabilité, une barre par classe et les mots qui ont le plus pesé sur la décision. Le poids d'un mot se mesure par occlusion : on retire chaque mot à tour de rôle, et la baisse de probabilité de la classe prédite donne son poids. L'interface suit la langue du navigateur et passe de l'anglais au français d'un clic.

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"text": "شو عم تعمل هلق؟", "explain": true}'
```

```json
{"label": "levant", "probabilities": {"gulf": 0.0343, "levant": 0.9108, "egypt": 0.0382, "maghreb": 0.0089, "msa": 0.0079},
 "n_words": 4, "in_scope": true,
 "words": [{"word": "شو", "weight": -0.014}, {"word": "عم", "weight": 0.0577}, {"word": "تعمل", "weight": 0.0056}, {"word": "هلق؟", "weight": 0.2743}]}
```

Un texte sans lettre arabe reçoit une erreur 422, et `in_scope` vaut `false` en dehors de 3 à 60 mots. Quand plusieurs mots désignent le même dialecte, en retirer un seul change peu la probabilité : l'influence d'un mot est alors faible même s'il est typique. Ce calcul demande un passage du modèle par mot : sur 2 threads, une requête avec `explain` prend 100 ms en médiane et 0,6 s au 95e centile, contre 12 et 24 ms sans.

`make space` prépare un dossier prêt à pousser sur un Space Hugging Face de type Docker (`space/Dockerfile`).

## Relancer

```bash
make install                   # uv, Python 3.12, dépendances figées, PyTorch pour GPU NVIDIA (CUDA 13.2)
make install TRAIN=train-rocm  # ou PyTorch pour GPU AMD (ROCm 7.2)
make all                       # données, référence, affinage, graines supplémentaires, export ONNX, évaluation, rapport, publication
```

L'affinage demande un GPU. Avec une carte NVIDIA, le pilote doit prendre en charge CUDA 13.2, pour lequel la roue PyTorch figée est compilée. Avec une carte AMD, le groupe `train-rocm` prend la roue ROCm 7.2 de PyTorch 2.14.1 (l'index ROCm ne propose pas la 2.15.0) ; PyTorch y présente la carte comme un périphérique `cuda`, sans autre changement. Les modèles publiés ont été affinés ainsi, sur une Radeon RX 9060 XT (16 Go) en bf16 : 2 min 20 s pour CAMeLBERT-DA, 3 min 30 s pour MARBERT, 18 min pour les six exécutions avec les graines supplémentaires. Tout le reste tourne sur le CPU d'un portable. Chaque étape se lance seule (`make data`, `make baseline`, `make finetune`, `make seeds`, `make export`, `make evaluate`, `make report`, `make publish`), et chaque étape de données écrit un fichier Parquet dans `data/interim/`. `results/open/split_fingerprint.txt` contient une empreinte du contenu du découpage : une machine qui affiche la même valeur a reconstruit le même découpage. La machine d'affinage a retrouvé la valeur obtenue sur le portable où le pipeline a été écrit ; la référence y a bougé d'un dixième de point (70,8 au lieu de 70,9 en test), un écart de calcul flottant d'un processeur à l'autre.

Pour la piste MADAR, il faut demander le corpus sur la [page du CAMeL Lab](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/), le décompresser dans `data/raw/madar/`, puis lancer `make all CORPUS=madar`. La licence autorise les résultats de recherche mais pas le modèle, et `make publish` s'arrête là. Une fois les fichiers en place, `make evaluate` sur le corpus ouvert note aussi le modèle public sur MADAR.

`make test` lance les tests : règles de normalisation, correspondance des étiquettes, absence de groupe ou de phrase commune entre entraînement et test, métriques, lecture de MADAR, réponses de l'API. GitHub Actions les fait tourner avec le linter à chaque push, sans les dépendances d'entraînement.

## Limites

- Golfe veut surtout dire Arabie saoudite. Presque toutes les phrases du Golfe à l'entraînement sont saoudiennes ; les parlers des Émirats, du Qatar, du Koweït, de Bahreïn et d'Oman sont à peine là. La version 2, avec une classe par pays du Golfe, demandera d'autres données.
- Chaque corpus a son registre, si bien que les modèles apprennent une part du registre au lieu du dialecte : beaucoup pour la référence, moins pour MARBERT, mais encore nettement sur les dialogues de SADA et les tweets maghrébins (voir [Résultats](#résultats)).
- Les tweets hors domaine n'ont pas de classe Golfe, et leur étiquette est le pays de l'auteur, pas une lecture du texte.
- Un texte irakien, yéménite ou soudanais reçoit quand même une des cinq étiquettes.
- En dessous de trois mots, la prédiction tient presque du hasard.

## Organisation du code

```
src/lahja/
  config.py      chemins, classes, sources figées et empreintes
  ingest.py      téléchargement, contrôle SHA-256, un lecteur par corpus
  normalize.py   normalisation commune à l'entraînement et à l'API
  labels.py      étiquettes des sources -> cinq classes
  data.py        ingest -> normalize -> label -> split, un Parquet par étape
  baseline.py    n-grammes + TF-IDF + régression logistique
  finetune.py    affinage (PyTorch, Transformers)
  export.py      choix du modèle en validation, export ONNX, quantification int8
  inference.py   inférence ONNX Runtime et influence des mots, sans PyTorch
  evaluate.py    métriques, calibration, latence, seuils
  report.py      rapport d'évaluation et fiche du modèle (EN/FR), figures
  publish.py     publication seulement si les seuils sont tenus
  api.py         FastAPI : POST /predict, page de démo
  web/           démo (HTML, CSS, JS)
tests/           pytest
space/           fichiers du Space Hugging Face
```

## Licence

Le code est sous licence MIT. Le modèle entraîné sera publié sous CC BY-NC-SA 4.0, parce que SADA est non commercial et à partage dans les mêmes conditions. Les corpus gardent leurs propres licences et sont téléchargés à la source, jamais redistribués ici.
