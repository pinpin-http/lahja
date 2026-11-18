# Lahja

[English version](README.md)

Lahja (لهجة, « dialecte ») lit un texte arabe court et dit de quelle région il vient : Golfe, Levant, Égypte, Maghreb ou arabe standard moderne. Le dépôt contient le pipeline de données, trois modèles (une référence en n-grammes de caractères et deux encodeurs BERT arabes affinés), leur évaluation, une API FastAPI et une démo web minimale en anglais et en français.

État au 18 novembre 2026 : le pipeline de données, la référence, l'export, l'évaluation, l'API, la démo et les tests sont écrits et passent une exécution de bout en bout. Les deux modèles affinés restent à entraîner sur une machine avec GPU. D'ici là, seule la référence a été mesurée.

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

### Mesuré à ce jour

La référence (C = 10, choisi en validation) atteint 70,9 % de macro-F1 en test (69,6 % d'exactitude) et 58,8 % sur les tweets. Ses erreurs penchent d'un côté : elle envoie beaucoup de phrases familières vers le Golfe, dont 431 des 1 000 tweets maghrébins. La plupart des phrases du Golfe à l'entraînement sont des dialogues télévisés, alors que celles du Maghreb et de l'arabe standard sont surtout des phrases de Wikipédia traduites ; pour ce modèle, le registre est un indice fort. Les modèles affinés diront quelle part de ce biais ils reprennent.

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
{"label": "levant", "probabilities": {"gulf": 0.02, "levant": 0.95, "egypt": 0.01, "maghreb": 0.01, "msa": 0.01},
 "n_words": 4, "in_scope": true, "words": [{"word": "شو", "weight": 0.31}, ...]}
```

(Format de la réponse ; les nombres sont donnés pour l'exemple.) Un texte sans lettre arabe reçoit une erreur 422, et `in_scope` vaut `false` en dehors de 3 à 60 mots.

`make space` prépare un dossier prêt à pousser sur un Space Hugging Face de type Docker (`space/Dockerfile`).

## Relancer

```bash
make install   # uv, Python 3.12, dépendances figées, PyTorch compris
make all       # données, référence, affinage, graines supplémentaires, export ONNX, évaluation, rapport, publication
```

L'affinage demande un GPU NVIDIA ; la roue PyTorch figée est compilée pour CUDA 13.2, le pilote doit donc la prendre en charge. Tout le reste tourne sur le CPU d'un portable. Chaque étape se lance seule (`make data`, `make baseline`, `make finetune`, `make seeds`, `make export`, `make evaluate`, `make report`, `make publish`), et chaque étape de données écrit un fichier Parquet dans `data/interim/`. `results/open/split_fingerprint.txt` contient une empreinte du contenu du découpage : une machine qui affiche la même valeur a reconstruit le même découpage.

Pour la piste MADAR, il faut demander le corpus sur la [page du CAMeL Lab](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/), le décompresser dans `data/raw/madar/`, puis lancer `make all CORPUS=madar`. La licence autorise les résultats de recherche mais pas le modèle, et `make publish` s'arrête là. Une fois les fichiers en place, `make evaluate` sur le corpus ouvert note aussi le modèle public sur MADAR.

`make test` lance les tests : règles de normalisation, correspondance des étiquettes, absence de groupe ou de phrase commune entre entraînement et test, métriques, lecture de MADAR, réponses de l'API. GitHub Actions les fait tourner avec le linter à chaque push, sans les dépendances d'entraînement.

## Limites

- Golfe veut surtout dire Arabie saoudite. Presque toutes les phrases du Golfe à l'entraînement sont saoudiennes ; les parlers des Émirats, du Qatar, du Koweït, de Bahreïn et d'Oman sont à peine là. La version 2, avec une classe par pays du Golfe, demandera d'autres données.
- Chaque corpus a son registre, si bien que les modèles peuvent apprendre le registre au lieu du dialecte, comme le fait déjà la référence.
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
