"""Text of the generated documents. English first, French second; same keys in both."""

# ruff: noqa: E501

TEXT = {
    "en": {
        "lang_link": "[Version française](evaluation.fr.md)",
        "eval_title": "Lahja: evaluation report",
        "eval_intro": (
            "Corpus `{corpus}`. The model kept for the demo is **{sel}**, chosen on the validation set. "
            "On the test set it reaches a macro-F1 of {f1} %, against {base} % for the character n-gram baseline, "
            "a gain of {gain} points. Every number below comes from `make evaluate` and is rewritten on each run."
        ),
        "gate": "Acceptance thresholds",
        "gate_intro": "The model is exported and published only if all three thresholds hold.",
        "criterion": "criterion",
        "threshold": "threshold",
        "measured": "measured",
        "passed": "met",
        "gain_row": "gain of the fine-tuned model over the baseline (test macro-F1)",
        "gain_thr": "≥ 5 points",
        "quant_row": "macro-F1 lost to int8 quantisation",
        "quant_thr": "≤ 1 point",
        "lat_row": "demo latency, 95th percentile, CPU",
        "lat_thr": "< 300 ms per sentence",
        "yes": "yes",
        "no": "**no**",
        "gate_pass": "All three thresholds are met: `make publish` copies the quantised model to `models/release/`.",
        "gate_fail": "At least one threshold is missed: `make publish` refuses to publish the model.",
        "comparison": "The three models",
        "comparison_intro": (
            "Same training data, same split, same normalisation. The test set shares no source sentence, no TV show "
            "and no dictionary headword with the training set. The two ONNX rows are the selected model after export, "
            "in 32-bit floats and in 8-bit integers (the demo runs the latter)."
        ),
        "model": "model",
        "acc_test": "accuracy (test)",
        "star": "★ selected for the demo. Validation macro-F1: {scores}.",
        "seeds_intro": "Each fine-tuned model was also trained with other seeds, everything else unchanged. Mean ± standard deviation over the runs:",
        "n_seeds": "seeds",
        "per_class": "Precision, recall and F1 per class (test)",
        "class": "class",
        "cm_alt": "Confusion matrices",
        "cm_note": "Each row adds up to 100: the share of a true class sent to each predicted class.",
        "by_source": "Test score by corpus of origin",
        "by_source_dropped": "{cls} left out of {src}, {n} sentences",
        "by_source_none": "no class left out",
        "by_source_intro": "The test set mixes three corpora with very different registers. Macro-F1 over the classes that have at least {min_rows} test sentences in each corpus ({dropped}):",
        "by_source_note": (
            "FLORES-200 holds formal sentences translated from English Wikipedia, SADA holds TV dialogue, and Lahga "
            "holds short example sentences from a dictionary."
        ),
        "ood": "Out-of-domain tests",
        "ood_intro": (
            "These sets come from sources the models never saw in training. Predicting a class absent from a set "
            "(MSA, for instance) counts as an error, and macro-F1 averages over the classes present."
        ),
        "ood_tweets": (
            "Tweets with QADI country labels (QCRI), 1,000 per class, drawn at random among those of 3 to 60 words "
            "after normalisation. The corpus available with its text only covers Egypt, Lebanon, Libya, Morocco and "
            "Sudan (left out), so there is no Gulf class here. A tweet's label is its author's country, not a reading "
            "of the text: some of these tweets are in MSA. No tweet text is published, only the scores."
        ),
        "ood_madar": "MADAR (CAMeL Lab), every city mapped to its class, all splits pooled.",
        "madar_missing": (
            "MADAR is not part of this run: its licence requires a manual download (see the README). "
            "Once the files are in `data/raw/madar/`, `make evaluate` adds it here."
        ),
        "calibration": "Calibration",
        "calibration_intro": (
            "Expected calibration error (ECE, 15 bins) of the selected model on the test set: {before} before "
            "temperature scaling, {after} after. The temperature is fitted on the validation set only "
            "(T = {t} for the PyTorch model, {t8} for the int8 model the demo runs). Scaling changes confidence, "
            "never the predicted class."
        ),
        "rel_alt": "Reliability diagram",
        "quant": "Quantisation",
        "quant_intro": (
            "Dynamic int8 quantisation of the weights with ONNX Runtime, after fusing attention and layer-norm ops. "
            "Test macro-F1: {f32} % in fp32, {f8} % in int8, a loss of {loss} point. The file goes from "
            "{mb32} MB to {mb8} MB."
        ),
        "latency": "CPU latency",
        "latency_intro": (
            "{n} test sentences sent one at a time, ONNX Runtime limited to {threads} threads, which is what a "
            "free Hugging Face CPU Space provides (processor: {cpu}). The API path covers the whole request: validation, "
            "normalisation, tokenisation, inference and JSON."
        ),
        "path": "path",
        "api_predict_ms": "API `POST /predict`",
        "model_predict_ms": "model alone (int8)",
        "api_predict_explain_ms": "API `POST /predict` with `explain` (word influence)",
        "baseline_predict_ms": "baseline (scikit-learn)",
        "latency_note": (
            "Word influence removes each word in turn and measures what the predicted class loses, which costs one "
            "extra forward pass per word, batched together. The threshold applies to the plain prediction."
        ),
        "ngrams": "What the baseline looks at",
        "ngrams_intro": "Character n-grams with the largest logistic-regression weight for each class, word edges removed and duplicates merged (`char_wb` n-grams carry the spaces around a word).",
        "ngrams_col": "n-grams",
        "training": "Training",
        "device": "device",
        "duration": "duration",
        "best_epoch": "best epoch",
        "val_history": "validation macro-F1 by epoch",
        "training_note": (
            "Fine-tuning: full model, 3 epochs, batches of 32, AdamW with a learning rate of 3e-5, linear warm-up "
            "over 6 % of the steps then linear decay, weight decay 0.01, mixed precision, cross-entropy weighted by "
            "inverse class frequency, seed 13. The epoch kept is the best on validation. "
            "Baseline: TF-IDF on character n-grams of 2 to 5 (`char_wb`), logistic regression with balanced class "
            "weights, C = {C} chosen on validation."
        ),
        "precision": "precision",
        "recall": "recall",
        "card": """# Model card: Lahja

[Version française](model_card.fr.md)

Lahja reads a short Arabic text (a sentence, a message) and says which of five groups it belongs to: Gulf, Levant, Egypt, Maghreb, or Modern Standard Arabic (MSA). This card covers the model behind the demo.

## Model

- **Architecture**: {sel}, a BERT-base encoder pre-trained by its authors (`{base}`, revision `{revision}`), with a classification head fine-tuned here.
- **Served as**: ONNX, weights quantised to 8-bit integers, run by ONNX Runtime on CPU.
- **Input**: Arabic script, normalised by `lahja.normalize` (alif variants unified, diacritics and tatweel removed, ta marbuta and alif maqsura folded, everything but Arabic letters dropped). It works best between 3 and 60 words.
- **Output**: a probability for each class, calibrated by temperature scaling on the validation set.
- **Licence**: CC BY-NC-SA 4.0 for the model, because one training corpus (SADA) is non-commercial and share-alike. The code is under MIT.

## Intended use

A demo and a portfolio project: guess the broad region a written sentence comes from. Do not use it to make decisions about people, such as origin, nationality, asylum or hiring. A dialect says little about where someone comes from, and nothing about who they are.

## Training data

{n_train} training sentences from three corpora with open licences, split 80/10/10 so that no source sentence, TV show or dictionary headword appears on both sides.

{data_table}

- **FLORES-200** (Meta AI): Wikipedia sentences translated by professionals into MSA and into dialects. Its "Najdi" version copies the MSA sentence word for word in most cases; those copies are removed, which leaves only a few hundred Gulf sentences from this corpus.
- **SADA** (SDAIA and the Saudi Broadcasting Authority): transcripts of Saudi TV shows, labelled with the speaker's dialect. It supplies most of the Gulf class, and some Egyptian and Levantine speech. Its "Modern Standard Arabic" label describes the speaker rather than the sentence, so it is not used.
- **Lahga** (lahga.fyi): a collaborative dictionary. Its example sentences give everyday speech in each dialect, its definitions give short MSA sentences. According to its maintainer, more than half of the forms were first drafted by a language model, then read by a person.

Gulf sentences by country:

{countries}

## Results

Test set, int8 model as served: macro-F1 {f1} %, accuracy {acc} %, calibration error (ECE) {ece}.

{per_class}

Out of domain:

{ood_lines}

The full comparison with the other two models, the confusion matrices and the calibration curve are in the [evaluation report](evaluation.md).

## Known limitations

- **Gulf mostly means Saudi Arabia.** Almost all Gulf training sentences are Saudi (table above). Emirati, Qatari, Kuwaiti, Bahraini and Omani speech is barely present, and the per-country version 2 will need other data.
- **Iraq, Yemen, Sudan and Mauritania are out of scope.** Their dialects still get one of the five labels, and no test set says which.
- **Register can stand in for dialect.** Each corpus has its own register: Gulf comes mostly from TV dialogue, Maghreb mostly from translated Wikipedia sentences. The model learns part of the register instead of the dialect: in the SADA TV transcripts, many Levantine and Egyptian sentences are labelled Gulf, and so are many Maghrebi tweets (see the scores by corpus and the out-of-domain tests in the evaluation report).
- **Short texts.** Under three words the answer is close to a guess, and the demo says so.
- **Not supported**: mixed dialects, Arabizi (Arabic in Latin letters), code-switching. Latin letters are removed before the model sees the text.
- **Formal Gulf and MSA** are hard to tell apart when the sentence has few dialect words.

## Latency

{p95} ms at the 95th percentile per sentence through the API, on 2 CPU threads.

## Sources

FLORES-200 (Meta AI, CC BY-SA 4.0), SADA (SDAIA and SBA, CC BY-NC-SA 4.0), Lahga (lahga.fyi, CC BY-SA 4.0), CAMeLBERT-DA (CAMeL Lab, Apache 2.0), MARBERT (UBC-NLP). Tweets serve for evaluation only (QADI labels, QCRI); no tweet text is published.""",
    },
    "fr": {
        "lang_link": "[English version](evaluation.md)",
        "eval_title": "Lahja : rapport d'évaluation",
        "eval_intro": (
            "Corpus `{corpus}`. Le modèle retenu pour la démo est **{sel}**, choisi sur le jeu de validation. "
            "Sur le jeu de test, il atteint une macro-F1 de {f1} %, contre {base} % pour la référence en n-grammes "
            "de caractères, soit {gain} points de plus. Tous les chiffres ci-dessous sortent de `make evaluate` et "
            "sont réécrits à chaque exécution."
        ),
        "gate": "Seuils d'acceptation",
        "gate_intro": "Le modèle n'est exporté et publié que si les trois seuils sont tenus.",
        "criterion": "critère",
        "threshold": "seuil",
        "measured": "mesuré",
        "passed": "tenu",
        "gain_row": "gain du modèle affiné sur la référence (macro-F1 en test)",
        "gain_thr": "≥ 5 points",
        "quant_row": "macro-F1 perdue à la quantification int8",
        "quant_thr": "≤ 1 point",
        "lat_row": "latence de la démo, 95e centile, CPU",
        "lat_thr": "< 300 ms par phrase",
        "yes": "oui",
        "no": "**non**",
        "gate_pass": "Les trois seuils sont tenus : `make publish` copie le modèle quantifié dans `models/release/`.",
        "gate_fail": "Au moins un seuil n'est pas tenu : `make publish` refuse de publier le modèle.",
        "comparison": "Les trois modèles",
        "comparison_intro": (
            "Mêmes données d'entraînement, même découpage, même normalisation. Le jeu de test ne partage aucune "
            "phrase source, aucune émission et aucun mot-vedette avec l'entraînement. Les deux lignes ONNX "
            "correspondent au modèle retenu après export, en flottants 32 bits puis en entiers 8 bits (c'est cette "
            "dernière version que la démo fait tourner)."
        ),
        "model": "modèle",
        "acc_test": "exactitude (test)",
        "star": "★ retenu pour la démo. Macro-F1 en validation : {scores}.",
        "seeds_intro": "Chaque modèle affiné a aussi été entraîné avec d'autres graines, sans rien changer d'autre. Moyenne ± écart-type sur ces exécutions :",
        "n_seeds": "graines",
        "per_class": "Précision, rappel et F1 par classe (test)",
        "class": "classe",
        "cm_alt": "Matrices de confusion",
        "cm_note": "Chaque ligne fait 100 : la part d'une classe réelle envoyée vers chaque classe prédite.",
        "by_source": "Score en test selon le corpus d'origine",
        "by_source_dropped": "{cls} écarté de {src}, {n} phrases",
        "by_source_none": "aucune classe écartée",
        "by_source_intro": "Le jeu de test mêle trois corpus de registres très différents. Macro-F1 sur les classes qui comptent au moins {min_rows} phrases de test dans chaque corpus ({dropped}) :",
        "by_source_note": (
            "FLORES-200 contient des phrases soutenues traduites de Wikipédia en anglais, SADA des dialogues "
            "d'émissions de télévision, Lahga de courtes phrases d'exemple tirées d'un dictionnaire."
        ),
        "ood": "Tests hors domaine",
        "ood_intro": (
            "Ces jeux viennent de sources que les modèles n'ont jamais vues à l'entraînement. Prédire une classe "
            "absente du jeu (l'arabe standard, par exemple) compte comme une erreur, et la macro-F1 fait la moyenne "
            "sur les classes présentes."
        ),
        "ood_tweets": (
            "Tweets étiquetés par pays selon QADI (QCRI), 1 000 par classe, tirés au hasard parmi ceux qui font de "
            "3 à 60 mots après normalisation. Le corpus disponible avec son texte ne couvre que l'Égypte, le Liban, "
            "la Libye, le Maroc et le Soudan (écarté) : il n'y a donc pas de classe Golfe ici. L'étiquette d'un tweet "
            "est le pays de son auteur, pas une lecture du texte, et certains sont en arabe standard. Aucun texte de "
            "tweet n'est publié, seulement les scores."
        ),
        "ood_madar": "MADAR (CAMeL Lab), chaque ville rattachée à sa classe, toutes partitions confondues.",
        "madar_missing": (
            "MADAR ne fait pas partie de cette exécution : sa licence impose un téléchargement manuel (voir le "
            "README). Une fois les fichiers dans `data/raw/madar/`, `make evaluate` l'ajoute ici."
        ),
        "calibration": "Calibration",
        "calibration_intro": (
            "Erreur de calibration attendue (ECE, 15 intervalles) du modèle retenu sur le jeu de test : {before} "
            "avant mise à l'échelle de la température, {after} après. La température est ajustée sur la validation "
            "seule (T = {t} pour le modèle PyTorch, {t8} pour le modèle int8 de la démo). Elle change la confiance, "
            "jamais la classe prédite."
        ),
        "rel_alt": "Diagramme de fiabilité",
        "quant": "Quantification",
        "quant_intro": (
            "Quantification dynamique des poids en int8 avec ONNX Runtime, après fusion des opérations d'attention "
            "et de normalisation. Macro-F1 en test : {f32} % en fp32, {f8} % en int8, soit {loss} point de perte. "
            "Le fichier passe de {mb32} Mo à {mb8} Mo."
        ),
        "latency": "Latence sur CPU",
        "latency_intro": (
            "{n} phrases du jeu de test envoyées une par une, ONNX Runtime limité à {threads} threads, ce que fournit "
            "un Space Hugging Face gratuit sur CPU (processeur : {cpu}). Le chemin API couvre toute la requête : validation, "
            "normalisation, découpage en tokens, inférence et JSON."
        ),
        "path": "chemin",
        "api_predict_ms": "API `POST /predict`",
        "model_predict_ms": "modèle seul (int8)",
        "api_predict_explain_ms": "API `POST /predict` avec `explain` (influence des mots)",
        "baseline_predict_ms": "référence (scikit-learn)",
        "latency_note": (
            "L'influence des mots retire chaque mot à tour de rôle et mesure ce que perd la classe prédite, soit un "
            "passage de plus par mot, regroupés en un seul lot. Le seuil porte sur la prédiction seule."
        ),
        "ngrams": "Ce que regarde la référence",
        "ngrams_intro": "N-grammes de caractères au plus fort poids dans la régression logistique, pour chaque classe, bords de mot retirés et doublons fusionnés (les n-grammes `char_wb` portent les espaces autour du mot).",
        "ngrams_col": "n-grammes",
        "training": "Entraînement",
        "device": "matériel",
        "duration": "durée",
        "best_epoch": "meilleure époque",
        "val_history": "macro-F1 en validation par époque",
        "training_note": (
            "Affinage : modèle complet, 3 époques, lots de 32, AdamW avec un taux d'apprentissage de 3e-5, montée "
            "linéaire sur 6 % des pas puis décroissance linéaire, weight decay de 0,01, précision mixte, entropie "
            "croisée pondérée par l'inverse de la fréquence des classes, graine 13. L'époque conservée est la "
            "meilleure en validation. Référence : TF-IDF sur n-grammes de caractères de 2 à 5 (`char_wb`), "
            "régression logistique à poids de classes équilibrés, C = {C} choisi en validation."
        ),
        "precision": "précision",
        "recall": "rappel",
        "card": """# Fiche du modèle : Lahja

[English version](model_card.md)

Lahja lit un texte arabe court (une phrase, un message) et dit auquel de cinq groupes il appartient : Golfe, Levant, Égypte, Maghreb ou arabe standard moderne. Cette fiche décrit le modèle de la démo.

## Modèle

- **Architecture** : {sel}, un encodeur BERT-base préentraîné par ses auteurs (`{base}`, révision `{revision}`), avec une tête de classification affinée ici.
- **Format servi** : ONNX, poids quantifiés en entiers 8 bits, exécuté par ONNX Runtime sur CPU.
- **Entrée** : écriture arabe, normalisée par `lahja.normalize` (variantes de l'alif unifiées, diacritiques et tatwil retirés, ta marbuta et alif maqsura ramenés à une seule forme, tout ce qui n'est pas une lettre arabe supprimé). Le modèle est le plus fiable entre 3 et 60 mots.
- **Sortie** : une probabilité par classe, calibrée par mise à l'échelle de la température sur le jeu de validation.
- **Licence** : CC BY-NC-SA 4.0 pour le modèle, parce qu'un des corpus d'entraînement (SADA) est non commercial et à partage dans les mêmes conditions. Le code est sous MIT.

## Usage prévu

Une démo et un projet de portfolio : deviner la grande région d'origine d'une phrase écrite. À ne pas utiliser pour prendre des décisions sur des personnes (origine, nationalité, asile, embauche). Un dialecte dit peu de chose de l'endroit d'où vient quelqu'un, et rien de qui il est.

## Données d'entraînement

{n_train} phrases d'entraînement issues de trois corpus sous licence ouverte, découpées 80/10/10 de sorte qu'aucune phrase source, aucune émission et aucun mot-vedette ne se retrouve des deux côtés.

{data_table}

- **FLORES-200** (Meta AI) : des phrases de Wikipédia traduites par des professionnels en arabe standard et en dialectes. Sa version « najdi » recopie le plus souvent la phrase en arabe standard mot pour mot ; ces copies sont retirées, si bien que ce corpus n'apporte que quelques centaines de phrases du Golfe.
- **SADA** (SDAIA et l'autorité saoudienne de l'audiovisuel) : des transcriptions d'émissions de télévision saoudiennes, étiquetées avec le dialecte du locuteur. Il fournit l'essentiel de la classe Golfe, et un peu d'égyptien et de levantin. Son étiquette « arabe standard » décrit le locuteur plutôt que la phrase, elle n'est donc pas utilisée.
- **Lahga** (lahga.fyi) : un dictionnaire collaboratif. Ses phrases d'exemple donnent la langue de tous les jours dans chaque dialecte, ses définitions de courtes phrases en arabe standard. D'après son mainteneur, plus de la moitié des formes ont d'abord été rédigées par un modèle de langue, puis relues par une personne.

Phrases du Golfe par pays :

{countries}

## Résultats

Jeu de test, modèle int8 tel qu'il est servi : macro-F1 {f1} %, exactitude {acc} %, erreur de calibration (ECE) {ece}.

{per_class}

Hors domaine :

{ood_lines}

La comparaison complète avec les deux autres modèles, les matrices de confusion et la courbe de calibration se trouvent dans le [rapport d'évaluation](evaluation.fr.md).

## Limites connues

- **Golfe veut surtout dire Arabie saoudite.** Presque toutes les phrases du Golfe à l'entraînement sont saoudiennes (tableau ci-dessus). Les parlers des Émirats, du Qatar, du Koweït, de Bahreïn et d'Oman sont à peine présents, et la version 2 par pays demandera d'autres données.
- **L'Irak, le Yémen, le Soudan et la Mauritanie sont hors périmètre.** Leurs dialectes reçoivent quand même une des cinq étiquettes, et aucun jeu de test ne dit laquelle.
- **Le registre peut se substituer au dialecte.** Chaque corpus a son registre : le Golfe vient surtout de dialogues télévisés, le Maghreb surtout de phrases de Wikipédia traduites. Le modèle apprend une part du registre au lieu du dialecte : dans les transcriptions télévisées de SADA, beaucoup de phrases levantines et égyptiennes sont classées Golfe, et c'est aussi le cas de nombreux tweets maghrébins (voir le score par corpus et les tests hors domaine dans le rapport d'évaluation).
- **Textes courts.** En dessous de trois mots, la réponse tient presque du hasard, et la démo le signale.
- **Non pris en charge** : les dialectes mélangés, l'arabizi (arabe en lettres latines), l'alternance avec une autre langue. Les lettres latines sont retirées avant que le modèle ne voie le texte.
- **Golfe soutenu et arabe standard** se distinguent mal quand la phrase compte peu de mots dialectaux.

## Latence

{p95} ms au 95e centile par phrase via l'API, sur 2 threads CPU.

## Sources

FLORES-200 (Meta AI, CC BY-SA 4.0), SADA (SDAIA et SBA, CC BY-NC-SA 4.0), Lahga (lahga.fyi, CC BY-SA 4.0), CAMeLBERT-DA (CAMeL Lab, Apache 2.0), MARBERT (UBC-NLP). Les tweets servent uniquement à l'évaluation (étiquettes QADI, QCRI) ; aucun texte de tweet n'est publié.""",
    },
}
