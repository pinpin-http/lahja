# Fiche du modèle : Lahja

[English version](model_card.md)

Lahja lit un texte arabe court (une phrase, un message) et dit auquel de cinq groupes il appartient : Golfe, Levant, Égypte, Maghreb ou arabe standard moderne. Cette fiche décrit le modèle de la démo.

## Modèle

- **Architecture** : MARBERT, un encodeur BERT-base préentraîné par ses auteurs (`UBC-NLP/MARBERT`, révision `88e1fa1`), avec une tête de classification affinée ici.
- **Format servi** : ONNX, poids quantifiés en entiers 8 bits, exécuté par ONNX Runtime sur CPU.
- **Entrée** : écriture arabe, normalisée par `lahja.normalize` (variantes de l'alif unifiées, diacritiques et tatwil retirés, ta marbuta et alif maqsura ramenés à une seule forme, tout ce qui n'est pas une lettre arabe supprimé). Le modèle est le plus fiable entre 3 et 60 mots.
- **Sortie** : une probabilité par classe, calibrée par mise à l'échelle de la température sur le jeu de validation.
- **Licence** : CC BY-NC-SA 4.0 pour le modèle, parce qu'un des corpus d'entraînement (SADA) est non commercial et à partage dans les mêmes conditions. Le code est sous MIT.

## Usage prévu

Une démo et un projet de portfolio : deviner la grande région d'origine d'une phrase écrite. À ne pas utiliser pour prendre des décisions sur des personnes (origine, nationalité, asile, embauche). Un dialecte dit peu de chose de l'endroit d'où vient quelqu'un, et rien de qui il est.

## Données d'entraînement

18 923 phrases d'entraînement issues de trois corpus sous licence ouverte, découpées 80/10/10 de sorte qu'aucune phrase source, aucune émission et aucun mot-vedette ne se retrouve des deux côtés.

|  | Golfe | Levant | Égypte | Maghreb | Arabe standard | total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| entraînement | 5 498 | 3 991 | 3 043 | 3 542 | 2 849 | 18 923 |
| validation | 656 | 650 | 316 | 440 | 356 | 2 418 |
| test | 652 | 764 | 534 | 443 | 356 | 2 749 |
| tweets (hors domaine) | – | 1 000 | 1 000 | 1 000 | – | 3 000 |

- **FLORES-200** (Meta AI) : des phrases de Wikipédia traduites par des professionnels en arabe standard et en dialectes. Sa version « najdi » recopie le plus souvent la phrase en arabe standard mot pour mot ; ces copies sont retirées, si bien que ce corpus n'apporte que quelques centaines de phrases du Golfe.
- **SADA** (SDAIA et l'autorité saoudienne de l'audiovisuel) : des transcriptions d'émissions de télévision saoudiennes, étiquetées avec le dialecte du locuteur. Il fournit l'essentiel de la classe Golfe, et un peu d'égyptien et de levantin. Son étiquette « arabe standard » décrit le locuteur plutôt que la phrase, elle n'est donc pas utilisée.
- **Lahga** (lahga.fyi) : un dictionnaire collaboratif. Ses phrases d'exemple donnent la langue de tous les jours dans chaque dialecte, ses définitions de courtes phrases en arabe standard. D'après son mainteneur, plus de la moitié des formes ont d'abord été rédigées par un modèle de langue, puis relues par une personne.

Phrases du Golfe par pays :

|  | n |
| :--- | ---: |
| Arabie saoudite | 6 679 |
| Émirats | 1 |
| Qatar | 0 |
| Koweït | 0 |
| Bahreïn | 0 |
| Oman | 0 |
| non précisé (Lahga « gulf ») | 126 |

## Résultats

Jeu de test, modèle int8 tel qu'il est servi : macro-F1 80,4 %, exactitude 79,2 %, erreur de calibration (ECE) 0,074.

| classe | précision | rappel | F1 | n |
| :--- | ---: | ---: | ---: | ---: |
| Golfe | 67,5 | 80,5 | 73,4 | 652 |
| Levant | 88,0 | 67,4 | 76,3 | 764 |
| Égypte | 78,5 | 71,7 | 75,0 | 534 |
| Maghreb | 88,5 | 91,9 | 90,1 | 443 |
| Arabe standard | 79,0 | 97,2 | 87,2 | 356 |

Hors domaine :

- tweets (hors domaine) : macro-F1 78,5

La comparaison complète avec les deux autres modèles, les matrices de confusion et la courbe de calibration se trouvent dans le [rapport d'évaluation](evaluation.fr.md).

## Limites connues

- **Golfe veut surtout dire Arabie saoudite.** Presque toutes les phrases du Golfe à l'entraînement sont saoudiennes (tableau ci-dessus). Les parlers des Émirats, du Qatar, du Koweït, de Bahreïn et d'Oman sont à peine présents, et la version 2 par pays demandera d'autres données.
- **L'Irak, le Yémen, le Soudan et la Mauritanie sont hors périmètre.** Leurs dialectes reçoivent quand même une des cinq étiquettes, et aucun jeu de test ne dit laquelle.
- **Le registre peut se substituer au dialecte.** Chaque corpus a son registre : le Golfe vient surtout de dialogues télévisés, le Maghreb surtout de phrases de Wikipédia traduites. Le modèle apprend une part du registre au lieu du dialecte : dans les transcriptions télévisées de SADA, beaucoup de phrases levantines et égyptiennes sont classées Golfe, et c'est aussi le cas de nombreux tweets maghrébins (voir le score par corpus et les tests hors domaine dans le rapport d'évaluation).
- **Textes courts.** En dessous de trois mots, la réponse tient presque du hasard, et la démo le signale.
- **Non pris en charge** : les dialectes mélangés, l'arabizi (arabe en lettres latines), l'alternance avec une autre langue. Les lettres latines sont retirées avant que le modèle ne voie le texte.
- **Golfe soutenu et arabe standard** se distinguent mal quand la phrase compte peu de mots dialectaux.

## Latence

24,0 ms au 95e centile par phrase via l'API, sur 2 threads CPU.

## Sources

FLORES-200 (Meta AI, CC BY-SA 4.0), SADA (SDAIA et SBA, CC BY-NC-SA 4.0), Lahga (lahga.fyi, CC BY-SA 4.0), CAMeLBERT-DA (CAMeL Lab, Apache 2.0), MARBERT (UBC-NLP). Les tweets servent uniquement à l'évaluation (étiquettes QADI, QCRI) ; aucun texte de tweet n'est publié.
