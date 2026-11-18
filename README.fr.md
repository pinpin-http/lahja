# Lahja

[English version](README.md) · **Démo : [lahja.soheib.fr](https://lahja.soheib.fr)** · Modèle : [Hugging Face](https://huggingface.co/pinpin-http/lahja)

Lahja (لهجة, « dialecte ») lit une phrase arabe et dit de quelle région elle vient : Golfe, Levant, Égypte, Maghreb ou arabe standard moderne. C'est un MARBERT affiné sur trois corpus sous licence ouverte, quantifié en entiers 8 bits. Sur la démo, il tourne dans le navigateur avec ONNX Runtime Web, et la phrase ne quitte pas l'appareil.

| Modèle | macro-F1 test | macro-F1 tweets (hors domaine) |
|---|---:|---:|
| n-grammes de caractères + régression logistique | 70,8 | 58,7 |
| CAMeLBERT-DA | 78,1 | 70,8 |
| MARBERT | 80,8 | 78,7 |
| MARBERT int8, dans le navigateur | 80,1 | 78,7 |

Le test compte 2 749 phrases (FLORES-200, SADA, Lahga), sans phrase source, émission ni mot-vedette partagés avec l'entraînement ; les tweets viennent d'une source jamais vue. Le point faible, ce sont les dialogues télévisés saoudiens de SADA, où 45 % des phrases levantines sont prises pour du Golfe.

Pour aller plus loin : [rapport d'évaluation](docs/evaluation.fr.md), [fiche du modèle](docs/model_card.fr.md), [description complète](docs/details.fr.md) (données, choix, limites).

## Relancer

```bash
make install TRAIN=train-rocm   # GPU AMD ; `make install` pour NVIDIA
make all                        # données, référence, affinage, export ONNX, évaluation, rapport, publication
make serve                      # démo locale : http://127.0.0.1:8000
make test
```

L'affinage des six modèles prend 18 min sur une Radeon RX 9060 XT, et le reste tourne sur CPU. `make hub` publie le modèle sur Hugging Face, `make site deploy-site` met la page en ligne sur Cloudflare.

## Licence

Code sous MIT, modèle sous CC BY-NC-SA 4.0 (SADA est non commercial). Les corpus sont téléchargés à la source, jamais redistribués ici.

---

Un projet de [Soheib](https://soheib.fr).
