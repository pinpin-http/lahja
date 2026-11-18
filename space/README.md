---
title: Lahja
emoji: 🌈
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
license: cc-by-nc-sa-4.0
short_description: Arabic dialect identification (Gulf, Levant, Egypt, Maghreb, MSA)
models:
- pinpin-http/lahja
---

API of [Lahja](https://lahja.soheib.fr): `POST /predict` with `{"text": "...", "explain": false}`, `GET /health`.
Model: [pinpin-http/lahja](https://huggingface.co/pinpin-http/lahja). By [Soheib](https://soheib.fr).
