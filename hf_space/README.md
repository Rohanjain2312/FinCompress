---
title: FinCompress
emoji: 🗜️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
short_description: FinBERT compression via KD, INT8 quant, and pruning
---

# 🗜️ FinCompress

Compressing **FinBERT** (109M params) into a 19M-parameter student using
knowledge distillation, INT8 quantization, and structured attention-head pruning —
all benchmarked on financial sentiment classification.

## What this Space shows

- **Live demo**: run both the 109M teacher and 19M student side-by-side on any financial sentence
- **Benchmark table**: all 7 model variants (teacher, KD students, PTQ, QAT, pruned)
- **Architecture explainer**: how KD, quantization, and pruning each work

## Links

- 📦 [GitHub — FinCompress](https://github.com/Rohanjain2312/FinCompress)
- 🤗 [Student Model Weights](https://huggingface.co/rohanjain2312/FinCompress_student)
- 📊 Dataset: [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank)
