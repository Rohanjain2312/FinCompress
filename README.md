# FinCompress — Compressing FinBERT for Production Inference

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HF%20Spaces-blue)](https://huggingface.co/spaces/rohanjain2312/FinCompress)
[![Model](https://img.shields.io/badge/🤗%20Student%20Weights-HF%20Hub-orange)](https://huggingface.co/rohanjain2312/FinCompress_student)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## About

**Rohan Jain** — MS Machine Learning, University of Maryland

MS ML student at UMD with a background in data science and analytics, focused on applied ML systems and NLP. This project explores a complete model compression pipeline — knowledge distillation, INT8 quantization, and structured pruning — applied to a domain-specific financial NLP model. The goal: get a production-viable model that actually fits on a CPU, not just one that scores well in a notebook.

| | |
|---|---|
| 🐙 GitHub | [github.com/Rohanjain2312](https://github.com/Rohanjain2312) |
| 🤗 HuggingFace | [huggingface.co/rohanjain2312](https://huggingface.co/rohanjain2312) |
| 💼 LinkedIn | [linkedin.com/in/jaroh23](https://www.linkedin.com/in/jaroh23/) |
| 📧 Email | jaroh23@umd.edu |

---

## What It Is

A systematic compression study on **FinBERT** (ProsusAI/finbert — BERT-base pre-trained on 4.9B tokens of financial text) for 3-class financial sentiment classification. Five compression techniques are implemented from scratch, each trained end-to-end on Google Colab and benchmarked on identical CPU hardware.

| Try it | |
|---|---|
| 🤗 [HF Spaces — no setup required](https://huggingface.co/spaces/rohanjain2312/FinCompress) | Teacher vs. student side-by-side, live benchmark table |
| 📓 [Google Colab — full pipeline](notebooks/fincompress_complete.ipynb) | Single notebook: train all 7 variants on a T4 GPU |

---

## Results

Training hardware: Google Colab T4 GPU. Benchmarking: CPU (median latency over 500 runs, 50 warmup).

| Model | Params | Size | Val Macro F1 | vs Teacher |
|-------|--------|------|-------------|------------|
| Teacher (FinBERT fine-tuned) | 109M | 437.9 MB | **0.8876** | baseline |
| Student — Vanilla KD | 19M | 76.1 MB | 0.8017 | 5.8× smaller · −8.6 F1 pts |
| Student — Intermediate KD | 19M | 76.1 MB | 0.7712 | 5.8× smaller · −11.6 F1 pts |
| Student — PTQ (INT8) | 12M | 47.7 MB | 0.7712 | **9.1× smaller** · same F1 as FP32 |
| Student — QAT (INT8) | 12M | 47.7 MB | 0.7601 | 9.1× smaller |
| Pruned Teacher 30% | 109M | 437.9 MB | **0.8966** | ↑ beats teacher by +0.9 pts |
| Pruned Teacher 50% | 109M | 437.9 MB | **0.8936** | ↑ beats teacher by +0.6 pts |

**Surprising finding:** Removing 30–50% of attention heads *improved* accuracy. Low-entropy heads (near-uniform attention distributions) add noise rather than signal — pruning them acts as structured regularisation. Full latency and throughput numbers are in the executed notebook.

---

## Compression Pipeline

```
                   FinBERT Teacher
                  (12 layers, 768d)
                  ProsusAI/finbert
                  Fine-tune on GPU
                        │
         ┌──────────────┼──────────────┐
         │              │              │
   Knowledge        Quantization   Structured
   Distillation                     Pruning
         │              │              │
   ┌─────┴─────┐   ┌────┴────┐   ┌────┴────┐
   │           │   │         │   │         │
 Vanilla   Intermed.  PTQ    QAT  30%     50%
   KD        KD    (INT8) (INT8) pruned  pruned
 (4L/384d) (4L/384d)
         │              │              │
         └──────────────┴──────────────┘
                        │
               CPU Benchmark (7 variants)
               Median latency · Macro F1 · Throughput
```

---

## Engineering Concepts Demonstrated

| Concept | Implementation |
|---------|----------------|
| **Knowledge distillation** | Soft-label KL divergence with temperature scaling (T=4) and T² gradient correction; layer-mapped hidden-state MSE + attention-pattern MSE for intermediate KD |
| **INT8 dynamic quantization** | `torch.quantization.quantize_dynamic` (fbgemm backend); QAT with fake-quant ops and straight-through estimator (STE) for gradient-through-discrete-ops |
| **Structured attention pruning** | Per-head entropy scoring over validation set; iterative prune + fine-tune recovery (5 rounds × 3 epochs); importance metric derived from attention distribution uniformity |
| **Custom transformer from scratch** | 4-layer BERT-style encoder in pure `torch.nn` — multi-head self-attention, positional + segment + token embeddings, GELU FFN, LayerNorm residuals |
| **Benchmarking protocol** | Median-over-500 CPU latency (not mean — right-skewed distribution from GC pauses); 50 warmup runs; throughput measured at batch=32 |
| **End-to-end reproducibility** | Single Colab notebook trains all 7 variants in sequence; `checkpoint_info.json` per model captures hyperparameters + metrics + timestamp |

---

## Techniques Covered

| Technique | Method | Reference |
|-----------|--------|-----------|
| **Vanilla KD** | Soft-label KL loss + CE loss, temperature scaling | [Hinton et al., 2015](https://arxiv.org/abs/1503.02531) |
| **Intermediate KD** | Hidden-state MSE + attention-pattern MSE at mapped layers | [TinyBERT, Jiao et al., 2020](https://arxiv.org/abs/1909.10351) |
| **INT8 PTQ** | Post-training dynamic quantization (fbgemm) | [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html) |
| **INT8 QAT** | Fake-quant + straight-through estimator fine-tuning | [PyTorch QAT Guide](https://pytorch.org/blog/quantization-in-practice/) |
| **Structured Pruning** | Entropy-based head importance, iterative prune + recover | [Michel et al., 2019](https://arxiv.org/abs/1905.10650) |

---

## How to Run

### Colab — full pipeline (recommended)

Open [`notebooks/fincompress_complete.ipynb`](notebooks/fincompress_complete.ipynb) in Google Colab with a T4 GPU runtime. The notebook runs the complete pipeline top-to-bottom: dataset prep → teacher training → distillation → quantization → pruning → benchmarking → plots. Runtime: ~3–4 hours total.

### Local — benchmark only

```bash
git clone https://github.com/Rohanjain2312/FinCompress.git
cd FinCompress/fincompress
pip install -r requirements.txt

# Download checkpoints from Google Drive into fincompress/checkpoints/
python -m fincompress.evaluation.benchmark
```

### Local — dataset prep only

```bash
python -m fincompress.data.prepare_dataset
# → fincompress/data/train.csv, val.csv, test.csv
```

---

## Key Learnings

### Knowledge Distillation
- Soft labels encode class uncertainty that hard one-hot labels discard — the teacher's [0.05, 0.72, 0.23] output teaches far more than the label "neutral"
- **T² scaling is non-optional:** high temperature flattens the soft distribution, reducing gradient magnitude of the KL term; multiplying by T² restores it — without this, CE loss dominates and you lose most of the distillation signal
- Layer mapping strategy matters: evenly-spaced pairing `{0→2, 1→5, 2→8, 3→11}` forces the student to mimic the full representational hierarchy (syntactic early layers, semantic middle, task-specific final), not just the output layers

### INT8 Quantization
- PTQ is nearly free — apply it as the first compression step before committing to QAT's training cost
- The first and last layers are sensitivity cliffs: quantizing the embedding layer and final classifier causes disproportionate F1 loss; `quantize_dynamic` wisely excludes them by default
- **STE is elegant:** the straight-through estimator propagates gradients through the floor() rounding operation as if it were identity during backprop — a simple trick that makes an otherwise non-differentiable step trainable

### Structured Pruning
- 30–50% of FinBERT's attention heads are redundant for 3-class financial sentiment — entropy scoring identifies them cheaply without computing saliency gradients
- **The regularisation effect is real:** removing redundant heads reduced overfitting enough to improve val F1 by +0.9 pts — the original teacher was slightly over-parameterised for this task
- The accuracy cliff is sharp: there is a head-count threshold beyond which each additional pruning round causes rapid F1 collapse; iterative pruning with recovery rounds defers this cliff significantly

---

## Project Structure

```
fincompress/
├── data/
│   └── prepare_dataset.py           Download + merge + split (CPU)
├── teacher/
│   └── train_teacher.py             Fine-tune FinBERT teacher (Colab/GPU)
├── distillation/
│   ├── student_architecture.py      4-layer custom transformer from scratch
│   ├── soft_label_distillation.py   Vanilla KD training loop (Colab/GPU)
│   └── intermediate_distillation.py Hidden + attention KD (Colab/GPU)
├── quantization/
│   ├── ptq.py                       Post-training INT8 quantization (CPU)
│   └── qat.py                       Quantization-aware training (Colab/GPU)
├── pruning/
│   ├── structured_pruning.py        Entropy-based head scoring + pruning
│   └── prune_finetune.py            Iterative prune + recover loop (Colab/GPU)
├── evaluation/
│   └── benchmark.py                 Master benchmark — 7 variants, CPU
├── checkpoints/                     Gitignored (large binaries on Drive)
│   └── */checkpoint_info.json       ✅ Committed — hyperparams + metrics
├── results/                         benchmark_results.csv/json committed here
└── logs/                            Training CSVs (gitignored)
notebooks/
└── fincompress_complete.ipynb       Single notebook — full pipeline on Colab
hf_space/
└── app.py                           Gradio 6 demo → HF Spaces
```

---

## Hardware Requirements

| Task | Hardware | Est. Time |
|------|----------|-----------|
| Dataset prep | Any CPU | ~2 min |
| Teacher fine-tuning | GPU ≥ 8 GB VRAM | ~30 min on T4 |
| Vanilla KD | GPU ≥ 8 GB VRAM | ~30 min on T4 |
| Intermediate KD | GPU ≥ 8 GB VRAM | ~45 min on T4 |
| QAT | GPU ≥ 8 GB VRAM | ~15 min on T4 |
| Pruning (both variants) | GPU ≥ 8 GB VRAM | ~60 min on T4 |
| PTQ + Benchmarking | CPU (x86) | ~15 min |

Google Colab free tier (T4) is sufficient for all GPU tasks.

---

## References

1. Hinton, G., Vinyals, O., Dean, J. (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
2. Jiao, X. et al. (2020). [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)
3. Michel, P. et al. (2019). [Are Sixteen Heads Really Better Than One?](https://arxiv.org/abs/1905.10650). NeurIPS 2019.
4. Araci, D. (2019). [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)
5. PyTorch Team. [Quantization — PyTorch Docs](https://pytorch.org/docs/stable/quantization.html)

---

## Development Notes

The compression pipeline architecture, distillation loss formulations, student architecture design, benchmarking protocol, and all key engineering decisions were designed and authored by Rohan Jain. [Claude Code](https://claude.ai/code) was used as an implementation accelerator for boilerplate, file scaffolding, and debugging — similar to how a senior engineer uses Copilot while retaining full design ownership.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by Rohan Jain — UMD MSML, Spring 2026*
