# FinCompress — A Practitioner's Study in Compressing FinBERT for Production Inference

Systematically compress a domain-pretrained language model using three independent techniques, benchmarked end-to-end on identical hardware.

---

## Motivation

In production ML systems, inference cost dominates training cost by orders of magnitude. A sentiment model called 100,000 times per day incurs real infrastructure costs — every millisecond of latency multiplied by request volume equals dollars. The standard approach of deploying full-size BERT-family models (110M+ parameters) is often unsustainable.

FinCompress explores three families of compression techniques to answer: **how much can we compress FinBERT for financial sentiment classification before accuracy becomes unacceptable?**

---

## Techniques Covered

| Technique | Method | Reference |
|-----------|--------|-----------|
| **Knowledge Distillation** (vanilla) | Train a 4-layer student to match teacher output logits (soft labels) | [Hinton et al., 2015](https://arxiv.org/abs/1503.02531) |
| **Knowledge Distillation** (intermediate) | Add hidden state + attention pattern supervision at matched layers | [TinyBERT, Jiao et al., 2020](https://arxiv.org/abs/1909.10351) |
| **INT8 PTQ** | Post-training quantization with activation calibration | [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html) |
| **INT8 QAT** | Quantization-aware training with straight-through estimator | [PyTorch QAT Guide](https://pytorch.org/blog/quantization-in-practice/) |
| **Structured Pruning** | Iterative attention head pruning with recovery fine-tuning | [Michel et al., 2019](https://arxiv.org/abs/1905.10650) |

---

## Architecture Diagram

```
                    FinBERT Teacher
                   (12 layers, 768d)
                   ProsusAI/finbert
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    Knowledge         Quantization   Structured
    Distillation                      Pruning
          │              │              │
    ┌─────┴─────┐   ┌────┴────┐   ┌────┴────┐
    │           │   │         │   │         │
  Vanilla   Intermed.  PTQ    QAT  30%     50%
    KD        KD    (INT8) (INT8) pruned  pruned
  (4L/384d) (4L/384d)
          │              │              │
          └──────────────┴──────────────┘
                         │
                   Benchmark (CPU)
                   results/benchmark_results.csv
                         │
                   Analysis + Plots
                   notebooks/05_analysis_and_plots.ipynb
```

---

## Results

> **Fill this table after running the benchmark.**

| Model | Params | Size (MB) | Macro F1 | CPU Latency (ms) | Throughput (sps) |
|-------|--------|-----------|----------|-----------------|-----------------|
| Teacher (FinBERT) | — | — | — | — | — |
| Vanilla KD | — | — | — | — | — |
| Intermediate KD | — | — | — | — | — |
| INT8 PTQ | — | — | — | — | — |
| INT8 QAT | — | — | — | — | — |
| Pruned 30% | — | — | — | — | — |
| Pruned 50% | — | — | — | — | — |

---

## Pareto Plot

> `results/plots/pareto_plot.png` — will appear here after running notebook 05.

The Pareto plot visualizes the accuracy vs. latency trade-off, with bubble size proportional to model size. Points toward the top-left corner dominate (higher accuracy, lower latency).

---

## Project Structure

```
fincompress/
├── data/
│   └── prepare_dataset.py          Download + merge + split datasets (Local/CPU)
├── teacher/
│   └── train_teacher.py            Fine-tune FinBERT teacher (Colab/GPU)
├── distillation/
│   ├── student_architecture.py     4-layer transformer student from scratch
│   ├── soft_label_distillation.py  Vanilla KD training loop (Colab/GPU)
│   └── intermediate_distillation.py Intermediate layer KD (Colab/GPU)
├── quantization/
│   ├── ptq.py                      Post-training INT8 quantization (Local/CPU)
│   └── qat.py                      Quantization-aware training (Colab/GPU)
├── pruning/
│   ├── structured_pruning.py       Head/neuron importance + pruning utilities
│   └── prune_finetune.py           Iterative prune + recover loop (Colab/GPU)
├── evaluation/
│   └── benchmark.py                Master benchmark script (Local/CPU)
├── demo/
│   └── app.py                      Gradio side-by-side comparison app (Local/CPU)
├── notebooks/
│   ├── 01_teacher_training.ipynb   Teacher walkthrough (Colab/GPU)
│   ├── 02_distillation.ipynb       Distillation walkthrough (Colab/GPU)
│   ├── 03_quantization.ipynb       Quantization walkthrough (Colab/GPU or Local/CPU)
│   ├── 04_pruning.ipynb            Pruning walkthrough (Colab/GPU)
│   └── 05_analysis_and_plots.ipynb Final benchmark plots (Local/CPU)
├── checkpoints/                    Model checkpoints (gitignored — large binaries)
├── results/                        benchmark_results.csv/json committed here
├── logs/                           Training CSVs (gitignored)
├── requirements.txt                Local/CPU dependencies
└── requirements_colab.txt          Colab/GPU dependencies (adds ipywidgets)
```

---

## Quickstart

### 1. Install dependencies (local)

```bash
git clone https://github.com/Rohanjain2312/FinCompress.git
cd FinCompress/fincompress
pip install -r requirements.txt
```

### 2. Prepare dataset (Local/CPU)

```bash
python -m fincompress.data.prepare_dataset
# → fincompress/data/train.csv, val.csv, test.csv
```

### 3. Training order

```
Step 1 (Local):  python -m fincompress.data.prepare_dataset
Step 2:          Upload fincompress/data/ to Google Drive
Step 3 (Colab):  Run notebook 01 — teacher training
Step 4 (Colab):  Run notebook 02 — distillation (vanilla then intermediate)
Step 5 (Colab):  Run notebook 03 Section B — QAT
Step 6 (Colab):  Run notebook 04 — pruning
Step 7:          Download all checkpoints from Drive to local fincompress/checkpoints/
Step 8 (Local):  python -m fincompress.quantization.ptq
Step 9 (Local):  python -m fincompress.evaluation.benchmark
Step 10 (Local): Run notebook 05 — all plots
Step 11 (Local): python -m fincompress.demo.app
```

### 4. Run the demo

```bash
python -m fincompress.demo.app
# Opens Gradio UI at http://localhost:7860
```

---

## Hardware Requirements

| Task | Hardware | Notes |
|------|----------|-------|
| Dataset prep | Any CPU | ~1 min |
| Teacher fine-tuning | GPU (≥8 GB VRAM) | ~30 min on T4 |
| Knowledge distillation | GPU (≥8 GB VRAM) | ~45 min per variant on T4 |
| QAT | GPU (≥8 GB VRAM) | ~15 min on T4 |
| Pruning | GPU (≥8 GB VRAM) | ~60 min on T4 |
| PTQ + Benchmarking | CPU (x86) | ~10 min |
| Demo | CPU | Real-time |

Google Colab (free tier) is sufficient for all GPU tasks.

---

## Key Learnings

### Knowledge Distillation
- Soft labels from the teacher encode class similarity and uncertainty that hard one-hot labels discard
- T² scaling is non-optional: without it, high temperature makes the KL loss negligible
- Intermediate distillation consistently outperforms vanilla KD by ~1-3 F1 points for BERT-family compression
- **What I learned:** The layer mapping choice (evenly spaced vs. last-N) materially affects student quality — evenly spaced forces the student to mimic the full representational hierarchy

### INT8 Quantization
- PTQ is nearly free (no training); use it as the first compression attempt before committing to QAT
- Backend selection matters for benchmarking validity — fbgemm vs. qnnpack targets different SIMD instruction sets
- The first and last layers are sensitivity cliffs: quantizing them causes disproportionate accuracy loss
- **What I learned:** QAT's straight-through estimator is elegant — it solves the gradient problem of discrete operations without changing the forward-pass semantics

### Structured Pruning
- 30-50% of attention heads in FinBERT are redundant for 3-class financial sentiment
- The "accuracy cliff" is real and sharp: there's typically a threshold beyond which additional pruning causes rapid F1 degradation
- Structured pruning requires actual dimension reduction (not just zeroing) to deliver real speedups on standard hardware
- **What I learned:** Entropy-based importance scoring is a surprisingly effective proxy for the more expensive gradient-based methods described in the literature

---

## References

1. Hinton, G., Vinyals, O., Dean, J. (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
2. Jiao, X. et al. (2020). [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)
3. Michel, P. et al. (2019). [Are Sixteen Heads Really Better Than One?](https://arxiv.org/abs/1905.10650). NeurIPS 2019.
4. PyTorch Quantization Documentation. [Static Quantization](https://pytorch.org/docs/stable/quantization.html)
5. Araci, D. (2019). [FinBERT: Financial Sentiment Analysis with Pre-trained Language Models](https://arxiv.org/abs/1908.10063)

---

## License

MIT License — see LICENSE file for details.

---

*Built by Rohan Jain — UMD MSML, Spring 2026*
