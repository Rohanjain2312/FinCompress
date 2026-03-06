#!/usr/bin/env python3
"""
generate_article_assets.py
============================
Generates 4 publication-quality PNG charts for the FinCompress Medium article.

Outputs (saved to fincompress/results/article_assets/):
  01_pruning_curve.png     — F1 vs. heads pruned % across 5 iterative rounds
  02_cpu_latency.png       — CPU latency bar chart for all 7 model variants
  03_entropy_heatmap.png   — Per-head attention entropy heatmap (12×12 teacher)
  04_kd_curves.png         — Val F1 per epoch: Vanilla KD vs. Intermediate KD
  article_data.json        — Raw numbers for all 4 assets

Run:
    python fincompress/generate_article_assets.py

Runtime: ~15 min total (benchmark dominates at ~12 min; entropy ~3 min).
"""

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent          # fincompress/
PROJECT_ROOT = BASE_DIR.parent                # repo root
CKPT_DIR     = BASE_DIR / "checkpoints"
DATA_DIR     = BASE_DIR / "data"
RESULTS_DIR  = BASE_DIR / "results"
OUTPUT_DIR   = RESULTS_DIR / "article_assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ensure fincompress is importable when calling benchmark
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Global style ───────────────────────────────────────────────────────────────
FIG_W, FIG_H = 16, 9
DPI = 100

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         14,
    "axes.titlesize":    18,
    "axes.titleweight":  "bold",
    "axes.labelsize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   13,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "axes.grid":         True,
    "grid.alpha":        0.5,
    "grid.color":        "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Accumulate raw data for article_data.json ─────────────────────────────────
article_data: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# ASSET 1 — Pruning Curve
# Data: hardcoded from verified notebook cell outputs (5 rounds × teacher)
# ══════════════════════════════════════════════════════════════════════════════

def make_pruning_curve() -> None:
    print("\n[Asset 1] Pruning curve...")

    # Per-round data extracted from executed Colab notebook outputs
    sparsity = [0.0,   9.7,   19.4,  29.2,  38.9,  48.6]
    val_f1   = [0.887626, 0.8818, 0.8905, 0.8966, 0.8961, 0.8936]
    teacher_baseline = 0.887626
    peak_idx = int(np.argmax(val_f1))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # Shade improvement region
    ax.fill_between(
        sparsity, teacher_baseline, val_f1,
        where=[f > teacher_baseline for f in val_f1],
        alpha=0.12, color="#27ae60",
        label="Beats teacher baseline",
    )

    # Main line
    ax.plot(
        sparsity, val_f1, "o-",
        color="darkorange", linewidth=3,
        markersize=10, markeredgecolor="white", markeredgewidth=2.5, zorder=5,
        label="Val Macro F1 (after recovery fine-tune)",
    )

    # Teacher baseline dashed
    ax.axhline(
        teacher_baseline, color="#555555", linestyle="--", linewidth=2.2,
        alpha=0.85, label=f"Teacher baseline  (F1 = {teacher_baseline:.4f})",
    )

    # Annotate peak
    ax.annotate(
        f"Peak: F1 = {val_f1[peak_idx]:.4f}\n@ {sparsity[peak_idx]:.1f}% pruned",
        xy=(sparsity[peak_idx], val_f1[peak_idx]),
        xytext=(sparsity[peak_idx] - 8, val_f1[peak_idx] + 0.0035),
        fontsize=13, fontweight="bold", color="darkorange",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
    )

    # Annotate each round
    round_labels = ["Baseline", "Round 1", "Round 2", "Round 3", "Round 4", "Round 5"]
    for i, (x, y, lbl) in enumerate(zip(sparsity, val_f1, round_labels)):
        if i == 0:
            ax.annotate(lbl, (x, y), xytext=(x + 1.5, y - 0.0028),
                        fontsize=11, color="#555555")

    ax.set_xlabel("Attention Heads Pruned (%)", fontsize=14)
    ax.set_ylabel("Val Macro F1", fontsize=14)
    ax.set_title(
        "Iterative Structured Pruning — F1 vs. Heads Pruned\n"
        "FinBERT Teacher (12 layers × 12 heads, 5 rounds × 3 recovery epochs each)",
        fontsize=18, fontweight="bold",
    )
    ax.set_xlim(-3, 56)
    ax.set_ylim(0.868, 0.912)
    ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    out = OUTPUT_DIR / "01_pruning_curve.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    article_data["pruning_curve"] = {
        "sparsity_pct": sparsity,
        "val_f1": val_f1,
        "teacher_baseline": teacher_baseline,
        "peak_sparsity_pct": sparsity[peak_idx],
        "peak_f1": val_f1[peak_idx],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ASSET 2 — CPU Latency Bar Chart
# Data: run benchmark.py if CSV not found, else read existing
# ══════════════════════════════════════════════════════════════════════════════

def make_latency_chart() -> None:
    print("\n[Asset 2] CPU latency bar chart...")

    # Official benchmark numbers from executed Colab notebook (x86 CPU, consistent hardware).
    # Local macOS benchmarking cannot load PTQ/QAT checkpoints (FBGEMM is x86-only).
    # Using Colab numbers for all 7 models ensures apples-to-apples comparison.
    COLAB_DATA = [
        # (model_name, latency_ms, f1_test, size_mb)
        ("teacher",                 286.88, 0.9182, 437.9),
        ("student_vanilla_kd",       24.20, 0.7713,  76.1),
        ("student_intermediate_kd",  24.06, 0.7881,  76.1),
        ("student_ptq",              19.83, 0.7864,  47.7),
        ("student_qat",              20.19, 0.7621,  47.7),
        ("pruned_teacher_30pct",    266.38, 0.9215, 437.9),
        ("pruned_teacher_50pct",    263.52, 0.9125, 437.9),
    ]
    df = pd.DataFrame(COLAB_DATA, columns=["model_name", "cpu_latency_ms_median", "macro_f1", "size_mb"])
    df = df.sort_values("cpu_latency_ms_median").reset_index(drop=True)

    COLORS = {
        "teacher":                 "#888888",
        "student_vanilla_kd":      "#4a90d9",
        "student_intermediate_kd": "#2c5f8a",
        "student_ptq":             "#27ae60",
        "student_qat":             "#1a7a44",
        "pruned_teacher_30pct":    "#e67e22",
        "pruned_teacher_50pct":    "#d35400",
    }
    DISPLAY_NAMES = {
        "teacher":                 "Teacher (FinBERT, 109M)",
        "student_vanilla_kd":      "Vanilla KD Student (19M)",
        "student_intermediate_kd": "Intermediate KD Student (19M)",
        "student_ptq":             "Student PTQ INT8 (12M)",
        "student_qat":             "Student QAT INT8 (12M)",
        "pruned_teacher_30pct":    "Pruned Teacher 30% (109M)",
        "pruned_teacher_50pct":    "Pruned Teacher 50% (109M)",
    }

    colors       = [COLORS.get(m, "#aaaaaa") for m in df["model_name"]]
    display_names = [DISPLAY_NAMES.get(m, m) for m in df["model_name"]]
    latencies    = df["cpu_latency_ms_median"].tolist()
    f1_scores    = df["macro_f1"].tolist()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.barh(
        display_names, latencies,
        color=colors, height=0.55,
        edgecolor="white", linewidth=1.5,
    )

    # Annotate with F1 score
    x_max = max(latencies)
    for bar, f1 in zip(bars, f1_scores):
        w = bar.get_width()
        ax.text(
            w + x_max * 0.01, bar.get_y() + bar.get_height() / 2,
            f"F1 = {f1:.4f}", va="center", fontsize=12, color="#333333",
        )

    # Annotate with latency value inside bar
    for bar, lat in zip(bars, latencies):
        ax.text(
            bar.get_width() * 0.02, bar.get_y() + bar.get_height() / 2,
            f"{lat:.1f} ms", va="center", ha="left",
            fontsize=11, color="white", fontweight="bold",
        )

    legend_patches = [
        mpatches.Patch(color="#888888", label="Teacher (baseline)"),
        mpatches.Patch(color="#4a90d9", label="Knowledge Distillation"),
        mpatches.Patch(color="#27ae60", label="INT8 Quantization"),
        mpatches.Patch(color="#e67e22", label="Structured Pruning"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.9)

    ax.set_xlabel("Median CPU Latency (ms) — single sample, 500 timed runs", fontsize=14)
    ax.set_title(
        "CPU Inference Latency — All 7 FinCompress Model Variants\n"
        "(50 warmup + 500 timed runs · batch size = 1 · Colab x86 CPU · annotated with Test Macro F1)",
        fontsize=18, fontweight="bold",
    )
    ax.invert_yaxis()
    ax.set_xlim(0, x_max * 1.22)  # room for F1 labels

    plt.tight_layout()
    out = OUTPUT_DIR / "02_cpu_latency.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    article_data["cpu_latency"] = df[
        ["model_name", "cpu_latency_ms_median", "macro_f1"]
    ].to_dict(orient="records")


# ══════════════════════════════════════════════════════════════════════════════
# ASSET 3 — Per-Head Entropy Heatmap
# Data: load teacher checkpoint + val.csv, compute entropy via forward passes
# ══════════════════════════════════════════════════════════════════════════════

class _ValDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int = 128) -> None:
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids", torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(0),
        }


def _compute_teacher_head_importance(
    model: torch.nn.Module,
    dataloader: DataLoader,
    num_batches: int = 30,
) -> torch.Tensor:
    """
    Entropy-based head importance for the HuggingFace teacher model.

    Importance = 1 - normalised_entropy, averaged over batches.
    High score  → focused (low entropy) → important → keep.
    Low score   → uniform (high entropy) → redundant → prune.

    Returns [num_layers, num_heads] float32 tensor.
    """
    model.eval()
    config    = model.config
    num_layers = config.num_hidden_layers    # 12
    num_heads  = config.num_attention_heads  # 12

    importance = torch.zeros(num_layers, num_heads)
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            out = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                output_attentions=True,
            )
            # out.attentions: tuple of [batch, num_heads, seq, seq], one per layer
            seq_len = batch["input_ids"].size(1)
            eps = 1e-9
            max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float))

            for layer_idx, attn in enumerate(out.attentions):
                entropy = -(attn * (attn + eps).log()).sum(dim=-1).mean(dim=-1)
                norm_entropy = entropy / (max_entropy + eps)
                importance[layer_idx] += (1.0 - norm_entropy).mean(dim=0)

            count += 1
            if count % 10 == 0:
                print(f"    batch {count}/{num_batches}")

    importance /= count
    return importance


def make_entropy_heatmap() -> None:
    print("\n[Asset 3] Per-head entropy heatmap (loads teacher — ~3 min on CPU)...")

    teacher_dir   = CKPT_DIR / "teacher"
    tokenizer_dir = teacher_dir / "tokenizer"

    print("  Loading teacher and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(teacher_dir))
    model.eval()

    df_val      = pd.read_csv(DATA_DIR / "val.csv")
    val_dataset = _ValDataset(df_val, tokenizer)
    val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print("  Computing head importance over 30 batches...")
    importance = _compute_teacher_head_importance(model, val_loader, num_batches=30)
    # importance: [12, 12]

    # Identify pruned heads at 50% sparsity (actual: 48.6% → ~70 heads of 144)
    n_pruned   = round(importance.numel() * 0.486)
    flat_idx   = importance.view(-1).argsort()[:n_pruned].tolist()
    pruned_mask = torch.zeros(12, 12, dtype=torch.bool)
    for fi in flat_idx:
        pruned_mask[fi // 12, fi % 12] = True

    imp_np = importance.numpy()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    sns.heatmap(
        imp_np, ax=ax,
        cmap="viridis_r",      # dark = high importance (low entropy)
        linewidths=0.8, linecolor="white",
        cbar_kws={
            "label": "Head Importance Score\n(1 − normalised entropy)",
            "shrink": 0.82,
        },
        xticklabels=[str(i) for i in range(12)],
        yticklabels=[str(i) for i in range(12)],
    )

    # Overlay X on pruned heads
    for li in range(12):
        for hi in range(12):
            if pruned_mask[li, hi]:
                ax.text(
                    hi + 0.5, li + 0.5, "X",
                    ha="center", va="center",
                    fontsize=9, color="red", fontweight="bold", alpha=0.85,
                )

    ax.set_xlabel("Attention Head Index", fontsize=14)
    ax.set_ylabel("Transformer Layer", fontsize=14)
    ax.set_title(
        "FinBERT Teacher — Per-Head Attention Importance (12 × 12)\n"
        "Dark = focused head (important). Light = uniform head (redundant). "
        "X = pruned at 50% sparsity.",
        fontsize=18, fontweight="bold",
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "03_entropy_heatmap.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    article_data["entropy_heatmap"] = {
        "importance_matrix": imp_np.tolist(),
        "pruned_head_count":  int(n_pruned),
        "total_heads":        144,
        "sparsity_pct":       48.6,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ASSET 4 — KD Training Curves
# Data: hardcoded from executed Colab notebook cell outputs (verified)
# ══════════════════════════════════════════════════════════════════════════════

def make_kd_curves() -> None:
    print("\n[Asset 4] KD training curves...")

    # Per-epoch Val Macro F1 — extracted from executed notebook cell outputs
    epochs      = list(range(1, 11))
    vanilla_f1  = [0.5191, 0.5346, 0.5435, 0.7648, 0.7608, 0.7837, 0.7926, 0.7813, 0.7892, 0.8017]
    intermed_f1 = [0.4731, 0.5292, 0.6215, 0.7247, 0.7371, 0.7280, 0.7616, 0.7538, 0.7712, 0.7674]

    best_vanilla  = max(vanilla_f1)
    best_intermed = max(intermed_f1)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    ax.plot(
        epochs, vanilla_f1, "o-",
        color="#4a90d9", linewidth=3,
        markersize=9, markeredgecolor="white", markeredgewidth=2,
        label="Vanilla KD  (soft-label KL + CE)",
        zorder=5,
    )
    ax.plot(
        epochs, intermed_f1, "s-",
        color="darkorange", linewidth=3,
        markersize=9, markeredgecolor="white", markeredgewidth=2,
        label="Intermediate KD  (+ hidden-state MSE + attention MSE)",
        zorder=5,
    )

    # Final / best lines
    ax.axhline(
        best_vanilla, color="#4a90d9", linestyle=":", linewidth=2, alpha=0.7,
        label=f"Vanilla KD peak: {best_vanilla:.4f}",
    )
    ax.axhline(
        best_intermed, color="darkorange", linestyle=":", linewidth=2, alpha=0.7,
        label=f"Intermediate KD peak: {best_intermed:.4f}",
    )

    # Annotate peak points
    v_peak_epoch = epochs[vanilla_f1.index(best_vanilla)]
    ax.annotate(
        f"{best_vanilla:.4f}",
        xy=(v_peak_epoch, best_vanilla),
        xytext=(v_peak_epoch - 1.5, best_vanilla + 0.012),
        fontsize=12, color="#4a90d9", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#4a90d9", lw=1.5),
    )

    i_peak_epoch = epochs[intermed_f1.index(best_intermed)]
    ax.annotate(
        f"{best_intermed:.4f}",
        xy=(i_peak_epoch, best_intermed),
        xytext=(i_peak_epoch + 0.5, best_intermed + 0.012),
        fontsize=12, color="darkorange", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.5),
    )

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel("Val Macro F1", fontsize=14)
    ax.set_title(
        "Knowledge Distillation — Val Macro F1 per Epoch\n"
        "4-layer / 384-hidden / 6-head student distilled from 12-layer FinBERT teacher",
        fontsize=18, fontweight="bold",
    )
    ax.set_xticks(epochs)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.43, 0.87)
    ax.legend(loc="lower right", framealpha=0.92)

    plt.tight_layout()
    out = OUTPUT_DIR / "04_kd_curves.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    article_data["kd_curves"] = {
        "epochs":                  epochs,
        "vanilla_kd_val_f1":      vanilla_f1,
        "intermediate_kd_val_f1": intermed_f1,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 64)
    print("FinCompress — Article Asset Generator")
    print("=" * 64)
    print(f"Output directory: {OUTPUT_DIR}")

    make_pruning_curve()    # instant — hardcoded data
    make_kd_curves()        # instant — hardcoded data
    make_entropy_heatmap()  # ~3 min  — loads teacher, forward pass on val set
    make_latency_chart()    # ~12 min — full 7-model CPU benchmark

    out_json = OUTPUT_DIR / "article_data.json"
    with open(out_json, "w") as f:
        json.dump(article_data, f, indent=2)
    print(f"\nRaw data saved: {out_json}")

    print("\n" + "=" * 64)
    print("All assets generated:")
    for name in ["01_pruning_curve.png", "02_cpu_latency.png",
                 "03_entropy_heatmap.png", "04_kd_curves.png", "article_data.json"]:
        p = OUTPUT_DIR / name
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        print(f"  {name:<30}  {size_kb:>6.0f} KB")
    print("=" * 64)
