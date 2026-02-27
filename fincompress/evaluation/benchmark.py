# RUN ON: Local/CPU
"""
evaluation/benchmark.py
=========================
The master benchmarking script. Loads every trained model variant and evaluates
all of them on the held-out test set using an identical hardware/timing protocol.

This script is the single source of truth for all comparison tables and plots.
All metrics are reproducible by re-running this script on the same hardware.

WHY MEDIAN NOT MEAN (for latency):
CPU latency distributions are right-skewed — occasional outlier spikes from
garbage collection pauses, OS context switches, and CPU frequency scaling
(thermal throttling) can be 10-100x the typical latency. These outliers inflate
the mean dramatically without representing typical user-facing performance.
Median (p50) gives the most representative single-number latency for an SLA
comparison. p95 captures tail latency for SLA analysis where we care about
worst-case user experience, not just the average.

Run:
    python -m fincompress.evaluation.benchmark
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score

from fincompress.distillation.student_architecture import (
    StudentClassifier,
    STUDENT_NUM_LAYERS,
    STUDENT_HIDDEN_SIZE,
    STUDENT_NUM_HEADS,
    STUDENT_INTERMEDIATE_SIZE,
    STUDENT_DROPOUT,
)

# ============================================================
# HYPERPARAMETERS — all tunable values live here, never inline
# ============================================================
SEED = 42
MAX_SEQ_LEN = 128
BATCH_SIZE_TRAIN = 32
NUM_CLASSES = 3
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Latency protocol parameters
BENCHMARK_WARMUP_RUNS = 50
BENCHMARK_TIMED_RUNS = 500
BENCHMARK_BATCH_THROUGHPUT = 32

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
TEACHER_TOKENIZER_DIR = CHECKPOINTS_ROOT / "teacher" / "tokenizer"


# ============================================================
# Model registry
# ============================================================

# Each entry: (model_name, checkpoint_dir, is_teacher_model, is_quantized)
MODEL_REGISTRY: list[tuple[str, Path, bool, bool]] = [
    ("teacher",               CHECKPOINTS_ROOT / "teacher",               True,  False),
    ("student_vanilla_kd",    CHECKPOINTS_ROOT / "student_vanilla_kd",    False, False),
    ("student_intermediate_kd", CHECKPOINTS_ROOT / "student_intermediate_kd", False, False),
    ("student_ptq",           CHECKPOINTS_ROOT / "student_ptq",           False, True),
    ("student_qat",           CHECKPOINTS_ROOT / "student_qat",           False, True),
    ("pruned_teacher_30pct",  CHECKPOINTS_ROOT / "pruned_teacher_30pct",  True,  False),
    ("pruned_teacher_50pct",  CHECKPOINTS_ROOT / "pruned_teacher_50pct",  True,  False),
]


# ============================================================
# Dataset
# ============================================================

class SentimentDataset(Dataset):
    """
    Tokenizes test set examples for benchmarking.

    Args:
        df: DataFrame with [text, label].
        tokenizer: BERT-vocabulary tokenizer.
        max_len: Padding/truncation length.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int) -> None:
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids", torch.zeros(self.max_len, dtype=torch.long)
            ).squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# Model loaders
# ============================================================

def load_teacher_model(checkpoint_dir: Path) -> Optional[nn.Module]:
    """
    Load a HuggingFace teacher or pruned teacher model.

    Args:
        checkpoint_dir: Directory containing saved_pretrained output.

    Returns:
        Model in eval mode, or None if loading fails.
    """
    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint_dir))
        model.eval()
        return model
    except Exception as e:
        print(f"  Error loading teacher from {checkpoint_dir}: {e}")
        return None


def load_student_model(checkpoint_dir: Path, is_quantized: bool) -> Optional[nn.Module]:
    """
    Load a student model (FP32 or quantized INT8).

    For quantized models, we must first prepare the quantization config and
    then load the state dict — because quantized models use different layer
    types that must be present before weights can be mapped.

    Args:
        checkpoint_dir: Directory with pytorch_model.bin.
        is_quantized: Whether the checkpoint contains INT8 quantized weights.

    Returns:
        Model in eval mode, or None if loading fails.
    """
    try:
        model = StudentClassifier(
            hidden_size=STUDENT_HIDDEN_SIZE,
            num_layers=STUDENT_NUM_LAYERS,
            num_heads=STUDENT_NUM_HEADS,
            intermediate_size=STUDENT_INTERMEDIATE_SIZE,
            dropout=STUDENT_DROPOUT,
            num_classes=NUM_CLASSES,
        )

        if is_quantized:
            # For quantized models: apply the same qconfig used during PTQ/QAT,
            # prepare the architecture (inserts quantized layer types), then
            # load the quantized state dict.
            torch.backends.quantized.engine = "fbgemm"
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            model.token_embedding.qconfig = None
            model.position_embedding.qconfig = None
            model.segment_embedding.qconfig = None
            model.classifier.qconfig = None
            torch.quantization.prepare(model, inplace=True)
            torch.quantization.convert(model, inplace=True)

        state_dict = torch.load(
            checkpoint_dir / "pytorch_model.bin",
            map_location="cpu",
        )
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        print(f"  Error loading student from {checkpoint_dir}: {e}")
        return None


# ============================================================
# Benchmarking helpers
# ============================================================

def get_model_size_mb(model: nn.Module) -> float:
    """
    Compute model size in MB including parameters and buffers.

    Args:
        model: Any nn.Module.

    Returns:
        Size in MB.
    """
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / 1e6


def forward_once(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    is_teacher: bool,
) -> None:
    """
    Run a single forward pass (result discarded — used for latency timing).

    Args:
        model: Model to run.
        batch: Input batch dict.
        is_teacher: Whether model is HuggingFace teacher or StudentClassifier.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    token_type_ids = batch["token_type_ids"]

    if is_teacher:
        model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    else:
        model(input_ids, attention_mask, token_type_ids)


def measure_latency(
    model: nn.Module,
    dataloader: DataLoader,
    is_teacher: bool,
) -> tuple[float, float, list[float]]:
    """
    Measure per-sample CPU latency using the standard protocol.

    Protocol:
      1. Run BENCHMARK_WARMUP_RUNS inferences (not timed) to warm caches/JIT.
      2. Run BENCHMARK_TIMED_RUNS inferences, recording each with perf_counter.
      3. Report median (p50) and p95.

    WHY MEDIAN NOT MEAN:
    CPU latency distributions are right-skewed — occasional outlier spikes from
    garbage collection pauses, OS context switches, and CPU frequency scaling
    can be 10-100x the typical latency. These outliers inflate the mean
    dramatically without representing typical user-facing performance. Median
    (p50) gives the most representative single-number latency. p95 captures
    tail latency for SLA analysis.

    Args:
        model: Model in eval mode.
        dataloader: Single-sample (batch_size=1) DataLoader for latency test.
        is_teacher: Whether model uses HuggingFace or StudentClassifier API.

    Returns:
        Tuple of (median_ms, p95_ms, raw_latencies_ms).
    """
    model.eval()
    iterator = iter(dataloader)

    def get_batch() -> dict[str, torch.Tensor]:
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            return next(iterator)

    # Warmup: allow CPU caches, JIT, and thread pools to reach steady state.
    with torch.no_grad():
        for _ in range(BENCHMARK_WARMUP_RUNS):
            forward_once(model, get_batch(), is_teacher)

    # Timed runs
    latencies_ms: list[float] = []
    with torch.no_grad():
        for _ in range(BENCHMARK_TIMED_RUNS):
            batch = get_batch()
            t0 = time.perf_counter()
            forward_once(model, batch, is_teacher)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    median_ms = float(np.median(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    return median_ms, p95_ms, latencies_ms


def measure_throughput(
    model: nn.Module,
    dataloader: DataLoader,
    is_teacher: bool,
) -> float:
    """
    Measure throughput as samples per second using BENCHMARK_BATCH_THROUGHPUT samples.

    Args:
        model: Model in eval mode.
        dataloader: Batch DataLoader for throughput test.
        is_teacher: Whether model uses HuggingFace or StudentClassifier API.

    Returns:
        Throughput in samples/second.
    """
    model.eval()
    batch = next(iter(dataloader))
    actual_batch_size = batch["input_ids"].shape[0]

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            forward_once(model, batch, is_teacher)

    # Time 20 batch passes and compute throughput
    with torch.no_grad():
        t0 = time.perf_counter()
        n_passes = 20
        for _ in range(n_passes):
            forward_once(model, batch, is_teacher)
        t1 = time.perf_counter()

    total_samples = actual_batch_size * n_passes
    elapsed_sec = t1 - t0
    return total_samples / elapsed_sec


def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    is_teacher: bool,
) -> tuple[float, float, dict[str, float]]:
    """
    Evaluate model accuracy on the test set.

    Args:
        model: Model in eval mode.
        dataloader: Test DataLoader (any batch size).
        is_teacher: Whether model uses HuggingFace or StudentClassifier API.

    Returns:
        Tuple of (macro_f1, accuracy, per_class_f1_dict).
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            token_type_ids = batch["token_type_ids"]
            labels = batch["label"]

            if is_teacher:
                out = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
                logits = out.logits
            else:
                out = model(input_ids, attention_mask, token_type_ids)
                logits = out["logits"]

            preds = logits.argmax(dim=-1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)
    per_class = f1_score(all_labels, all_preds, average=None, labels=[0, 1, 2])
    per_class_dict = {LABEL_MAP[i]: round(float(per_class[i]), 6) for i in range(NUM_CLASSES)}

    return macro_f1, accuracy, per_class_dict


# ============================================================
# Main benchmark loop
# ============================================================

def main() -> None:
    """Benchmark all model variants and save results to JSON and CSV."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # All benchmarking on CPU — consistent hardware for fair comparison.
    device = torch.device("cpu")
    print("=" * 70)
    print("FinCompress — Master Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Warmup runs: {BENCHMARK_WARMUP_RUNS} | Timed runs: {BENCHMARK_TIMED_RUNS}\n")

    # --- Validate test set ---
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {test_path}\n"
            "Run 'python -m fincompress.data.prepare_dataset' first to generate it."
        )

    if not TEACHER_TOKENIZER_DIR.exists():
        raise FileNotFoundError(
            f"Required file not found: {TEACHER_TOKENIZER_DIR}\n"
            "Run 'python -m fincompress.teacher.train_teacher' first to generate it."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(TEACHER_TOKENIZER_DIR))
    df_test = pd.read_csv(test_path)

    # DataLoaders: one for accuracy (batch=32), one for latency (batch=1)
    test_dataset = SentimentDataset(df_test, tokenizer, MAX_SEQ_LEN)
    accuracy_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=False)
    latency_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    throughput_loader = DataLoader(test_dataset, batch_size=BENCHMARK_BATCH_THROUGHPUT, shuffle=False)

    # --- Benchmark each model ---
    results: dict[str, dict] = {}
    latency_raw: dict[str, list[float]] = {}

    for model_name, checkpoint_dir, is_teacher, is_quantized in MODEL_REGISTRY:
        print(f"[{model_name}]")

        if not checkpoint_dir.exists():
            print(f"  WARNING: checkpoint not found at {checkpoint_dir} "
                  f"— skipping {model_name}")
            print()
            continue

        # --- Load model ---
        if is_teacher:
            model = load_teacher_model(checkpoint_dir)
        else:
            model = load_student_model(checkpoint_dir, is_quantized)

        if model is None:
            print(f"  ERROR: failed to load {model_name} — skipping")
            print()
            continue

        model.eval()

        # --- Size ---
        num_params = sum(p.numel() for p in model.parameters())
        size_mb = get_model_size_mb(model)
        print(f"  Parameters: {num_params:,}  |  Size: {size_mb:.1f} MB")

        # --- Accuracy ---
        print("  Evaluating accuracy on test set...")
        macro_f1, accuracy, per_class_f1 = evaluate_accuracy(
            model, accuracy_loader, is_teacher
        )
        print(f"  Macro F1: {macro_f1:.4f}  |  Accuracy: {accuracy:.4f}")

        # --- Latency (single-sample) ---
        print(f"  Measuring latency ({BENCHMARK_WARMUP_RUNS} warmup + "
              f"{BENCHMARK_TIMED_RUNS} timed)...")
        median_ms, p95_ms, raw_latencies = measure_latency(model, latency_loader, is_teacher)
        print(f"  Latency — median: {median_ms:.2f} ms  |  p95: {p95_ms:.2f} ms")

        # --- Throughput ---
        throughput = measure_throughput(model, throughput_loader, is_teacher)
        print(f"  Throughput: {throughput:.1f} samples/sec")

        # --- Store results ---
        results[model_name] = {
            "model_name": model_name,
            "num_parameters": num_params,
            "size_mb": round(size_mb, 2),
            "macro_f1": round(macro_f1, 6),
            "accuracy": round(accuracy, 6),
            "per_class_f1": per_class_f1,
            "cpu_latency_ms_median": round(median_ms, 3),
            "cpu_latency_ms_p95": round(p95_ms, 3),
            "cpu_throughput_sps": round(throughput, 1),
            "device": "cpu",
        }
        latency_raw[model_name] = [round(x, 4) for x in raw_latencies]
        print()

    # --- Save outputs ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # Flatten per_class_f1 for CSV
    flat_rows = []
    for model_name, metrics in results.items():
        row = {k: v for k, v in metrics.items() if k != "per_class_f1"}
        for cls, score in metrics["per_class_f1"].items():
            row[f"f1_{cls}"] = score
        flat_rows.append(row)

    csv_path = RESULTS_DIR / "benchmark_results.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    latency_raw_path = RESULTS_DIR / "latency_raw.json"
    with open(latency_raw_path, "w") as f:
        json.dump(latency_raw, f)
    print(f"Saved: {latency_raw_path}")

    # --- Summary table ---
    if results:
        print("\n" + "=" * 90)
        print(f"{'Model':<28} {'Params':>10} {'MB':>7} {'F1':>7} "
              f"{'Lat(ms)':>9} {'p95(ms)':>9} {'Tput(sps)':>10}")
        print("-" * 90)
        for model_name, m in results.items():
            print(f"{model_name:<28} {m['num_parameters']:>10,} "
                  f"{m['size_mb']:>7.1f} {m['macro_f1']:>7.4f} "
                  f"{m['cpu_latency_ms_median']:>9.2f} {m['cpu_latency_ms_p95']:>9.2f} "
                  f"{m['cpu_throughput_sps']:>10.1f}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
