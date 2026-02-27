# RUN ON: Colab/GPU
"""
quantization/qat.py
====================
Quantization-Aware Training (QAT) — retrain the distilled student with
simulated INT8 quantization so the model learns weights that are robust
to rounding before we do the final conversion.

QAT vs. PTQ conceptual difference (also documented in the training loop below):
  PTQ is a post-hoc approximation; QAT is optimization-time awareness.
  QAT recovers accuracy lost by PTQ at the cost of a short fine-tuning run.

Run:
    python -m fincompress.quantization.qat
"""

import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import wandb

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

QAT_EPOCHS = 3
QAT_LR = 1e-5
WEIGHT_DECAY = 0.01

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "student_intermediate_kd"
QAT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "student_qat"
TEACHER_TOKENIZER_DIR = PROJECT_ROOT / "checkpoints" / "teacher" / "tokenizer"
LOG_DIR = PROJECT_ROOT / "logs"


# ============================================================
# Dataset
# ============================================================

class SentimentDataset(Dataset):
    """
    Tokenizes financial sentiment examples for QAT fine-tuning.

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
# Helpers
# ============================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate model on validation set.

    Args:
        model: Model in eval mode.
        dataloader: Validation DataLoader.
        device: Target device.

    Returns:
        Tuple of (avg_ce_loss, macro_f1).
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            out = model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(out["logits"], labels)
            total_loss += loss.item() * labels.size(0)

            preds = out["logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


def get_model_size_mb(model: nn.Module) -> float:
    """
    Estimate model size from parameters and buffers (covers quantized models).

    Args:
        model: Any nn.Module.

    Returns:
        Size in MB.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / 1e6


# ============================================================
# QAT training loop
# ============================================================

def train_qat(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Fine-tune the student with fake quantization nodes active.

    # CONCEPTUAL DIFFERENCE BETWEEN PTQ AND QAT:
    #
    # PTQ (Post-Training Quantization): After training completes, we approximate
    # FP32 weights as INT8. The model was never asked to be robust to rounding —
    # it just has to tolerate it. Weights sit at FP32 optima that may be far from
    # any INT8-representable grid point, so rounding causes accuracy loss.
    # PTQ is fast but yields sub-optimal accuracy because the optimization and the
    # quantization are decoupled.
    #
    # QAT (Quantization-Aware Training): We insert FakeQuantize nodes that
    # simulate INT8 rounding during the forward pass, but the backward pass uses
    # the straight-through estimator (STE) — gradients pass through as if
    # quantization didn't happen. This means the optimizer explicitly minimizes
    # loss SUBJECT TO quantization noise: it finds weight values that land near
    # INT8 grid points while still being task-accurate. The result is a model
    # whose FP32 weights are "quantization-friendly" before we do final INT8
    # conversion, recovering much of the accuracy that PTQ loses.
    #
    # Trade-off: QAT requires a training run (3 epochs here); PTQ does not.
    # When training data is available, QAT is almost always worth the cost.

    Args:
        model: StudentClassifier with fake-quant nodes (from prepare_qat).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Target device (CUDA preferred for speed).

    Returns:
        Best val Macro F1 achieved during QAT.
    """
    optimizer = AdamW(model.parameters(), lr=QAT_LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * QAT_EPOCHS
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "qat_training.csv"
    log_fields = ["epoch", "train_loss", "val_f1"]
    with open(log_path, "w", newline="") as f:
        import csv
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    best_val_f1 = -1.0

    print(f"\n{'Epoch':>5} {'Train Loss':>11} {'Val F1':>8}")
    print("-" * 28)

    for epoch in range(1, QAT_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"QAT Epoch {epoch}/{QAT_EPOCHS}", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            out = model(input_ids, attention_mask, token_type_ids)
            # Pure CE loss during QAT — no distillation needed.
            # The fake-quant noise serves as the implicit regularizer here.
            loss = F.cross_entropy(out["logits"], labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_samples
        _, val_f1 = evaluate(model, val_loader, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

        wandb.log({"epoch": epoch, "qat_train_loss": avg_loss, "qat_val_f1": val_f1})

        with open(log_path, "a", newline="") as f:
            import csv
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch": epoch,
                "train_loss": round(avg_loss, 6),
                "val_f1": round(val_f1, 6),
            })

        print(f"{epoch:>5} {avg_loss:>11.4f} {val_f1:>8.4f}")

    return best_val_f1


# ============================================================
# Main
# ============================================================

def main() -> None:
    """QAT pipeline: load → prepare_qat → train → convert → save."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Validate inputs ---
    for path, script in [
        (SOURCE_CHECKPOINT_DIR / "pytorch_model.bin",
         "fincompress.distillation.intermediate_distillation"),
        (DATA_DIR / "train.csv", "fincompress.data.prepare_dataset"),
        (DATA_DIR / "val.csv", "fincompress.data.prepare_dataset"),
        (TEACHER_TOKENIZER_DIR, "fincompress.teacher.train_teacher"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Run 'python -m {script}' first to generate it."
            )

    # --- wandb ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project="fincompress",
        name=f"qat_{timestamp}",
        config={
            "epochs": QAT_EPOCHS,
            "lr": QAT_LR,
            "batch_size": BATCH_SIZE_TRAIN,
            "max_seq_len": MAX_SEQ_LEN,
            "seed": SEED,
        },
    )

    # --- Load student ---
    print(f"Loading student from {SOURCE_CHECKPOINT_DIR}...")
    model = StudentClassifier(
        hidden_size=STUDENT_HIDDEN_SIZE,
        num_layers=STUDENT_NUM_LAYERS,
        num_heads=STUDENT_NUM_HEADS,
        intermediate_size=STUDENT_INTERMEDIATE_SIZE,
        dropout=STUDENT_DROPOUT,
        num_classes=NUM_CLASSES,
    )
    state_dict = torch.load(
        SOURCE_CHECKPOINT_DIR / "pytorch_model.bin",
        map_location="cpu",
    )
    model.load_state_dict(state_dict)

    original_size_mb = get_model_size_mb(model)

    # QAT must be prepared on CPU because fake-quant ops in PyTorch require CPU
    # for the prepare step, even if training will run on GPU after that.
    model.cpu()
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

    # Keep embeddings and classifier head in FP32 (same reasons as PTQ).
    model.token_embedding.qconfig = None
    model.position_embedding.qconfig = None
    model.segment_embedding.qconfig = None
    model.classifier.qconfig = None

    # prepare_qat inserts FakeQuantize nodes that simulate INT8 during forward
    # but use straight-through estimator during backward.
    torch.quantization.prepare_qat(model, inplace=True)

    model.to(device)

    # --- DataLoaders ---
    tokenizer = AutoTokenizer.from_pretrained(str(TEACHER_TOKENIZER_DIR))
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_val = pd.read_csv(DATA_DIR / "val.csv")

    train_loader = DataLoader(
        SentimentDataset(df_train, tokenizer, MAX_SEQ_LEN),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        SentimentDataset(df_val, tokenizer, MAX_SEQ_LEN),
        batch_size=BATCH_SIZE_TRAIN * 2,
        shuffle=False,
        num_workers=2,
    )

    # --- QAT training ---
    print(f"Starting QAT for {QAT_EPOCHS} epochs...")
    best_val_f1 = train_qat(model, train_loader, val_loader, device)

    # --- Convert to INT8 ---
    # After QAT, the FakeQuantize nodes are replaced with real INT8 quantized ops.
    model.cpu()
    model.eval()
    torch.quantization.convert(model, inplace=True)

    quant_size_mb = get_model_size_mb(model)
    compression_ratio = original_size_mb / quant_size_mb

    print(f"\nQAT Results:")
    print(f"  FP32 size:         {original_size_mb:.1f} MB")
    print(f"  INT8 size:         {quant_size_mb:.1f} MB")
    print(f"  Compression ratio: {compression_ratio:.1f}×")
    print(f"  Best val F1:       {best_val_f1:.4f}")

    # --- Save ---
    QAT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), QAT_CHECKPOINT_DIR / "pytorch_model.bin")

    info = {
        "model_name": "student_qat",
        "technique": "qat",
        "val_macro_f1": round(best_val_f1, 6),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "size_mb": round(quant_size_mb, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": {
            "qat_epochs": QAT_EPOCHS,
            "qat_lr": QAT_LR,
            "batch_size": BATCH_SIZE_TRAIN,
            "max_seq_len": MAX_SEQ_LEN,
            "backend": "fbgemm",
            "seed": SEED,
        },
    }
    with open(QAT_CHECKPOINT_DIR / "checkpoint_info.json", "w") as f:
        json.dump(info, f, indent=2)

    wandb.summary["best_val_f1"] = best_val_f1
    print(f"\nCheckpoint saved to: {QAT_CHECKPOINT_DIR}")

    wandb.finish()


if __name__ == "__main__":
    main()
