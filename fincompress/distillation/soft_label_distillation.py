# RUN ON: Colab/GPU
"""
distillation/soft_label_distillation.py
=========================================
Train the student using vanilla knowledge distillation (KD).

The key insight: instead of training the student to predict hard one-hot labels
(0 or 1), we train it to match the teacher's output probability distribution.
The teacher assigns small but non-zero probabilities to incorrect classes —
these "soft labels" encode relationships between classes that hard labels cannot
express. For example, "slightly negative" text might get teacher outputs like
[0.7, 0.25, 0.05] — the 0.25 neutral probability tells the student that this
sentence is ambiguous, a signal hard labels discard entirely.

Reference: Hinton et al. (2015), "Distilling the Knowledge in a Neural Network"

Run:
    python -m fincompress.distillation.soft_label_distillation
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import wandb

from fincompress.distillation.student_architecture import StudentClassifier

# ============================================================
# HYPERPARAMETERS — all tunable values live here, never inline
# ============================================================
SEED = 42
MAX_SEQ_LEN = 128
BATCH_SIZE_TRAIN = 32
NUM_CLASSES = 3
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

STUDENT_NUM_LAYERS = 4
STUDENT_HIDDEN_SIZE = 384
STUDENT_NUM_HEADS = 6
STUDENT_INTERMEDIATE_SIZE = 1536
STUDENT_DROPOUT = 0.1

# Distillation temperature: controls how "soft" the teacher's distribution is.
# Higher T → softer distribution → more inter-class relationship information.
# Lower T → sharper → approaches hard labels.
# T=4 is a common sweet spot for BERT-class models (Sanh et al., DistilBERT).
KD_TEMPERATURE = 4.0

# KD_ALPHA: weight of soft-label KL loss vs. hard-label CE loss.
# 0.5 balances both signals equally; task performance drives CE,
# representational mimicry drives KL.
KD_ALPHA = 0.5

KD_EPOCHS = 10
KD_LR = 3e-4
WEIGHT_DECAY = 0.01

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEACHER_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "teacher"
STUDENT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "student_vanilla_kd"
LOG_DIR = PROJECT_ROOT / "logs"


# ============================================================
# Dataset
# ============================================================

class SentimentDataset(Dataset):
    """
    Tokenizes text for both teacher and student (they share the same tokenizer).

    Args:
        df: DataFrame with [text, label].
        tokenizer: Shared BERT-vocabulary tokenizer.
        max_len: Maximum sequence length.
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
# KD loss
# ============================================================

def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the combined knowledge distillation loss.

    L_total = alpha * L_KL + (1 - alpha) * L_CE

    WHY SOFT LABELS CARRY MORE INFORMATION THAN HARD LABELS:
    Hard (one-hot) labels only tell the student which class is correct. Soft
    labels encode the teacher's uncertainty and class similarity structure. For
    financial sentiment, "The company reported disappointing but not catastrophic
    results" might get teacher outputs [0.45, 0.50, 0.05] — the near-tie between
    negative and neutral tells the student this is a borderline case. Hard labels
    would just say "neutral" (1) with no nuance. This richer signal guides the
    student toward representations that generalize better to ambiguous inputs.

    WHY T^2 SCALING IS NECESSARY FOR GRADIENT MAGNITUDE CONSISTENCY:
    KL divergence between two softmax distributions (p/T and q/T) scales as 1/T^2
    because each distribution is divided by T before softmax. As T increases, both
    distributions become more uniform, and their KL divergence shrinks by 1/T^2.
    Without the T^2 correction, a high temperature like T=4 would make L_KL 16×
    smaller than at T=1, drowning out the soft-label signal relative to L_CE and
    effectively making KD equivalent to hard-label training. Multiplying by T^2
    restores the gradient magnitude to what it would be at T=1, keeping the balance
    between L_KL and L_CE stable across temperature choices.
    (Hinton et al., 2015 — "Distilling the Knowledge in a Neural Network")

    Args:
        student_logits: [batch, num_classes] raw student outputs.
        teacher_logits: [batch, num_classes] raw teacher outputs (detached).
        hard_labels: [batch] integer class indices.
        temperature: Softening temperature T.
        alpha: Weight on soft-label KL loss (1-alpha on hard-label CE).

    Returns:
        Tuple of (total_loss, ce_loss, kl_loss) — separate components for logging.
    """
    # Hard-label cross-entropy: trains the student to be correct on the task.
    ce_loss = F.cross_entropy(student_logits, hard_labels)

    # Soft-label KL divergence with temperature scaling and T^2 correction.
    # log_softmax(student/T) for numerical stability (avoids log(0)).
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # kl_div expects (log_input, target); reduction='batchmean' averages over batch.
    kl_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature ** 2)  # T^2 scaling restores gradient magnitude — see docstring

    total_loss = alpha * kl_loss + (1.0 - alpha) * ce_loss
    return total_loss, ce_loss, kl_loss


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    student: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Evaluate student model and return CE loss and Macro F1.

    Args:
        student: StudentClassifier in eval mode.
        dataloader: Validation DataLoader.
        device: Target device.

    Returns:
        Tuple of (avg_ce_loss, macro_f1).
    """
    student.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            out = student(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(out["logits"], labels)
            total_loss += loss.item() * labels.size(0)

            preds = out["logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


def get_teacher_f1(teacher: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    Compute teacher Macro F1 for the comparison summary printed at end of training.

    Args:
        teacher: Teacher model in eval mode.
        dataloader: Validation DataLoader.
        device: Target device.

    Returns:
        Teacher Macro F1.
    """
    teacher.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            out = teacher(input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
            preds = out.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    return f1_score(all_labels, all_preds, average="macro")


def save_student_checkpoint(
    student: nn.Module,
    val_f1: float,
    checkpoint_dir: Path,
    hyperparams: dict,
) -> None:
    """
    Save student weights and checkpoint_info.json.

    Args:
        student: Trained StudentClassifier.
        val_f1: Best validation Macro F1.
        checkpoint_dir: Destination directory.
        hyperparams: Flat dict of all hyperparameters used.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), checkpoint_dir / "pytorch_model.bin")

    num_params = sum(p.numel() for p in student.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in student.parameters()) / 1e6

    info = {
        "model_name": "student_vanilla_kd",
        "technique": "vanilla_kd",
        "val_macro_f1": round(val_f1, 6),
        "num_parameters": num_params,
        "size_mb": round(size_mb, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparams,
    }
    with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
        json.dump(info, f, indent=2)


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Vanilla KD pipeline: load teacher → train student → evaluate → checkpoint."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    from transformers import set_seed as hf_set_seed
    hf_set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Validate inputs ---
    for path, script in [
        (DATA_DIR / "train.csv", "fincompress.data.prepare_dataset"),
        (DATA_DIR / "val.csv", "fincompress.data.prepare_dataset"),
        (TEACHER_CHECKPOINT_DIR / "checkpoint_info.json", "fincompress.teacher.train_teacher"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Run 'python -m {script}' first to generate it."
            )

    # --- wandb (optional — disabled automatically if WANDB_API_KEY is not set) ---
    import os as _os
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _wandb_mode = "online" if _os.environ.get("WANDB_API_KEY") else "disabled"
    if _wandb_mode == "disabled":
        print("WandB disabled (no WANDB_API_KEY found). Training metrics logged to CSV only.")
    wandb.init(
        project="fincompress",
        name=f"vanilla_kd_{timestamp}",
        mode=_wandb_mode,
        config={
            "temperature": KD_TEMPERATURE,
            "alpha": KD_ALPHA,
            "epochs": KD_EPOCHS,
            "lr": KD_LR,
            "batch_size": BATCH_SIZE_TRAIN,
            "max_seq_len": MAX_SEQ_LEN,
            "student_num_layers": STUDENT_NUM_LAYERS,
            "student_hidden_size": STUDENT_HIDDEN_SIZE,
            "student_num_heads": STUDENT_NUM_HEADS,
            "student_intermediate_size": STUDENT_INTERMEDIATE_SIZE,
            "student_dropout": STUDENT_DROPOUT,
            "seed": SEED,
        },
    )

    # --- Load teacher ---
    print(f"Loading teacher from {TEACHER_CHECKPOINT_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(TEACHER_CHECKPOINT_DIR / "tokenizer"))
    teacher = AutoModelForSequenceClassification.from_pretrained(str(TEACHER_CHECKPOINT_DIR))
    teacher.to(device)
    # Teacher must stay frozen for the entire run — we're distilling its knowledge,
    # not further fine-tuning it. eval() disables dropout so outputs are deterministic.
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # --- Initialize student ---
    student = StudentClassifier(
        hidden_size=STUDENT_HIDDEN_SIZE,
        num_layers=STUDENT_NUM_LAYERS,
        num_heads=STUDENT_NUM_HEADS,
        intermediate_size=STUDENT_INTERMEDIATE_SIZE,
        dropout=STUDENT_DROPOUT,
        num_classes=NUM_CLASSES,
    )
    student.to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    compression_ratio = teacher_params / student_params
    print(f"Teacher params: {teacher_params:,}")
    print(f"Student params: {student_params:,}  (compression: {compression_ratio:.1f}×)")

    # --- DataLoaders ---
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_val = pd.read_csv(DATA_DIR / "val.csv")

    train_dataset = SentimentDataset(df_train, tokenizer, MAX_SEQ_LEN)
    val_dataset = SentimentDataset(df_val, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_TRAIN * 2, shuffle=False, num_workers=2
    )

    # --- Optimizer ---
    optimizer = AdamW(student.parameters(), lr=KD_LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * KD_EPOCHS
    warmup_steps = int(total_steps * 0.05)  # 5% warmup for student (shorter than teacher)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # --- Logging ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "vanilla_kd_training.csv"
    log_fields = ["epoch", "train_total_loss", "train_ce_loss", "train_kl_loss", "val_f1"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # --- Training loop ---
    best_val_f1 = -1.0
    hyperparams = {
        "temperature": KD_TEMPERATURE, "alpha": KD_ALPHA, "epochs": KD_EPOCHS,
        "lr": KD_LR, "batch_size": BATCH_SIZE_TRAIN, "max_seq_len": MAX_SEQ_LEN,
        "student_num_layers": STUDENT_NUM_LAYERS, "student_hidden_size": STUDENT_HIDDEN_SIZE,
        "student_num_heads": STUDENT_NUM_HEADS, "seed": SEED,
    }

    print(f"\n{'Epoch':>5} {'Total L':>9} {'CE L':>8} {'KL L':>8} {'Val F1':>8}")
    print("-" * 44)

    for epoch in range(1, KD_EPOCHS + 1):
        student.train()
        ep_total, ep_ce, ep_kl = 0.0, 0.0, 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{KD_EPOCHS}", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            # Teacher forward pass — no gradients needed; teacher is frozen.
            with torch.no_grad():
                teacher_out = teacher(input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)
                teacher_logits = teacher_out.logits  # [batch, num_classes]

            # Student forward pass
            student_out = student(input_ids, attention_mask, token_type_ids)
            student_logits = student_out["logits"]

            total_loss, ce_loss, kl_loss = kd_loss(
                student_logits, teacher_logits, labels, KD_TEMPERATURE, KD_ALPHA
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = labels.size(0)
            ep_total += total_loss.item() * bs
            ep_ce += ce_loss.item() * bs
            ep_kl += kl_loss.item() * bs
            n_samples += bs
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        ep_total /= n_samples
        ep_ce /= n_samples
        ep_kl /= n_samples

        _, val_f1 = evaluate(student, val_loader, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_student_checkpoint(student, val_f1, STUDENT_CHECKPOINT_DIR, hyperparams)

        wandb.log({
            "epoch": epoch,
            "total_loss": ep_total,
            "ce_loss": ep_ce,
            "kl_loss": ep_kl,
            "val_f1": val_f1,
        })

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch": epoch,
                "train_total_loss": round(ep_total, 6),
                "train_ce_loss": round(ep_ce, 6),
                "train_kl_loss": round(ep_kl, 6),
                "val_f1": round(val_f1, 6),
            })

        print(f"{epoch:>5} {ep_total:>9.4f} {ep_ce:>8.4f} {ep_kl:>8.4f} {val_f1:>8.4f}")

    wandb.summary["best_val_f1"] = best_val_f1

    # --- Final comparison summary ---
    teacher_f1 = get_teacher_f1(teacher, val_loader, device)
    f1_gap = teacher_f1 - best_val_f1

    print(f"\n{'='*60}")
    print("Vanilla KD Distillation Summary")
    print(f"{'='*60}")
    print(f"  Teacher val F1:       {teacher_f1:.4f}")
    print(f"  Student val F1:       {best_val_f1:.4f}")
    print(f"  F1 gap:               {f1_gap:+.4f}")
    print(f"  Teacher params:       {teacher_params:,}")
    print(f"  Student params:       {student_params:,}")
    print(f"  Compression ratio:    {compression_ratio:.1f}×")

    wandb.finish()


if __name__ == "__main__":
    main()
