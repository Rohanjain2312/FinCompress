# RUN ON: Colab/GPU
"""
distillation/intermediate_distillation.py
==========================================
Train the student using intermediate layer distillation.

Beyond matching the teacher's output logits (vanilla KD), this script also
supervises the student's internal hidden states and attention patterns at
corresponding teacher layers. The key motivation: a student can produce correct
output logits via many different internal computation paths — some of which don't
generalize. By constraining the internal representations to match the teacher's,
we steer the student toward the teacher's proven representational strategy.

Reference: Jiao et al. (2020), "TinyBERT: Distilling BERT for Natural Language Understanding"

Run:
    python -m fincompress.distillation.intermediate_distillation
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
from sklearn.metrics import f1_score
import wandb

from fincompress.distillation.student_architecture import StudentClassifier

# ============================================================
# HYPERPARAMETERS — all tunable values live here, never inline
# ============================================================
SEED = 42
MAX_SEQ_LEN = 128
BATCH_SIZE_TRAIN = 32
NUM_CLASSES = 3

STUDENT_NUM_LAYERS = 4
STUDENT_HIDDEN_SIZE = 384
STUDENT_NUM_HEADS = 6
STUDENT_INTERMEDIATE_SIZE = 1536
STUDENT_DROPOUT = 0.1

KD_TEMPERATURE = 4.0
KD_ALPHA = 0.5
KD_EPOCHS = 10
KD_LR = 3e-4

# Weights for intermediate layer losses — tuned so they add meaningful signal
# without overwhelming the task loss. These were chosen to give roughly equal
# gradient magnitude contribution across the three loss terms.
INTERMEDIATE_LAMBDA_HIDDEN = 0.1
INTERMEDIATE_LAMBDA_ATTN = 0.1

# LAYER_MAP: student layer index → teacher layer index (0-indexed)
# WHY EVENLY SPACED MAPPING:
# The teacher has 12 layers representing a hierarchy of increasingly abstract
# representations: early layers capture lexical/syntactic patterns, middle layers
# capture phrase-level semantics, and final layers encode task-relevant abstractions.
# By mapping student layers 0,1,2,3 to teacher layers 2,5,8,11 (every 3rd), we
# supervise the student at evenly spaced points across this full representational
# hierarchy — ensuring the student learns both low-level and high-level features,
# not just final-layer abstractions that might be shortcut solutions.
LAYER_MAP = {0: 2, 1: 5, 2: 8, 3: 11}

WEIGHT_DECAY = 0.01

# Teacher hidden size (FinBERT = BERT-base = 768)
TEACHER_HIDDEN_SIZE = 768

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEACHER_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "teacher"
STUDENT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "student_intermediate_kd"
LOG_DIR = PROJECT_ROOT / "logs"


# ============================================================
# Dataset
# ============================================================

class SentimentDataset(Dataset):
    """
    Shared tokenization for teacher and student (same vocabulary).

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
# Loss function
# ============================================================

def intermediate_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    student_hidden: list[torch.Tensor],
    teacher_hidden: list[torch.Tensor],
    student_attn: list[torch.Tensor],
    teacher_attn: list[torch.Tensor],
    projections: nn.ModuleList,
    layer_map: dict[int, int],
    temperature: float,
    alpha: float,
    lambda_hidden: float,
    lambda_attn: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the full intermediate distillation loss.

    L_total = alpha * L_KL + (1 - alpha) * L_CE
            + lambda_hidden * L_hidden
            + lambda_attn * L_attn

    WHY ATTENTION TRANSFER WORKS:
    Attention matrices encode which tokens each layer attends to — they are the
    model's internal "routing" decisions that determine what information flows
    forward. Forcing student attention patterns to match teacher patterns at
    corresponding layers constrains the student to develop similar token-relationship
    understanding, not just produce similar final-layer outputs. This prevents the
    student from learning attention shortcuts (e.g. always attending to [CLS]) that
    happen to give correct training predictions but fail to generalize to held-out
    distributions. By supervising intermediate attention, we're distilling the
    teacher's computational strategy, not just its final answers.
    (TinyBERT, Jiao et al. 2020; BERT-PKD, Sun et al. 2019)

    Args:
        student_logits: [batch, num_classes]
        teacher_logits: [batch, num_classes] — must be detached
        hard_labels: [batch] integer class indices
        student_hidden: list of [batch, seq, student_hidden] — one per student layer
        teacher_hidden: list of [batch, seq, teacher_hidden] — one per teacher layer
        student_attn: list of [batch, student_heads, seq, seq]
        teacher_attn: list of [batch, teacher_heads, seq, seq]
        projections: nn.ModuleList of Linear(student_hidden → teacher_hidden)
        layer_map: dict mapping student_layer_idx → teacher_layer_idx
        temperature: KD temperature T
        alpha: KL loss weight
        lambda_hidden: Hidden state MSE weight
        lambda_attn: Attention MSE weight

    Returns:
        Tuple of (total_loss, ce_loss, kl_loss, hidden_loss, attn_loss).
    """
    # --- Task loss (hard labels) ---
    ce_loss = F.cross_entropy(student_logits, hard_labels)

    # --- Soft-label KL loss with T^2 correction (see soft_label_distillation.py) ---
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

    # --- Hidden state MSE ---
    # Projection layers learned here map student's smaller hidden space to teacher's
    # space so the MSE is computed in the same dimension. Projections are LEARNED
    # parameters (not fixed) — they allow the student to have a different basis for
    # its representations while still matching the teacher's content after projection.
    hidden_losses = []
    for student_idx, teacher_idx in layer_map.items():
        proj_hidden = projections[student_idx](student_hidden[student_idx])
        # Detach teacher: we want to match it, not backprop through it.
        target_hidden = teacher_hidden[teacher_idx].detach()
        hidden_losses.append(F.mse_loss(proj_hidden, target_hidden))
    hidden_loss = torch.stack(hidden_losses).mean()

    # --- Attention pattern MSE ---
    # Teacher may have different num_heads than student. We match the attention
    # distribution shapes by comparing the normalized attention weights directly.
    # If head counts differ, we use the student's own heads (no projection needed —
    # what matters is the positional pattern, not the head count).
    attn_losses = []
    for student_idx, teacher_idx in layer_map.items():
        s_attn = student_attn[student_idx]    # [batch, student_heads, seq, seq]
        t_attn = teacher_attn[teacher_idx].detach()  # [batch, teacher_heads, seq, seq]

        # Average over heads to get a single [batch, seq, seq] attention map per model.
        # This allows MSE regardless of head count mismatch between teacher and student.
        s_attn_avg = s_attn.mean(dim=1)  # [batch, seq, seq]
        t_attn_avg = t_attn.mean(dim=1)  # [batch, seq, seq]
        attn_losses.append(F.mse_loss(s_attn_avg, t_attn_avg))
    attn_loss = torch.stack(attn_losses).mean()

    total_loss = (
        alpha * kl_loss
        + (1.0 - alpha) * ce_loss
        + lambda_hidden * hidden_loss
        + lambda_attn * attn_loss
    )
    return total_loss, ce_loss, kl_loss, hidden_loss, attn_loss


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    student: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute Macro F1 on validation set.

    Args:
        student: StudentClassifier.
        dataloader: Validation DataLoader.
        device: Target device.

    Returns:
        Macro F1 score.
    """
    student.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            out = student(input_ids, attention_mask, token_type_ids)
            preds = out["logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    return f1_score(all_labels, all_preds, average="macro")


def save_student_checkpoint(
    student: nn.Module,
    projections: nn.Module,
    val_f1: float,
    checkpoint_dir: Path,
    hyperparams: dict,
) -> None:
    """
    Save student state dict, projection weights, and checkpoint_info.json.

    Args:
        student: Trained StudentClassifier.
        projections: Trained projection layers (nn.ModuleList).
        val_f1: Best validation Macro F1.
        checkpoint_dir: Destination.
        hyperparams: Flat hyperparameter dict.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), checkpoint_dir / "pytorch_model.bin")
    torch.save(projections.state_dict(), checkpoint_dir / "projections.bin")

    num_params = sum(p.numel() for p in student.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in student.parameters()) / 1e6

    info = {
        "model_name": "student_intermediate_kd",
        "technique": "intermediate_kd",
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
    """Intermediate KD pipeline: load teacher → create projections → train → save."""
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
        name=f"intermediate_kd_{timestamp}",
        mode=_wandb_mode,
        config={
            "temperature": KD_TEMPERATURE,
            "alpha": KD_ALPHA,
            "lambda_hidden": INTERMEDIATE_LAMBDA_HIDDEN,
            "lambda_attn": INTERMEDIATE_LAMBDA_ATTN,
            "layer_map": str(LAYER_MAP),
            "epochs": KD_EPOCHS,
            "lr": KD_LR,
            "batch_size": BATCH_SIZE_TRAIN,
            "max_seq_len": MAX_SEQ_LEN,
            "student_num_layers": STUDENT_NUM_LAYERS,
            "student_hidden_size": STUDENT_HIDDEN_SIZE,
            "seed": SEED,
        },
    )

    # --- Load teacher ---
    print(f"Loading teacher from {TEACHER_CHECKPOINT_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(TEACHER_CHECKPOINT_DIR / "tokenizer"))
    teacher = AutoModelForSequenceClassification.from_pretrained(str(TEACHER_CHECKPOINT_DIR))
    teacher.to(device)
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

    # --- Projection layers: student_hidden → teacher_hidden ---
    # One projection per student layer in LAYER_MAP.
    # bias=False because the MSE target (teacher hidden) has zero mean after
    # LayerNorm, so a bias term would just learn to zero itself out.
    projections = nn.ModuleList([
        nn.Linear(STUDENT_HIDDEN_SIZE, TEACHER_HIDDEN_SIZE, bias=False)
        for _ in range(STUDENT_NUM_LAYERS)
    ])
    projections.to(device)

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

    # --- Optimizer: optimize both student and projections jointly ---
    all_params = list(student.parameters()) + list(projections.parameters())
    optimizer = AdamW(all_params, lr=KD_LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * KD_EPOCHS
    warmup_steps = int(total_steps * 0.05)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # --- Logging ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "intermediate_kd_training.csv"
    log_fields = [
        "epoch", "train_total_loss", "train_ce_loss", "train_kl_loss",
        "train_hidden_loss", "train_attn_loss", "val_f1"
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # --- Training state ---
    best_val_f1 = -1.0
    hyperparams = {
        "temperature": KD_TEMPERATURE, "alpha": KD_ALPHA,
        "lambda_hidden": INTERMEDIATE_LAMBDA_HIDDEN,
        "lambda_attn": INTERMEDIATE_LAMBDA_ATTN,
        "layer_map": LAYER_MAP, "epochs": KD_EPOCHS,
        "lr": KD_LR, "batch_size": BATCH_SIZE_TRAIN,
        "max_seq_len": MAX_SEQ_LEN, "seed": SEED,
    }

    print(f"\n{'Epoch':>5} {'Total':>8} {'CE':>8} {'KL':>8} {'Hid':>8} {'Attn':>8} {'Val F1':>8}")
    print("-" * 60)

    for epoch in range(1, KD_EPOCHS + 1):
        student.train()
        projections.train()
        ep_total, ep_ce, ep_kl, ep_hid, ep_attn = 0.0, 0.0, 0.0, 0.0, 0.0
        n_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{KD_EPOCHS}", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            # Teacher forward — frozen, no gradient.
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_hidden_states=True,
                    output_attentions=True,
                )
                # teacher_out.hidden_states: tuple of [batch, seq, 768], includes embedding layer
                # Index 0 = embedding, 1..12 = transformer layer outputs (0-indexed: 1..12)
                # So teacher layer i (0-indexed) is at hidden_states[i+1].
                teacher_hidden_all = teacher_out.hidden_states[1:]   # layers 0..11
                teacher_attn_all = teacher_out.attentions             # layers 0..11
                teacher_logits = teacher_out.logits

            # Student forward
            student_out = student(input_ids, attention_mask, token_type_ids)
            student_logits = student_out["logits"]
            student_hidden_all = student_out["hidden_states"]
            student_attn_all = student_out["attention_weights"]

            total_loss, ce_loss, kl_loss, hidden_loss, attn_loss = intermediate_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                hard_labels=labels,
                student_hidden=student_hidden_all,
                teacher_hidden=list(teacher_hidden_all),
                student_attn=student_attn_all,
                teacher_attn=list(teacher_attn_all),
                projections=projections,
                layer_map=LAYER_MAP,
                temperature=KD_TEMPERATURE,
                alpha=KD_ALPHA,
                lambda_hidden=INTERMEDIATE_LAMBDA_HIDDEN,
                lambda_attn=INTERMEDIATE_LAMBDA_ATTN,
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            bs = labels.size(0)
            ep_total += total_loss.item() * bs
            ep_ce += ce_loss.item() * bs
            ep_kl += kl_loss.item() * bs
            ep_hid += hidden_loss.item() * bs
            ep_attn += attn_loss.item() * bs
            n_samples += bs
            pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        for key in ["ep_total", "ep_ce", "ep_kl", "ep_hid", "ep_attn"]:
            locals()[key]  # keep reference
        ep_total /= n_samples
        ep_ce /= n_samples
        ep_kl /= n_samples
        ep_hid /= n_samples
        ep_attn /= n_samples

        val_f1 = evaluate(student, val_loader, device)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_student_checkpoint(student, projections, val_f1, STUDENT_CHECKPOINT_DIR, hyperparams)

        wandb.log({
            "epoch": epoch,
            "total_loss": ep_total,
            "ce_loss": ep_ce,
            "kl_loss": ep_kl,
            "hidden_loss": ep_hid,
            "attn_loss": ep_attn,
            "val_f1": val_f1,
        })

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch": epoch,
                "train_total_loss": round(ep_total, 6),
                "train_ce_loss": round(ep_ce, 6),
                "train_kl_loss": round(ep_kl, 6),
                "train_hidden_loss": round(ep_hid, 6),
                "train_attn_loss": round(ep_attn, 6),
                "val_f1": round(val_f1, 6),
            })

        print(f"{epoch:>5} {ep_total:>8.4f} {ep_ce:>8.4f} {ep_kl:>8.4f} "
              f"{ep_hid:>8.4f} {ep_attn:>8.4f} {val_f1:>8.4f}")

    wandb.summary["best_val_f1"] = best_val_f1
    print(f"\nBest val Macro F1: {best_val_f1:.4f}")
    print(f"Checkpoint saved to: {STUDENT_CHECKPOINT_DIR}")

    wandb.finish()


if __name__ == "__main__":
    main()
