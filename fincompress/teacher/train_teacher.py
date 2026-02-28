# RUN ON: Colab/GPU
"""
teacher/train_teacher.py
=========================
Fine-tune ProsusAI/finbert on the combined financial sentiment dataset.

Implements a clean, manual PyTorch training loop — no HuggingFace Trainer.
Saves the best checkpoint by val Macro F1 with early stopping.

Why start with FinBERT rather than BERT-base?
  FinBERT was pre-trained on financial news, research reports, and earnings calls.
  It already understands domain vocabulary like "EPS beat", "margin compression",
  and "short squeeze" as semantic units rather than OOV tokens. Starting with a
  domain-aware teacher gives the student a richer target to distill.

Run:
    python -m fincompress.teacher.train_teacher
"""

import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
import wandb

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# ============================================================
# HYPERPARAMETERS — all tunable values live here, never inline
# ============================================================
SEED = 42
MAX_SEQ_LEN = 128
BATCH_SIZE_TRAIN = 32
NUM_CLASSES = 3
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

TEACHER_MODEL_NAME = "ProsusAI/finbert"
TEACHER_EPOCHS = 5
TEACHER_LR = 2e-5
TEACHER_WARMUP_RATIO = 0.1
TEACHER_EARLY_STOPPING_PATIENCE = 3

# AdamW weight decay applied to all parameters except bias and LayerNorm.
# Bias terms and LayerNorm scale/shift parameters are small and should not be
# penalized — regularizing them hurts performance without meaningful benefit.
WEIGHT_DECAY = 0.01

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "teacher"
LOG_DIR = PROJECT_ROOT / "logs"


# ============================================================
# Dataset
# ============================================================

class SentimentDataset(Dataset):
    """
    Tokenizes text and returns padded input tensors for the teacher model.

    Args:
        df: DataFrame with columns [text, label].
        tokenizer: HuggingFace tokenizer (must match teacher model).
        max_len: Maximum sequence length (longer sequences are truncated).
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int) -> None:
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_len, dtype=torch.long)).squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# Training helpers
# ============================================================

def get_optimizer_grouped_parameters(model: nn.Module, weight_decay: float) -> list[dict]:
    """
    Split parameters into two groups: those with and without weight decay.

    Bias terms and LayerNorm parameters should NOT receive weight decay because:
      - LayerNorm weight/bias are scale/shift parameters that should adapt freely
      - Bias terms in attention and FFN layers are small and penalty is ineffective

    Args:
        model: The model whose parameters to group.
        weight_decay: Decay value for the regular group.

    Returns:
        List of two dicts suitable for AdamW's param_groups argument.
    """
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    return [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    """
    Evaluate model on a DataLoader and compute loss, Macro F1, and accuracy.

    Args:
        model: Model to evaluate (set to eval() before calling).
        dataloader: Validation DataLoader.
        device: torch device.
        criterion: Loss function.

    Returns:
        Tuple of (val_loss, macro_f1, accuracy).
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            logits = outputs.logits

            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, macro_f1, accuracy


def save_checkpoint(
    model: nn.Module,
    tokenizer,
    val_f1: float,
    checkpoint_dir: Path,
    hyperparams: dict,
) -> None:
    """
    Save model weights, tokenizer, and checkpoint_info.json to checkpoint_dir.

    Args:
        model: Trained model (HuggingFace AutoModelForSequenceClassification).
        tokenizer: HuggingFace tokenizer.
        val_f1: Best validation Macro F1 achieved.
        checkpoint_dir: Directory to save into.
        hyperparams: Dict of hyperparameter values used during training.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model and tokenizer in HuggingFace format so downstream scripts
    # can load with AutoModel/AutoTokenizer without knowing the class name.
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir / "tokenizer"))

    num_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    info = {
        "model_name": "teacher",
        "technique": "teacher",
        "val_macro_f1": round(val_f1, 6),
        "num_parameters": num_params,
        "size_mb": round(size_mb, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": hyperparams,
    }
    with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
        json.dump(info, f, indent=2)


# ============================================================
# Main training loop
# ============================================================

def main() -> None:
    """Full teacher fine-tuning pipeline: load → train → evaluate → checkpoint."""
    # --- Reproducibility ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    from transformers import set_seed as hf_set_seed
    hf_set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Validate inputs ---
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {train_path}\n"
            "Run 'python -m fincompress.data.prepare_dataset' first to generate it."
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {val_path}\n"
            "Run 'python -m fincompress.data.prepare_dataset' first to generate it."
        )

    # --- wandb (optional — disabled automatically if WANDB_API_KEY is not set) ---
    import os as _os
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _wandb_mode = "online" if _os.environ.get("WANDB_API_KEY") else "disabled"
    wandb.init(
        project="fincompress",
        name=f"teacher_{timestamp}",
        mode=_wandb_mode,
        config={
            "model_name": TEACHER_MODEL_NAME,
            "epochs": TEACHER_EPOCHS,
            "lr": TEACHER_LR,
            "warmup_ratio": TEACHER_WARMUP_RATIO,
            "early_stopping_patience": TEACHER_EARLY_STOPPING_PATIENCE,
            "batch_size": BATCH_SIZE_TRAIN,
            "max_seq_len": MAX_SEQ_LEN,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
    )
    if _wandb_mode == "disabled":
        print("WandB disabled (no WANDB_API_KEY found). Training metrics logged to CSV only.")

    # --- Load model and tokenizer ---
    print(f"Loading {TEACHER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        TEACHER_MODEL_NAME, num_labels=NUM_CLASSES
    )
    model.to(device)

    # --- DataLoaders ---
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    train_dataset = SentimentDataset(df_train, tokenizer, MAX_SEQ_LEN)
    val_dataset = SentimentDataset(df_val, tokenizer, MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE_TRAIN * 2, shuffle=False, num_workers=2
    )

    # --- Optimizer and scheduler ---
    param_groups = get_optimizer_grouped_parameters(model, WEIGHT_DECAY)
    optimizer = AdamW(param_groups, lr=TEACHER_LR)

    total_steps = len(train_loader) * TEACHER_EPOCHS
    warmup_steps = int(total_steps * TEACHER_WARMUP_RATIO)
    # Linear warmup then linear decay: standard for BERT fine-tuning.
    # Warmup prevents large gradient updates at initialization when parameters
    # are random; linear decay ensures convergence rather than oscillation.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()

    # --- Logging setup ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "teacher_training.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "val_f1", "val_accuracy", "learning_rate"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # --- Training state ---
    best_val_f1 = -1.0
    patience_counter = 0
    hyperparams = {
        "model_name": TEACHER_MODEL_NAME,
        "epochs": TEACHER_EPOCHS,
        "lr": TEACHER_LR,
        "warmup_ratio": TEACHER_WARMUP_RATIO,
        "early_stopping_patience": TEACHER_EARLY_STOPPING_PATIENCE,
        "batch_size": BATCH_SIZE_TRAIN,
        "max_seq_len": MAX_SEQ_LEN,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
    }

    print(f"\nTraining for up to {TEACHER_EPOCHS} epochs "
          f"(early stop patience={TEACHER_EARLY_STOPPING_PATIENCE})")
    print(f"Target val Macro F1 >= 0.87 before distillation\n")

    print(f"{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>9} {'Val F1':>8} "
          f"{'LR':>10} {'Saved?':>7}")
    print("-" * 58)

    for epoch in range(1, TEACHER_EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{TEACHER_EPOCHS}", leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            loss = criterion(outputs.logits, labels)
            loss.backward()

            # Gradient clipping: prevents exploding gradients in BERT fine-tuning,
            # especially in early epochs when the classification head is random.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)

        # --- Evaluate ---
        val_loss, val_f1, val_accuracy = evaluate(model, val_loader, device, criterion)
        current_lr = scheduler.get_last_lr()[0]

        # --- Checkpoint ---
        saved = False
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_checkpoint(model, tokenizer, val_f1, CHECKPOINT_DIR, hyperparams)
            saved = True
        else:
            patience_counter += 1

        # --- Logging ---
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr,
        })

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_f1": round(val_f1, 6),
                "val_accuracy": round(val_accuracy, 6),
                "learning_rate": f"{current_lr:.2e}",
            })

        print(f"{epoch:>5} {train_loss:>11.4f} {val_loss:>9.4f} {val_f1:>8.4f} "
              f"{current_lr:>10.2e} {'✓' if saved else '':>7}")

        # --- Early stopping ---
        if patience_counter >= TEACHER_EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping: val F1 did not improve for "
                  f"{TEACHER_EARLY_STOPPING_PATIENCE} consecutive epochs.")
            break

    wandb.summary["best_val_f1"] = best_val_f1

    print(f"\n{'='*58}")
    print(f"Training complete. Best val Macro F1: {best_val_f1:.4f}")
    if best_val_f1 < 0.87:
        print("WARNING: val F1 < 0.87 — consider longer training before distillation.")
    print(f"Checkpoint saved to: {CHECKPOINT_DIR}")

    wandb.finish()


if __name__ == "__main__":
    main()
