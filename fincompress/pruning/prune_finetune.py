# RUN ON: Colab/GPU
"""
pruning/prune_finetune.py
===========================
Iterative structured pruning on the teacher model.

Each round:
1. Rank all attention heads by importance (ascending).
2. Prune the bottom PRUNING_HEAD_FRACTION_PER_ROUND fraction globally.
3. Fine-tune for PRUNING_RECOVERY_EPOCHS to recover lost accuracy.
4. Repeat for PRUNING_MAX_ROUNDS rounds.

Save checkpoints at 30% and 50% cumulative head sparsity.
All rounds logged to results/pruning_curve.csv.

Run:
    python -m fincompress.pruning.prune_finetune
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

from fincompress.pruning.structured_pruning import (
    compute_head_importance,
    prune_heads,
    get_pruning_summary,
)

# ============================================================
# HYPERPARAMETERS — all tunable values live here, never inline
# ============================================================
SEED = 42
MAX_SEQ_LEN = 128
BATCH_SIZE_TRAIN = 32
NUM_CLASSES = 3

PRUNING_HEAD_FRACTION_PER_ROUND = 0.10
PRUNING_MAX_ROUNDS = 5
PRUNING_RECOVERY_EPOCHS = 3
PRUNING_RECOVERY_LR = 1e-5

WEIGHT_DECAY = 0.01

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEACHER_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "teacher"
PRUNED_30_DIR = PROJECT_ROOT / "checkpoints" / "pruned_teacher_30pct"
PRUNED_50_DIR = PROJECT_ROOT / "checkpoints" / "pruned_teacher_50pct"
RESULTS_DIR = PROJECT_ROOT / "results"


# ============================================================
# Dataset
# ============================================================

class SentimentDataset(Dataset):
    """
    Standard tokenization wrapper for the teacher model.

    Args:
        df: DataFrame with [text, label].
        tokenizer: HuggingFace tokenizer.
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
# Evaluation and helpers
# ============================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute Macro F1 on validation set.

    Args:
        model: Model to evaluate.
        dataloader: Validation DataLoader.
        device: Target device.

    Returns:
        Macro F1 score.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"]

            out = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
            preds = out.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return f1_score(all_labels, all_preds, average="macro")


def get_teacher_head_importance(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: int = 50,
) -> torch.Tensor:
    """
    Compute per-head importance for the HuggingFace teacher model (BERT-style).

    Uses attention weight entropy as the importance proxy (consistent with
    StudentClassifier's compute_head_importance). Lower entropy = more focused
    attention = higher importance.

    Args:
        model: Teacher model (AutoModelForSequenceClassification).
        dataloader: DataLoader for calibration.
        device: Target device.
        num_batches: Number of batches to average over.

    Returns:
        Importance tensor [num_layers, num_heads], float32.
    """
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads

    importance = torch.zeros(num_layers, num_heads, device=device)
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            out = model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=True,
            )
            # out.attentions: tuple of [batch, num_heads, seq, seq], one per layer
            seq_len = input_ids.size(1)
            eps = 1e-9
            max_entropy = torch.log(torch.tensor(seq_len, dtype=torch.float, device=device))

            for layer_idx, attn_weights in enumerate(out.attentions):
                entropy = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1).mean(dim=-1)
                # entropy: [batch, num_heads]
                normalized_entropy = entropy / (max_entropy + eps)
                head_imp = (1.0 - normalized_entropy).mean(dim=0)  # [num_heads]
                importance[layer_idx] += head_imp

            count += 1

    importance /= count
    return importance.cpu()


def prune_teacher_heads(
    model: nn.Module,
    heads_to_prune: dict[int, list[int]],
) -> nn.Module:
    """
    Zero out Q, K, V, and output weights for specified teacher heads.

    The teacher uses BERT's BertAttention layout: query/key/value are separate
    Linear modules inside model.bert.encoder.layer[i].attention.self.

    Args:
        model: Teacher model.
        heads_to_prune: Dict of layer_idx → list of head indices.

    Returns:
        Modified model.
    """
    config = model.config
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    for layer_idx, head_indices in heads_to_prune.items():
        attn_self = model.bert.encoder.layer[layer_idx].attention.self
        attn_output = model.bert.encoder.layer[layer_idx].attention.output

        for head_idx in head_indices:
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim

            with torch.no_grad():
                attn_self.query.weight.data[start:end, :] = 0.0
                attn_self.key.weight.data[start:end, :] = 0.0
                attn_self.value.weight.data[start:end, :] = 0.0

                if attn_self.query.bias is not None:
                    attn_self.query.bias.data[start:end] = 0.0
                    attn_self.key.bias.data[start:end] = 0.0
                    attn_self.value.bias.data[start:end] = 0.0

                # Zero corresponding output projection columns
                attn_output.dense.weight.data[:, start:end] = 0.0

    return model


def get_teacher_sparsity(model: nn.Module) -> tuple[float, int]:
    """
    Compute overall sparsity percentage and count of non-zero parameters.

    Args:
        model: Teacher model.

    Returns:
        Tuple of (sparsity_pct, total_params).
    """
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum((p != 0).sum().item() for p in model.parameters())
    sparsity_pct = 100.0 * (1.0 - nonzero / total)
    return sparsity_pct, total


def finetune_recovery(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Fine-tune the pruned model for PRUNING_RECOVERY_EPOCHS to recover accuracy.

    Args:
        model: Pruned teacher model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: Target device.

    Returns:
        Best val Macro F1 achieved during recovery.
    """
    # A very small learning rate is used here — we're nudging weights to adapt
    # around the zeroed-out heads, not relearning from scratch. A large LR
    # would destabilize the intact heads that are already well-trained.
    optimizer = AdamW(model.parameters(), lr=PRUNING_RECOVERY_LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * PRUNING_RECOVERY_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    best_f1 = -1.0

    for epoch in range(1, PRUNING_RECOVERY_EPOCHS + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"  Recovery epoch {epoch}/{PRUNING_RECOVERY_EPOCHS}", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            out = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
            loss = F.cross_entropy(out.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        val_f1 = evaluate(model, val_loader, device)
        if val_f1 > best_f1:
            best_f1 = val_f1

    return best_f1


def save_pruned_teacher(
    model: nn.Module,
    tokenizer,
    val_f1: float,
    sparsity_pct: float,
    checkpoint_dir: Path,
    technique: str,
) -> None:
    """
    Save pruned teacher and checkpoint_info.json.

    Args:
        model: Pruned teacher.
        tokenizer: HuggingFace tokenizer.
        val_f1: Best validation Macro F1 after pruning and recovery.
        sparsity_pct: Cumulative head sparsity percentage.
        checkpoint_dir: Destination directory.
        technique: Technique tag (e.g. "pruned_30" or "pruned_50").
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir / "tokenizer"))

    num_params = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    info = {
        "model_name": technique,
        "technique": technique,
        "val_macro_f1": round(val_f1, 6),
        "num_parameters": num_params,
        "size_mb": round(size_mb, 2),
        "sparsity_pct": round(sparsity_pct, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hyperparameters": {
            "head_fraction_per_round": PRUNING_HEAD_FRACTION_PER_ROUND,
            "max_rounds": PRUNING_MAX_ROUNDS,
            "recovery_epochs": PRUNING_RECOVERY_EPOCHS,
            "recovery_lr": PRUNING_RECOVERY_LR,
            "seed": SEED,
        },
    }
    with open(checkpoint_dir / "checkpoint_info.json", "w") as f:
        json.dump(info, f, indent=2)


# ============================================================
# Main pruning loop
# ============================================================

def main() -> None:
    """Iterative prune-and-recover loop on the teacher model."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Validate inputs ---
    for path, script in [
        (TEACHER_CHECKPOINT_DIR / "checkpoint_info.json", "fincompress.teacher.train_teacher"),
        (DATA_DIR / "train.csv", "fincompress.data.prepare_dataset"),
        (DATA_DIR / "val.csv", "fincompress.data.prepare_dataset"),
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
        name=f"pruning_{timestamp}",
        mode=_wandb_mode,
        config={
            "head_fraction_per_round": PRUNING_HEAD_FRACTION_PER_ROUND,
            "max_rounds": PRUNING_MAX_ROUNDS,
            "recovery_epochs": PRUNING_RECOVERY_EPOCHS,
            "recovery_lr": PRUNING_RECOVERY_LR,
            "seed": SEED,
        },
    )

    # --- Load teacher ---
    print(f"Loading teacher from {TEACHER_CHECKPOINT_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(TEACHER_CHECKPOINT_DIR / "tokenizer"))
    model = AutoModelForSequenceClassification.from_pretrained(str(TEACHER_CHECKPOINT_DIR))
    model.to(device)

    config = model.config
    total_heads = config.num_hidden_layers * config.num_attention_heads

    # --- DataLoaders ---
    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_val = pd.read_csv(DATA_DIR / "val.csv")

    train_loader = DataLoader(
        SentimentDataset(df_train, tokenizer, MAX_SEQ_LEN),
        batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        SentimentDataset(df_val, tokenizer, MAX_SEQ_LEN),
        batch_size=BATCH_SIZE_TRAIN * 2, shuffle=False, num_workers=2,
    )

    # --- Results CSV ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = RESULTS_DIR / "pruning_curve.csv"
    curve_fields = ["round", "heads_pruned_pct", "val_f1", "model_size_mb"]
    with open(curve_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=curve_fields).writeheader()

    # --- Baseline ---
    baseline_f1 = evaluate(model, val_loader, device)
    print(f"Baseline val Macro F1: {baseline_f1:.4f}")

    prev_f1 = baseline_f1
    cumulative_pruned = 0
    saved_30 = False
    saved_50 = False

    for round_num in range(1, PRUNING_MAX_ROUNDS + 1):
        print(f"\n{'='*60}")
        print(f"Pruning Round {round_num}/{PRUNING_MAX_ROUNDS}")
        print(f"{'='*60}")

        # --- Compute importance ---
        print("  Computing head importance...")
        importance = get_teacher_head_importance(model, val_loader, device, num_batches=50)
        # importance: [num_layers, num_heads]

        # --- Globally rank all heads and select bottom fraction to prune ---
        n_heads_to_prune = max(1, int(total_heads * PRUNING_HEAD_FRACTION_PER_ROUND))
        # Remaining heads after this round
        remaining_heads = total_heads - cumulative_pruned - n_heads_to_prune

        # Flatten importance, get indices of bottom n_heads_to_prune
        flat_importance = importance.view(-1)
        # Exclude already-zeroed heads from selection by setting their importance to inf
        # (they'll never be re-selected — they're already pruned)
        sorted_indices = flat_importance.argsort()

        heads_to_prune: dict[int, list[int]] = {}
        selected = 0
        for flat_idx in sorted_indices.tolist():
            if selected >= n_heads_to_prune:
                break
            layer_idx = flat_idx // config.num_attention_heads
            head_idx = flat_idx % config.num_attention_heads
            if layer_idx not in heads_to_prune:
                heads_to_prune[layer_idx] = []
            heads_to_prune[layer_idx].append(head_idx)
            selected += 1

        cumulative_pruned += n_heads_to_prune
        sparsity_pct = 100.0 * cumulative_pruned / total_heads

        print(f"  Pruning {n_heads_to_prune} heads "
              f"(cumulative: {cumulative_pruned}/{total_heads} = {sparsity_pct:.1f}%)")

        # --- Prune ---
        model = prune_teacher_heads(model, heads_to_prune)

        # --- Recovery fine-tuning ---
        print(f"  Fine-tuning for {PRUNING_RECOVERY_EPOCHS} recovery epochs...")
        val_f1 = finetune_recovery(model, train_loader, val_loader, device)

        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
        delta_f1 = val_f1 - prev_f1

        print(f"  Round {round_num} | Heads pruned: {n_heads_to_prune} "
              f"| Sparsity: {sparsity_pct:.1f}% "
              f"| Val F1: {val_f1:.4f} "
              f"| ΔF1: {delta_f1:+.4f}")

        # --- Log ---
        wandb.log({
            "round": round_num,
            "val_f1": val_f1,
            "heads_remaining": total_heads - cumulative_pruned,
            "cumulative_sparsity_pct": sparsity_pct,
            "f1_delta": delta_f1,
        })

        with open(curve_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=curve_fields).writerow({
                "round": round_num,
                "heads_pruned_pct": round(sparsity_pct, 2),
                "val_f1": round(val_f1, 6),
                "model_size_mb": round(size_mb, 2),
            })

        # --- Save milestones ---
        if not saved_30 and sparsity_pct >= 28.0:
            print(f"  → Saving pruned_teacher_30pct checkpoint ({sparsity_pct:.1f}% sparsity)")
            save_pruned_teacher(model, tokenizer, val_f1, sparsity_pct, PRUNED_30_DIR, "pruned_30")
            saved_30 = True

        if not saved_50 and sparsity_pct >= 48.0:
            print(f"  → Saving pruned_teacher_50pct checkpoint ({sparsity_pct:.1f}% sparsity)")
            save_pruned_teacher(model, tokenizer, val_f1, sparsity_pct, PRUNED_50_DIR, "pruned_50")
            saved_50 = True

        prev_f1 = val_f1

    wandb.summary["baseline_f1"] = baseline_f1
    wandb.summary["final_f1"] = prev_f1
    wandb.summary["final_sparsity_pct"] = sparsity_pct

    print(f"\nPruning complete. Results saved to: {curve_path}")

    if not saved_30:
        print("WARNING: 30% sparsity milestone not reached. "
              "Increase PRUNING_MAX_ROUNDS or PRUNING_HEAD_FRACTION_PER_ROUND.")
    if not saved_50:
        print("WARNING: 50% sparsity milestone not reached. "
              "Increase PRUNING_MAX_ROUNDS or PRUNING_HEAD_FRACTION_PER_ROUND.")

    wandb.finish()


if __name__ == "__main__":
    main()
