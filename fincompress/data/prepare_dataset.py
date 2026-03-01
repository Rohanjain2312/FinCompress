# RUN ON: Local/CPU
"""
data/prepare_dataset.py
=======================
Download, merge, normalize, clean, and split financial sentiment datasets
into train/val/test CSVs. This is the single source of truth for all data
in the FinCompress project — every downstream script reads from these CSVs.

Datasets used:
  1. FinancialPhraseBank (sentences_allagree config) — expert-annotated
     financial news sentences. Labels are already 0/1/2.
  2. FiQA-2018 (pauri32/fiqa-2018) — financial Q&A with continuous sentiment
     scores normalized here to 3 discrete classes.

Run:
    python -m fincompress.data.prepare_dataset
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ============================================================
# HYPERPARAMETERS — all tunable values live here, never inline
# ============================================================
SEED = 42
NUM_CLASSES = 3
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# FiQA continuous score → discrete label thresholds
# Chosen to match FinancialPhraseBank's ternary labeling convention.
FIQA_POS_THRESHOLD = 0.1
FIQA_NEG_THRESHOLD = -0.1

# Stratified split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # implicitly 1 - TRAIN - VAL


# ============================================================
# Path helpers
# ============================================================
# Go up one level from fincompress/data/ → fincompress/ (the package root)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# Dataset loaders
# ============================================================

def load_financial_phrasebank() -> pd.DataFrame:
    """
    Load the FinancialPhraseBank dataset (sentences_allagree split).

    The 'allagree' split only includes sentences where ALL annotators
    agreed on the label — giving us the highest-quality subset. Labels
    are already encoded as {0: negative, 1: neutral, 2: positive}.

    Returns:
        DataFrame with columns [text, label].
    """
    try:
        # Use takala/financial_phrasebank — a Parquet-based mirror of the original
        # dataset. The original financial_phrasebank uses a loading script that
        # is no longer supported by datasets >= 2.20. This mirror is identical:
        # same sentences_allagree split, same sentence/label columns.
        dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree")
    except Exception as e:
        print(f"ERROR: Failed to download financial_phrasebank: {e}")
        print("Check your internet connection or HuggingFace Hub status.")
        raise

    # HuggingFace returns a DatasetDict; 'train' split contains all data here.
    split = dataset["train"]
    df = pd.DataFrame({"text": split["sentence"], "label": split["label"]})
    print(f"  FinancialPhraseBank loaded: {len(df)} samples")
    return df


def load_fiqa_sentiment() -> pd.DataFrame:
    """
    Load the FiQA-2018 sentiment dataset and normalize continuous scores
    to 3 discrete classes matching FinancialPhraseBank's convention.

    The raw FiQA scores are floats in roughly [-1, 1]. We discretize with
    symmetric thresholds: score > 0.1 → positive, score < -0.1 → negative,
    else → neutral.

    Returns:
        DataFrame with columns [text, label].
    """
    try:
        # trust_remote_code removed — no longer supported in datasets >= 2.20.
        dataset = load_dataset("pauri32/fiqa-2018")
    except Exception as e:
        print(f"ERROR: Failed to download pauri32/fiqa-2018: {e}")
        print("Check your internet connection or HuggingFace Hub status.")
        raise

    records = []
    # FiQA may have multiple splits; collect all available.
    for split_name in dataset.keys():
        split = dataset[split_name]
        for item in split:
            score = float(item.get("sentiment_score", 0.0))
            text = item.get("sentence", "") or item.get("text", "")
            if not text:
                continue
            # Discretize: thresholds are symmetric around 0 to avoid label bias.
            if score > FIQA_POS_THRESHOLD:
                label = 2  # positive
            elif score < FIQA_NEG_THRESHOLD:
                label = 0  # negative
            else:
                label = 1  # neutral
            records.append({"text": text, "label": label})

    df = pd.DataFrame(records)
    print(f"  FiQA-2018 loaded: {len(df)} samples (after score normalization)")
    return df


# ============================================================
# Data processing
# ============================================================

def merge_and_clean(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate multiple DataFrames, drop exact text duplicates, and
    validate that labels are in the expected range.

    Args:
        dfs: List of DataFrames each with [text, label] columns.

    Returns:
        Cleaned merged DataFrame with columns [text, label].
    """
    combined = pd.concat(dfs, ignore_index=True)
    before = len(combined)

    # Drop duplicate texts — identical sentences from both datasets would
    # inflate evaluation metrics and cause train/test leakage.
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    after = len(combined)
    print(f"  Dropped {before - after} duplicate texts ({before} → {after} samples)")

    # Sanity check: all labels must be valid class indices.
    assert combined["label"].isin(range(NUM_CLASSES)).all(), (
        f"Found labels outside [0, {NUM_CLASSES - 1}]"
    )

    # Strip leading/trailing whitespace from text.
    combined["text"] = combined["text"].str.strip()
    combined = combined[combined["text"].str.len() > 0].reset_index(drop=True)

    return combined[["text", "label"]]


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform two-stage stratified splitting: first carve out the test set,
    then split the remainder into train/val.

    Stratification ensures each split has the same class distribution as the
    full dataset — critical for fair evaluation with imbalanced classes.

    Args:
        df: Full merged DataFrame with [text, label].

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    # Stage 1: split off test set
    # val_ratio relative to the FULL dataset, not the remainder.
    remaining_ratio = TRAIN_RATIO + VAL_RATIO  # fraction kept after test split
    df_train_val, df_test = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df["label"],
        random_state=SEED,
    )

    # Stage 2: split remainder into train/val
    # val_relative is the fraction of the remainder that becomes val.
    val_relative = VAL_RATIO / remaining_ratio
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_relative,
        stratify=df_train_val["label"],
        random_state=SEED,
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True),
    )


# ============================================================
# Reporting utilities
# ============================================================

def print_class_distribution(df: pd.DataFrame, split_name: str) -> None:
    """
    Print a formatted table of per-class counts and percentages.

    Args:
        df: DataFrame with a 'label' column.
        split_name: Human-readable name for the split (e.g. "train").
    """
    total = len(df)
    print(f"\n  {split_name} ({total} samples):")
    print(f"  {'Label':<12} {'Class':<12} {'Count':>8} {'Pct':>8}")
    print(f"  {'-'*44}")
    for label_idx, label_name in LABEL_MAP.items():
        count = (df["label"] == label_idx).sum()
        pct = 100.0 * count / total
        print(f"  {label_idx:<12} {label_name:<12} {count:>8} {pct:>7.1f}%")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """
    Orchestrate the full dataset preparation pipeline:
    download → merge → clean → split → save → report.
    """
    random.seed(SEED)
    np.random.seed(SEED)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FinCompress — Dataset Preparation")
    print("=" * 60)

    # --- Load datasets ---
    print("\n[1/4] Downloading datasets...")
    df_fpb = load_financial_phrasebank()
    df_fiqa = load_fiqa_sentiment()

    # --- Merge and clean ---
    print("\n[2/4] Merging and cleaning...")
    df_full = merge_and_clean([df_fpb, df_fiqa])
    print(f"  Combined dataset size: {len(df_full)} samples")

    # --- Stratified split ---
    print("\n[3/4] Splitting (70/15/15 stratified)...")
    df_train, df_val, df_test = stratified_split(df_full)

    # --- Save to CSV ---
    print("\n[4/4] Saving CSVs...")
    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    print(f"  Saved: {train_path}")
    print(f"  Saved: {val_path}")
    print(f"  Saved: {test_path}")

    # --- Report class distributions ---
    print("\n" + "=" * 60)
    print("Class Distribution by Split")
    print("=" * 60)
    print_class_distribution(df_train, "train")
    print_class_distribution(df_val, "val")
    print_class_distribution(df_test, "test")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
