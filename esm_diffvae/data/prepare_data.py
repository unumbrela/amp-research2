"""Data preparation: download, merge, deduplicate, and split AMP datasets.

Strategy: Start with Diff-AMP's local data for quick development,
then optionally expand to DBAASP/APD/DRAMP for full training.
"""

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Standard 20 amino acids
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCES_DIR = PROJECT_ROOT.parent / "references"
DATA_DIR = PROJECT_ROOT / "data"


def load_diffamp_data() -> pd.DataFrame:
    """Load data from Diff-AMP reference repository."""
    diffamp_dir = REFERENCES_DIR / "diff-amp" / "data"
    frames = []

    for csv_file in ["AMPdb_data.csv", "training_data.csv", "val_data.csv"]:
        path = diffamp_dir / csv_file
        if path.exists():
            df = pd.read_csv(path)
            # Diff-AMP uses columns: Seq, Label
            if "Seq" in df.columns and "Label" in df.columns:
                df = df.rename(columns={"Seq": "sequence", "Label": "is_amp"})
                df["source"] = f"diffamp_{csv_file}"
                frames.append(df[["sequence", "is_amp", "source"]])
                print(f"  Loaded {len(df)} sequences from {csv_file}")

    if frames:
        return pd.concat(frames, ignore_index=True)
    print("  WARNING: No Diff-AMP data found")
    return pd.DataFrame(columns=["sequence", "is_amp", "source"])


def load_hydramp_data() -> pd.DataFrame:
    """Load data from HydrAMP reference repository (if downloaded)."""
    hydramp_dir = REFERENCES_DIR / "hydramp" / "data"
    frames = []

    # HydrAMP stores data as CSV with Name, Sequence columns
    for csv_file in hydramp_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if "Sequence" in df.columns:
                df = df.rename(columns={"Sequence": "sequence"})
                df["is_amp"] = 1  # HydrAMP dataset contains only AMPs
                df["source"] = f"hydramp_{csv_file.name}"
                frames.append(df[["sequence", "is_amp", "source"]])
                print(f"  Loaded {len(df)} sequences from {csv_file.name}")
        except Exception as e:
            print(f"  Skipping {csv_file.name}: {e}")

    if frames:
        return pd.concat(frames, ignore_index=True)
    print("  WARNING: No HydrAMP data found (run HydrAMP's data download first)")
    return pd.DataFrame(columns=["sequence", "is_amp", "source"])


def load_external_csv(path: str) -> pd.DataFrame:
    """Load an external CSV file. Expects 'sequence' column at minimum."""
    df = pd.read_csv(path)
    required_cols = {"sequence"}
    if not required_cols.issubset(df.columns):
        # Try common column name variants
        col_map = {}
        for col in df.columns:
            if col.lower() in ("seq", "sequence", "peptide", "peptide_sequence"):
                col_map[col] = "sequence"
            elif col.lower() in ("label", "is_amp", "activity", "amp"):
                col_map[col] = "is_amp"
        df = df.rename(columns=col_map)

    if "sequence" not in df.columns:
        raise ValueError(f"Cannot find sequence column in {path}. Columns: {list(df.columns)}")

    if "is_amp" not in df.columns:
        df["is_amp"] = 1  # Assume all are AMPs if no label

    df["source"] = f"external_{Path(path).name}"
    return df[["sequence", "is_amp", "source"]]


def validate_sequence(seq: str) -> bool:
    """Check if sequence contains only standard amino acids."""
    return bool(seq) and all(c in STANDARD_AA for c in seq.upper())


def filter_sequences(df: pd.DataFrame, min_len: int = 5, max_len: int = 50) -> pd.DataFrame:
    """Filter sequences by validity and length."""
    original_len = len(df)
    df = df.copy()
    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()

    # Remove empty / NaN
    df = df.dropna(subset=["sequence"])
    df = df[df["sequence"].str.len() > 0]

    # Standard AA only
    df = df[df["sequence"].apply(validate_sequence)]

    # Length filter
    df = df[df["sequence"].str.len().between(min_len, max_len)]

    print(f"  Filtered: {original_len} -> {len(df)} sequences (len {min_len}-{max_len}, standard AA)")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate sequences, keeping first occurrence."""
    original_len = len(df)
    df = df.drop_duplicates(subset=["sequence"], keep="first")
    print(f"  Deduplicated: {original_len} -> {len(df)} unique sequences")
    return df


def add_property_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add placeholder property columns needed for conditioning."""
    df = df.copy()

    # Ensure is_amp is numeric
    df["is_amp"] = pd.to_numeric(df["is_amp"], errors="coerce").fillna(0).astype(int)

    # MIC value: NaN placeholder (to be filled from external sources if available)
    if "mic_value" not in df.columns:
        df["mic_value"] = np.nan

    # Toxicity: NaN placeholder
    if "is_toxic" not in df.columns:
        df["is_toxic"] = np.nan

    # Hemolysis: NaN placeholder
    if "is_hemolytic" not in df.columns:
        df["is_hemolytic"] = np.nan

    # Normalized length (useful as conditioning signal)
    df["length"] = df["sequence"].str.len()
    df["length_norm"] = df["length"] / 50.0  # normalize to [0, 1] range

    return df


def split_data(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1,
               seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train/val/test."""
    from sklearn.model_selection import train_test_split

    # Stratify by AMP label
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=seed, stratify=df["is_amp"]
    )
    relative_val = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), random_state=seed, stratify=temp_df["is_amp"]
    )

    print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Prepare AMP dataset")
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-csv", type=str, nargs="*", default=[],
                        help="Additional CSV files to include")
    args = parser.parse_args()

    print("=== AMP Data Preparation ===\n")

    # Step 1: Load data from all sources
    print("Loading Diff-AMP data...")
    df_diffamp = load_diffamp_data()

    print("\nLoading HydrAMP data...")
    df_hydramp = load_hydramp_data()

    frames = [df_diffamp, df_hydramp]

    for ext_csv in args.external_csv:
        print(f"\nLoading external: {ext_csv}")
        frames.append(load_external_csv(ext_csv))

    # Step 2: Merge
    print("\nMerging all sources...")
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total: {len(df)} sequences from {df['source'].nunique()} sources")

    # Step 3: Filter
    print("\nFiltering sequences...")
    df = filter_sequences(df, min_len=args.min_len, max_len=args.max_len)

    # Step 4: Deduplicate
    print("\nDeduplicating...")
    df = deduplicate(df)

    # Step 5: Add property columns
    print("\nAdding property columns...")
    df = add_property_columns(df)

    # Step 6: Split
    print("\nSplitting data...")
    train_df, val_df, test_df = split_data(df, seed=args.seed)

    # Step 7: Save
    processed_dir = DATA_DIR / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)
    df.to_csv(processed_dir / "all.csv", index=False)

    print(f"\nSaved to {processed_dir}/")
    print(f"  train.csv: {len(train_df)} sequences ({train_df['is_amp'].mean():.1%} AMP)")
    print(f"  val.csv:   {len(val_df)} sequences ({val_df['is_amp'].mean():.1%} AMP)")
    print(f"  test.csv:  {len(test_df)} sequences ({test_df['is_amp'].mean():.1%} AMP)")

    # Print summary stats
    print(f"\nSequence length stats:")
    print(f"  Mean: {df['length'].mean():.1f}")
    print(f"  Median: {df['length'].median():.0f}")
    print(f"  Range: {df['length'].min()}-{df['length'].max()}")


if __name__ == "__main__":
    main()
