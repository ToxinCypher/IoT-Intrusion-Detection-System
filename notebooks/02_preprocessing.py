#!/usr/bin/env python3
"""
02_preprocessing.py — Data Cleaning, Encoding, and Split Generation

This script processes the raw CICIoT2023 dataset:
1. Loads train, validation, and test sets (configurable sample size matching Alve et al. 2025 ~1M rows).
2. Cleans features (drops NaN/Inf, removes zero-variance/constant features, removes duplicates).
3. Downcasts numeric types to float32 to minimize memory usage.
4. Encodes multi-level targets:
   - 34 fine-grained classes
   - 8 coarse categories (matching paper: Benign, DDoS, DoS, Mirai, Recon, Web, Spoofing, BruteForce)
   - Binary classes (Attack vs. Benign)
5. Saves clean, high-performance Parquet files to data/splits/.

Usage:
    python notebooks/02_preprocessing.py
    python notebooks/02_preprocessing.py --train-samples 800000 --test-samples 200000
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib

from src.data_loader import LABEL_CATEGORY_MAP, LABEL_BINARY_MAP
from src.preprocessing import clean_data, encode_labels
from src.utils import logger, load_config, set_seed, ensure_dirs, dataframe_memory_mb, log_memory_usage


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess CICIoT2023 dataset")
    parser.add_argument("--train-samples", type=int, default=800000,
                        help="Number of train samples to load (default: 800,000, matching ~1M total dataset in Alve et al.)")
    parser.add_argument("--test-samples", type=int, default=200000,
                        help="Number of test samples to load (default: 200,000)")
    parser.add_argument("--val-samples", type=int, default=100000,
                        help="Number of validation samples to load (default: 100,000)")
    parser.add_argument("--raw-dir", type=str, default="data/raw/CICIOT23",
                        help="Path to raw CICIOT23 directory")
    parser.add_argument("--output-dir", type=str, default="data/splits",
                        help="Path to save processed parquet splits")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    config = load_config("configs/experiment_config.yaml")
    ensure_dirs(config)

    print("=" * 70)
    print("  STAGE 1: PREPROCESSING & DATA SPLIT GENERATION")
    print("=" * 70)
    logger.info(f"Target sample size: {args.train_samples:,} train, {args.test_samples:,} test, {args.val_samples:,} val")

    raw_path = Path(args.raw_dir)
    train_csv = raw_path / "train" / "train.csv"
    val_csv = raw_path / "validation" / "validation.csv"
    test_csv = raw_path / "test" / "test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Expected CSV files not found in {raw_path}. Looked for:\n"
            f"  {train_csv}\n  {test_csv}"
        )

    # -----------------------------------------------------------------------
    # 1. Load Data
    # -----------------------------------------------------------------------
    logger.info("Loading raw CSV splits...")
    t0 = time.time()
    train_df = pd.read_csv(train_csv, nrows=args.train_samples, low_memory=False)
    test_df = pd.read_csv(test_csv, nrows=args.test_samples, low_memory=False)
    val_df = pd.read_csv(val_csv, nrows=args.val_samples, low_memory=False) if val_csv.exists() else None
    logger.info(f"Loaded CSVs in {time.time() - t0:.2f}s")

    # -----------------------------------------------------------------------
    # 2. Clean Data
    # -----------------------------------------------------------------------
    logger.info("Cleaning training split...")
    train_df = clean_data(train_df)

    logger.info("Cleaning testing split...")
    test_df = clean_data(test_df)

    if val_df is not None:
        logger.info("Cleaning validation split...")
        val_df = clean_data(val_df)

    # -----------------------------------------------------------------------
    # 3. Label Encoding
    # -----------------------------------------------------------------------
    logger.info("Encoding labels across all splits...")
    train_df, encoders = encode_labels(train_df, label_col="label")

    le_fine = encoders["fine"]
    le_cat = encoders["category"]
    le_bin = encoders["binary"]

    # Transform test set using fitted encoders
    test_df["label_encoded"] = test_df["label"].map(
        lambda s: le_fine.transform([s])[0] if s in le_fine.classes_ else -1
    )
    test_df["label_category"] = test_df["label"].map(LABEL_CATEGORY_MAP).fillna("Unknown")
    test_df["label_category_encoded"] = test_df["label_category"].map(
        lambda s: le_cat.transform([s])[0] if s in le_cat.classes_ else -1
    )
    test_df["label_binary"] = test_df["label"].map(LABEL_BINARY_MAP).fillna("Attack")
    test_df["label_binary_encoded"] = le_bin.transform(test_df["label_binary"])

    if val_df is not None:
        val_df["label_encoded"] = val_df["label"].map(
            lambda s: le_fine.transform([s])[0] if s in le_fine.classes_ else -1
        )
        val_df["label_category"] = val_df["label"].map(LABEL_CATEGORY_MAP).fillna("Unknown")
        val_df["label_category_encoded"] = val_df["label_category"].map(
            lambda s: le_cat.transform([s])[0] if s in le_cat.classes_ else -1
        )
        val_df["label_binary"] = val_df["label"].map(LABEL_BINARY_MAP).fillna("Attack")
        val_df["label_binary_encoded"] = le_bin.transform(val_df["label_binary"])

    # Align feature columns (ensure identical features across splits)
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    train_df = train_df[common_cols]
    test_df = test_df[common_cols]

    # -----------------------------------------------------------------------
    # 4. Enforce Strict 80% Train to 20% Test Stratified Split
    # -----------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    logger.info("Enforcing strict 80% Train to 20% Test stratified split...")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    train_df, test_df = train_test_split(
        combined_df,
        test_size=0.20,
        random_state=42,
        stratify=combined_df["label_category_encoded"]
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    total_samples = len(train_df) + len(test_df)
    logger.info(
        f"  Strict 80:20 Split: Train={len(train_df):,} ({len(train_df)/total_samples*100:.1f}%), "
        f"Test={len(test_df):,} ({len(test_df)/total_samples*100:.1f}%)"
    )

    # -----------------------------------------------------------------------
    # 5. Save Processed Parquet Files & Encoders
    # -----------------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving Parquet splits to {out_dir}...")
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)
    if val_df is not None:
        val_df.to_parquet(out_dir / "val.parquet", index=False)

    encoder_save_path = models_dir / "label_encoders.joblib"
    joblib.dump(encoders, encoder_save_path)
    logger.info(f"Saved encoders to {encoder_save_path}")

    # -----------------------------------------------------------------------
    # 5. Summary Statistics
    # -----------------------------------------------------------------------
    feature_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if "label" not in c]

    summary_rows = [
        {"Split": "Train", "Rows": len(train_df), "Features": len(feature_cols), "Memory (MB)": f"{dataframe_memory_mb(train_df):.1f}"},
        {"Split": "Test", "Rows": len(test_df), "Features": len(feature_cols), "Memory (MB)": f"{dataframe_memory_mb(test_df):.1f}"},
    ]
    if val_df is not None:
        summary_rows.append(
            {"Split": "Validation", "Rows": len(val_df), "Features": len(feature_cols), "Memory (MB)": f"{dataframe_memory_mb(val_df):.1f}"}
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = Path("results/tables/preprocessing_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 70)
    print("  PREPROCESSING COMPLETE")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nFeature count: {len(feature_cols)}")
    print(f"Features list: {feature_cols[:8]} ... (+{len(feature_cols)-8} more)")
    print(f"Classes (Category): {list(le_cat.classes_)}")
    print(f"Parquet files saved to: {out_dir.absolute()}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
