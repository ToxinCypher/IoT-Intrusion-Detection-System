"""
Preprocessing — cleaning, encoding, scaling, train/test splitting.
"""

import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data_loader import get_label_mapping
from src.utils import logger, timed, dataframe_memory_mb


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

@timed
def clean_data(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Clean the raw CICIoT2023 DataFrame.

    Steps:
    1. Drop rows with NaN or Inf values
    2. Remove constant-value (zero-variance) columns
    3. Remove exact duplicate rows
    4. Downcast float64 → float32 to reduce memory

    Parameters
    ----------
    df : pd.DataFrame
        Raw data.
    label_col : str
        Name of the label column (excluded from numeric cleaning).

    Returns
    -------
    pd.DataFrame : cleaned data
    """
    n_original = len(df)
    logger.info(f"Cleaning data: {n_original:,} rows, {df.shape[1]} columns")

    # --- 1. Replace inf with NaN, then drop rows with any NaN ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    n_before = len(df)
    df = df.dropna()
    n_dropped_nan = n_before - len(df)
    if n_dropped_nan > 0:
        logger.info(f"  Dropped {n_dropped_nan:,} rows with NaN/Inf values")

    # --- 2. Remove constant columns ---
    feature_cols = [c for c in numeric_cols if c != label_col and c in df.columns]
    constant_cols = [c for c in feature_cols if df[c].nunique() <= 1]
    if constant_cols:
        logger.info(f"  Dropping {len(constant_cols)} constant columns: {constant_cols}")
        df = df.drop(columns=constant_cols)

    # --- 3. Remove duplicates ---
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped_dup = n_before - len(df)
    if n_dropped_dup > 0:
        logger.info(f"  Dropped {n_dropped_dup:,} duplicate rows")

    # --- 4. Downcast floats to save memory ---
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        mem_before = dataframe_memory_mb(df)
        df[float_cols] = df[float_cols].astype(np.float32)
        mem_after = dataframe_memory_mb(df)
        logger.info(
            f"  Downcast float64→float32: {mem_before:.1f} MB → {mem_after:.1f} MB "
            f"(saved {mem_before - mem_after:.1f} MB)"
        )

    logger.info(
        f"  Cleaning complete: {len(df):,} rows remaining "
        f"(dropped {n_original - len(df):,} total)"
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

@timed
def encode_labels(
    df: pd.DataFrame,
    label_col: str = "label",
    add_category: bool = True,
    add_binary: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode labels at multiple granularity levels.

    Parameters
    ----------
    df : pd.DataFrame
        Data with a string label column.
    label_col : str
        Name of the label column.
    add_category : bool
        Add an 8-category label column ('label_category').
    add_binary : bool
        Add a binary label column ('label_binary').

    Returns
    -------
    (pd.DataFrame, dict) : data with encoded labels, and encoders dict
    """
    encoders = {}

    # Fine-grained label encoding (34 classes)
    le_fine = LabelEncoder()
    df["label_encoded"] = le_fine.fit_transform(df[label_col])
    encoders["fine"] = le_fine
    logger.info(f"  Encoded {len(le_fine.classes_)} fine-grained classes")

    # Category-level encoding (8 categories)
    if add_category:
        cat_map = get_label_mapping("category")
        df["label_category"] = df[label_col].map(cat_map)

        # Check for unmapped labels
        unmapped = df["label_category"].isna().sum()
        if unmapped > 0:
            unknown_labels = df.loc[df["label_category"].isna(), label_col].unique()
            logger.warning(
                f"  {unmapped} rows have unmapped labels: {unknown_labels}. "
                "Mapping to 'Unknown'."
            )
            df["label_category"] = df["label_category"].fillna("Unknown")

        le_cat = LabelEncoder()
        df["label_category_encoded"] = le_cat.fit_transform(df["label_category"])
        encoders["category"] = le_cat
        logger.info(f"  Encoded {len(le_cat.classes_)} category classes: {list(le_cat.classes_)}")

    # Binary encoding
    if add_binary:
        bin_map = get_label_mapping("binary")
        df["label_binary"] = df[label_col].map(bin_map).fillna("Attack")
        le_bin = LabelEncoder()
        df["label_binary_encoded"] = le_bin.fit_transform(df["label_binary"])
        encoders["binary"] = le_bin
        logger.info(f"  Encoded binary classes: {list(le_bin.classes_)}")

    return df, encoders


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

@timed
def scale_features(
    X_train: pd.DataFrame,
    X_val: Optional[pd.DataFrame] = None,
    X_test: Optional[pd.DataFrame] = None,
) -> Tuple:
    """
    Standardize features using StandardScaler (fit on train only).

    Returns
    -------
    Scaled versions of (X_train, X_val, X_test, scaler).
    X_val and X_test are None if not provided.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_val_scaled = None
    if X_val is not None:
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    logger.info("  Feature scaling complete (StandardScaler)")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# Train / Val / Test Split
# ---------------------------------------------------------------------------

@timed
def create_splits(
    df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label_category_encoded",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    save_dir: Optional[str] = "data/splits",
) -> dict:
    """
    Create stratified train/val/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data with encoded labels.
    feature_cols : list
        List of feature column names.
    label_col : str
        Encoded label column to use.
    test_size : float
        Fraction for test set.
    val_size : float
        Fraction for validation set (from remaining after test).
    random_state : int
        Seed.
    save_dir : str, optional
        Directory to save splits as parquet files.

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df[feature_cols]
    y = df[label_col]

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Second split: train vs val
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_trainval,
    )

    splits = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    logger.info(
        f"  Splits: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}"
    )

    # Save to parquet
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, data in splits.items():
            path = os.path.join(save_dir, f"{name}.parquet")
            if isinstance(data, pd.DataFrame):
                data.to_parquet(path)
            else:
                pd.DataFrame(data).to_parquet(path)
            logger.debug(f"  Saved {name} → {path}")
        logger.info(f"  All splits saved to {save_dir}/")

    return splits
