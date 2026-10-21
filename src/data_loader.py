"""
Data loader — reads and merges CICIoT2023 CSV files from disk.

Expected raw data layout:
    data/raw/
    ├── part-00000-*.csv
    ├── part-00001-*.csv
    └── ...
    (310+ CSV files, each containing a subset of flows)

Or organized by attack category in subdirectories.
"""

import os
import glob
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.utils import logger, timed, log_memory_usage, dataframe_memory_mb


# ---------------------------------------------------------------------------
# Label mapping: 34 attack types → 8 categories
# ---------------------------------------------------------------------------

LABEL_CATEGORY_MAP = {
    "BenignTraffic": "Benign",
    # DDoS variants
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    # DoS variants
    "DoS-UDP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    # Mirai botnet
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # Reconnaissance
    "Recon-PingSweep": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PortScan": "Recon",
    "Recon-HostDiscovery": "Recon",
    "VulnerabilityScan": "Recon",
    # Web attacks
    "BrowserHijacking": "Web",
    "CommandInjection": "Web",
    "SqlInjection": "Web",
    "XSS": "Web",
    "Backdoor_Malware": "Web",
    "Uploading_Attack": "Web",
    # Spoofing
    "MITM-ArpSpoofing": "Spoofing",
    "DNS_Spoofing": "Spoofing",
    # Brute Force
    "DictionaryBruteForce": "BruteForce",
}

# Binary mapping
LABEL_BINARY_MAP = {k: ("Benign" if v == "Benign" else "Attack") for k, v in LABEL_CATEGORY_MAP.items()}


def get_label_mapping(level: str = "category") -> dict:
    """
    Return label mapping dict.

    Parameters
    ----------
    level : str
        'category' — 34 types → 8 categories
        'binary'   — 34 types → Attack / Benign

    Returns
    -------
    dict : mapping from fine-grained labels to coarse labels
    """
    if level == "category":
        return LABEL_CATEGORY_MAP.copy()
    elif level == "binary":
        return LABEL_BINARY_MAP.copy()
    else:
        raise ValueError(f"Unknown label level: {level}. Use 'category' or 'binary'.")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

@timed
def load_single_csv(filepath: str, sample_frac: float = 1.0,
                    random_state: int = 42) -> pd.DataFrame:
    """
    Load a single CSV file, optionally sampling a fraction of rows.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    sample_frac : float
        Fraction of rows to keep (0.0–1.0). Default 1.0 = keep all.
    random_state : int
        Seed for sampling reproducibility.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath, low_memory=False)

    if sample_frac < 1.0:
        n_sample = max(1, int(len(df) * sample_frac))
        df = df.sample(n=n_sample, random_state=random_state)

    return df


@timed
def load_all_csvs(
    raw_dir: str = "data/raw",
    sample_frac: float = 0.1,
    max_files: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load and concatenate all CSV files from the raw data directory.

    Supports both flat structure (CSVs directly in raw_dir) and
    nested structure (CSVs in subdirectories by attack type).

    Parameters
    ----------
    raw_dir : str
        Path to directory containing raw CSV files.
    sample_frac : float
        Fraction of rows to sample from each file (0.0–1.0).
        Use 0.1 for development, 1.0 for final training.
    max_files : int, optional
        Maximum number of CSV files to load (for quick testing).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame : concatenated data from all CSVs
    """
    # Find all CSV files (flat + nested)
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            f"Please download the CICIoT2023 dataset and place CSV files in {raw_dir}/"
        )

    if max_files:
        csv_files = csv_files[:max_files]

    logger.info(f"Found {len(csv_files)} CSV files in {raw_dir}")
    log_memory_usage("before loading")

    frames = []
    for i, fpath in enumerate(tqdm(csv_files, desc="Loading CSVs")):
        try:
            df = pd.read_csv(fpath, low_memory=False)

            if sample_frac < 1.0:
                n_sample = max(1, int(len(df) * sample_frac))
                df = df.sample(n=n_sample, random_state=random_state)

            frames.append(df)

            if (i + 1) % 50 == 0:
                logger.info(f"  Loaded {i + 1}/{len(csv_files)} files...")

        except Exception as e:
            logger.warning(f"  Skipping {fpath}: {e}")

    logger.info("Concatenating all DataFrames...")
    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        f"Loaded dataset: {combined.shape[0]:,} rows × {combined.shape[1]} columns "
        f"({dataframe_memory_mb(combined):.1f} MB)"
    )
    log_memory_usage("after loading")

    return combined


# ---------------------------------------------------------------------------
# Quick dataset summary
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame, label_col: str = "label") -> dict:
    """
    Compute a quick summary of the loaded dataset.

    Returns
    -------
    dict with keys: shape, dtypes, null_counts, duplicate_count,
                    label_counts, label_percentages, memory_mb
    """
    summary = {
        "shape": df.shape,
        "n_features": df.shape[1] - 1,  # exclude label
        "dtypes": df.dtypes.value_counts().to_dict(),
        "null_counts": df.isnull().sum().sum(),
        "null_columns": df.columns[df.isnull().any()].tolist(),
        "duplicate_count": df.duplicated().sum(),
        "memory_mb": dataframe_memory_mb(df),
    }

    if label_col in df.columns:
        counts = df[label_col].value_counts()
        summary["label_counts"] = counts.to_dict()
        summary["label_percentages"] = (counts / len(df) * 100).round(2).to_dict()
        summary["n_classes"] = df[label_col].nunique()

    return summary


def print_summary(summary: dict):
    """Pretty-print the dataset summary."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape:           {summary['shape'][0]:,} rows × {summary['shape'][1]} columns")
    print(f"Features:        {summary['n_features']}")
    print(f"Memory:          {summary['memory_mb']:.1f} MB")
    print(f"Null values:     {summary['null_counts']:,}")
    print(f"Duplicates:      {summary['duplicate_count']:,}")

    if "n_classes" in summary:
        print(f"Classes:         {summary['n_classes']}")
        print(f"\nLabel Distribution:")
        for label, count in sorted(summary["label_counts"].items(), key=lambda x: -x[1]):
            pct = summary["label_percentages"][label]
            print(f"  {label:35s}  {count:>10,}  ({pct:6.2f}%)")
    print("=" * 60)
