#!/usr/bin/env python3
"""
01_data_loading_eda.py — Data Loading & Exploratory Data Analysis

This script loads the CICIoT2023 dataset, performs initial inspection,
and generates EDA visualizations. Run this FIRST after placing the raw
CSV files in data/raw/.

Usage:
    python notebooks/01_data_loading_eda.py

Or run interactively in Jupyter / VS Code (cell markers: # %%).
"""

# %% [markdown]
# # 1. Data Loading & Exploratory Data Analysis
# **Dataset**: CICIoT2023 (Canadian Institute for Cybersecurity)
# **Goal**: Load, inspect, and understand the dataset before preprocessing.

# %% — Imports and Setup
import sys
import os

# Add project root to path so we can import src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.data_loader import load_all_csvs, dataset_summary, print_summary, get_label_mapping
from src.utils import load_config, set_seed, ensure_dirs, log_memory_usage

# Configure plotting style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Reproducibility
set_seed(42)

# Load config
config = load_config("configs/experiment_config.yaml")
ensure_dirs(config)

print("=" * 60)
print("  CICIoT2023 — Data Loading & EDA")
print("=" * 60)

# %% — Load Data
# ========================================================================
# STEP 1: Load the raw CSV files
# ========================================================================
# Adjust sample_frac to control how much data to load:
#   - 0.05 for quick testing (~500K rows)
#   - 0.1  for development  (~1M rows)
#   - 1.0  for full dataset (~10M+ rows, needs 16+ GB RAM)

SAMPLE_FRAC = config["data"].get("sample_frac", 0.1)
RAW_DIR = config["paths"]["raw_data_dir"]

print(f"\nLoading data from: {RAW_DIR}")
print(f"Sample fraction: {SAMPLE_FRAC}")

# Check if data directory exists and has files
raw_path = Path(RAW_DIR)
if not raw_path.exists():
    raw_path.mkdir(parents=True, exist_ok=True)
    print(f"\n{'!'*60}")
    print(f"  DATA NOT FOUND!")
    print(f"  Please download CICIoT2023 and place CSV files in:")
    print(f"  {raw_path.absolute()}")
    print(f"")
    print(f"  Download options:")
    print(f"  1. Kaggle: https://www.kaggle.com/datasets/madhavmalhotra/unb-cic-iot-dataset-2023")
    print(f"  2. UNB:    https://www.unb.ca/cic/datasets/iotdataset-2023.html")
    print(f"{'!'*60}")

    # Create a small synthetic demo dataset for testing the pipeline
    print("\n  Generating synthetic demo data for pipeline testing...")
    from src.data_loader import LABEL_CATEGORY_MAP

    np.random.seed(42)
    n_samples = 10000
    n_features = 46

    # Generate feature names mimicking CICIoT2023
    feature_names = [
        "flow_duration", "Header_Length", "Protocol Type", "Duration",
        "Rate", "Srate", "Drate", "fin_flag_number", "syn_flag_number",
        "rst_flag_number", "psh_flag_number", "ack_flag_number",
        "ece_flag_number", "cwr_flag_number", "ack_count",
        "syn_count", "fin_count", "urg_count", "rst_count",
        "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
        "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
        "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
        "IAT", "Number", "Magnitue", "Radius", "Covariance",
        "Variance", "Weight",
    ]

    # Create synthetic data with different distributions per attack type
    attack_types = list(LABEL_CATEGORY_MAP.keys())
    rows_per_type = n_samples // len(attack_types)

    dfs = []
    for i, attack in enumerate(attack_types):
        data = np.random.randn(rows_per_type, len(feature_names)) + i * 0.3
        data = np.abs(data)  # Make non-negative
        df_attack = pd.DataFrame(data, columns=feature_names)
        df_attack["label"] = attack
        dfs.append(df_attack)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save demo data
    demo_path = os.path.join(RAW_DIR, "demo_synthetic.csv")
    df.to_csv(demo_path, index=False)
    print(f"  Saved synthetic demo data to {demo_path}")
    print(f"  Shape: {df.shape}")
    print(f"\n  NOTE: Replace with real CICIoT2023 data for actual experiments!\n")

else:
    csv_count = len(list(raw_path.glob("**/*.csv")))
    if csv_count == 0:
        print(f"\n  No CSV files found in {raw_path.absolute()}")
        print(f"  Please download the CICIoT2023 dataset.")
        sys.exit(1)
    print(f"  Found {csv_count} CSV files")

# Load data
df = load_all_csvs(RAW_DIR, sample_frac=SAMPLE_FRAC)

# %% — Dataset Summary
# ========================================================================
# STEP 2: Basic dataset inspection
# ========================================================================
print("\n" + "=" * 60)
print("  DATASET INSPECTION")
print("=" * 60)

summary = dataset_summary(df, label_col="label")
print_summary(summary)

# Dtypes
print("\nColumn types:")
print(df.dtypes.value_counts())

# First few rows
print("\nSample rows:")
print(df.head())

# Null values per column
null_cols = df.isnull().sum()
null_cols = null_cols[null_cols > 0]
if len(null_cols) > 0:
    print(f"\nColumns with null values ({len(null_cols)}):")
    print(null_cols.sort_values(ascending=False))
else:
    print("\nNo null values found ✓")

# Inf values
numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df[numeric_cols]).sum()
inf_cols = inf_counts[inf_counts > 0]
if len(inf_cols) > 0:
    print(f"\nColumns with inf values ({len(inf_cols)}):")
    print(inf_cols.sort_values(ascending=False))
else:
    print("No infinite values found ✓")

# Memory usage
log_memory_usage("after inspection")

# %% — Class Distribution Visualization
# ========================================================================
# STEP 3: Visualize class distribution
# ========================================================================
print("\n" + "=" * 60)
print("  CLASS DISTRIBUTION")
print("=" * 60)

os.makedirs(config["paths"]["figures_dir"], exist_ok=True)

# --- 3a. Fine-grained (34-class) distribution ---
fig, ax = plt.subplots(figsize=(18, 7))
label_counts = df["label"].value_counts()
bars = ax.bar(range(len(label_counts)), label_counts.values, color=sns.color_palette("husl", len(label_counts)))
ax.set_xticks(range(len(label_counts)))
ax.set_xticklabels(label_counts.index, rotation=60, ha="right", fontsize=9)
ax.set_ylabel("Number of Samples")
ax.set_title("CICIoT2023 — Fine-Grained Attack Type Distribution (34 Classes)")
ax.set_yscale("log")  # Log scale due to heavy imbalance

# Add count labels on bars
for bar, count in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f"{count:,}", ha="center", va="bottom", fontsize=7, rotation=45)

plt.tight_layout()
fig.savefig(
    os.path.join(config["paths"]["figures_dir"], "class_distribution_34class.png"),
    dpi=150, bbox_inches="tight"
)
plt.show()
print("Saved: class_distribution_34class.png")

# --- 3b. Category-level (8-category) distribution ---
cat_map = get_label_mapping("category")
df["_category"] = df["label"].map(cat_map).fillna("Unknown")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart
cat_counts = df["_category"].value_counts()
colors = sns.color_palette("Set2", len(cat_counts))
axes[0].bar(range(len(cat_counts)), cat_counts.values, color=colors)
axes[0].set_xticks(range(len(cat_counts)))
axes[0].set_xticklabels(cat_counts.index, rotation=45, ha="right")
axes[0].set_ylabel("Number of Samples")
axes[0].set_title("Category-Level Distribution (8 Categories)")

for i, (cat, count) in enumerate(cat_counts.items()):
    axes[0].text(i, count, f"{count:,}", ha="center", va="bottom", fontsize=9)

# Pie chart
axes[1].pie(
    cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%",
    colors=colors, startangle=90, pctdistance=0.85
)
axes[1].set_title("Category Proportions")

plt.tight_layout()
fig.savefig(
    os.path.join(config["paths"]["figures_dir"], "class_distribution_categories.png"),
    dpi=150, bbox_inches="tight"
)
plt.show()
print("Saved: class_distribution_categories.png")

# Clean up temp column
df.drop(columns=["_category"], inplace=True)

# --- 3c. Imbalance ratio ---
print("\nClass Imbalance Analysis:")
max_class = label_counts.max()
for label, count in label_counts.items():
    ratio = max_class / count
    print(f"  {label:35s}  {count:>10,}  ratio={ratio:>8.1f}:1")

# %% — Feature Statistics
# ========================================================================
# STEP 4: Feature-level statistics
# ========================================================================
print("\n" + "=" * 60)
print("  FEATURE STATISTICS")
print("=" * 60)

feature_cols = [c for c in df.columns if c != "label"]
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"\nTotal features: {len(feature_cols)}")
print(f"Numeric features: {len(numeric_features)}")
print(f"Non-numeric features: {len(feature_cols) - len(numeric_features)}")

# Basic stats
print("\nFeature statistics (numeric):")
stats = df[numeric_features].describe().T
stats["null_count"] = df[numeric_features].isnull().sum()
stats["inf_count"] = np.isinf(df[numeric_features]).sum()
stats["unique_count"] = df[numeric_features].nunique()
stats["zero_frac"] = (df[numeric_features] == 0).mean()
print(stats[["mean", "std", "min", "max", "null_count", "zero_frac"]].round(4).to_string())

# Save stats table
os.makedirs(config["paths"]["tables_dir"], exist_ok=True)
stats.to_csv(os.path.join(config["paths"]["tables_dir"], "feature_statistics.csv"))
print("\nSaved: feature_statistics.csv")

# %% — Feature Distributions
# ========================================================================
# STEP 5: Feature distributions (top features by variance)
# ========================================================================
print("\n" + "=" * 60)
print("  FEATURE DISTRIBUTIONS")
print("=" * 60)

# Plot distributions of top-12 features by variance
top_var_features = df[numeric_features].var().nlargest(12).index.tolist()

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
for i, feat in enumerate(top_var_features):
    ax = axes[i // 4, i % 4]
    # Use clipped data for better visualization
    data = df[feat].clip(lower=df[feat].quantile(0.01), upper=df[feat].quantile(0.99))
    ax.hist(data, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_title(feat, fontsize=10)
    ax.set_ylabel("Count")

plt.suptitle("Distribution of Top-12 Features (by Variance)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(
    os.path.join(config["paths"]["figures_dir"], "feature_distributions_top12.png"),
    dpi=150, bbox_inches="tight"
)
plt.show()
print("Saved: feature_distributions_top12.png")

# %% — Correlation Heatmap
# ========================================================================
# STEP 6: Correlation analysis
# ========================================================================
print("\n" + "=" * 60)
print("  CORRELATION ANALYSIS")
print("=" * 60)

# Compute correlation matrix (on a sample for speed)
sample_for_corr = df[numeric_features].sample(n=min(50000, len(df)), random_state=42)
corr_matrix = sample_for_corr.corr()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.95:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j],
            ))

print(f"\nHighly correlated feature pairs (|r| > 0.95): {len(high_corr_pairs)}")
for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2]))[:15]:
    print(f"  {f1:30s} ↔ {f2:30s}  r={r:.4f}")

# Heatmap (show top-20 features by variance for readability)
top_20 = df[numeric_features].var().nlargest(20).index.tolist()
corr_top20 = corr_matrix.loc[top_20, top_20]

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_top20, dtype=bool), k=1)
sns.heatmap(
    corr_top20, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
    square=True, cbar_kws={"shrink": 0.8}
)
ax.set_title("Feature Correlation Heatmap (Top 20 by Variance)", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(
    os.path.join(config["paths"]["figures_dir"], "correlation_heatmap.png"),
    dpi=150, bbox_inches="tight"
)
plt.show()
print("Saved: correlation_heatmap.png")

# %% — Per-Category Feature Means
# ========================================================================
# STEP 7: Feature behavior per attack category
# ========================================================================
print("\n" + "=" * 60)
print("  PER-CATEGORY FEATURE ANALYSIS")
print("=" * 60)

cat_map = get_label_mapping("category")
df["_category"] = df["label"].map(cat_map).fillna("Unknown")

# Mean feature values per category (top-10 features by variance)
top_10_feats = df[numeric_features].var().nlargest(10).index.tolist()
category_means = df.groupby("_category")[top_10_feats].mean()

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(
    category_means.T, annot=True, fmt=".2f", cmap="YlOrRd",
    ax=ax, linewidths=0.5,
)
ax.set_title("Mean Feature Values per Attack Category (Top 10 Features)", fontsize=14, fontweight="bold")
ax.set_xlabel("Attack Category")
ax.set_ylabel("Feature")
plt.tight_layout()
fig.savefig(
    os.path.join(config["paths"]["figures_dir"], "category_feature_means.png"),
    dpi=150, bbox_inches="tight"
)
plt.show()
print("Saved: category_feature_means.png")

# Clean up
df.drop(columns=["_category"], inplace=True)

# %% — Save Processed Summary
# ========================================================================
# STEP 8: Save EDA summary for reference
# ========================================================================
print("\n" + "=" * 60)
print("  SAVING EDA RESULTS")
print("=" * 60)

# Save the label counts
label_counts_df = pd.DataFrame({
    "attack_type": label_counts.index,
    "count": label_counts.values,
    "percentage": (label_counts.values / label_counts.sum() * 100).round(2),
})
label_counts_df.to_csv(
    os.path.join(config["paths"]["tables_dir"], "label_distribution.csv"),
    index=False,
)
print("Saved: label_distribution.csv")

# Save correlation matrix
corr_matrix.to_csv(
    os.path.join(config["paths"]["tables_dir"], "correlation_matrix.csv"),
)
print("Saved: correlation_matrix.csv")

# %% — Summary
print("\n" + "=" * 60)
print("  EDA COMPLETE ✓")
print("=" * 60)
print(f"""
Key findings:
  • Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns
  • Attack types: {df['label'].nunique()} (34 expected)
  • Numeric features: {len(numeric_features)}
  • Highly correlated pairs (|r|>0.95): {len(high_corr_pairs)}
  • Null values: {df.isnull().sum().sum():,}

Figures saved to: {config['paths']['figures_dir']}/
Tables saved to:  {config['paths']['tables_dir']}/

Next step: Run notebooks/02_preprocessing.py
""")

log_memory_usage("end of EDA")
