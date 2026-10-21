#!/usr/bin/env python3
"""
06_zero_day_experiments.py — Phase 2.3: Zero-Day & Novel Attack Generalization

This script implements Phase 2.3:
1. Tests open-world resilience using Leave-One-Attack-Out (LOAO) experiments.
2. Evaluates 3 distinct holdout configurations:
   - Config 1 (Web Threats): ['Backdoor_Malware', 'MITM-ArpSpoofing', 'SqlInjection']
   - Config 2 (Botnets & Spoofing): ['XSS', 'DNS_Spoofing', 'Mirai-udpplain']
   - Config 3 (Recon & BruteForce): ['CommandInjection', 'DictionaryBruteForce', 'Recon-OSScan']
3. Trains Decision Tree exclusively on known traffic (with strict 80/20 train/val split).
4. Evaluates:
   - Binary Anomaly Recall (% of zero-day attacks flagged as "Attack").
   - Multi-Class Misclassification Mapping (Which known category does it default to?).
   - Prediction Confidence Degradation.
5. Saves results table to results/tables/ and comparison chart to results/figures/.

Usage:
    python notebooks/06_zero_day_experiments.py
"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score

from src.utils import logger, load_config, set_seed, ensure_dirs


def main():
    set_seed(42)
    config = load_config("configs/experiment_config.yaml")
    ensure_dirs(config)

    print("=" * 80)
    print("  PHASE 2.3: ZERO-DAY & NOVEL ATTACK GENERALIZATION EXPERIMENTS")
    print("=" * 80)

    # 1. Load Data
    splits_dir = Path("data/splits")
    train_parquet = splits_dir / "train.parquet"
    test_parquet = splits_dir / "test.parquet"

    if not train_parquet.exists() or not test_parquet.exists():
        raise FileNotFoundError("Preprocessed splits not found!")

    logger.info("Loading preprocessed dataset splits...")
    t0 = time.time()
    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    logger.info(f"Loaded {len(full_df):,} total flows in {time.time()-t0:.2f}s")

    feature_cols = [c for c in full_df.select_dtypes(include=[np.number]).columns if "label" not in c]

    # Encoders
    encoders = joblib.load("models/label_encoders.joblib")
    cat_encoder = encoders["category"]
    benign_cat_idx = cat_encoder.transform(["Benign"])[0]

    # Holdout configurations
    holdout_configs = {
        "Config 1 (Web Threats)": ["Backdoor_Malware", "MITM-ArpSpoofing", "SqlInjection"],
        "Config 2 (Botnets & Spoofing)": ["XSS", "DNS_Spoofing", "Mirai-udpplain"],
        "Config 3 (Recon & Brute Force)": ["CommandInjection", "DictionaryBruteForce", "Recon-OSScan"],
    }

    results = []
    figures_dir = Path("results/figures")
    tables_dir = Path("results/tables")

    for config_name, holdout_attacks in holdout_configs.items():
        print("\n" + "-" * 75)
        logger.info(f"Running {config_name}: Withholding {holdout_attacks}...")

        # Separate zero-day data
        zeroday_mask = full_df["label"].isin(holdout_attacks)
        zeroday_df = full_df[zeroday_mask]
        known_df = full_df[~zeroday_mask]

        logger.info(f"  Known flows: {len(known_df):,}, Held-out Zero-Day flows: {len(zeroday_df):,}")

        # Train on known flows only (strict 80/20 train/test split)
        X_known = known_df[feature_cols]
        y_known = known_df["label_category_encoded"]

        X_train_known, X_test_known, y_train_known, y_test_known = train_test_split(
            X_known, y_known, test_size=0.20, random_state=42, stratify=y_known
        )

        dt = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=30,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        t0_train = time.time()
        dt.fit(X_train_known, y_train_known)
        logger.info(f"  Trained model on known traffic in {time.time()-t0_train:.2f}s")

        # Performance on known test traffic
        y_pred_known = dt.predict(X_test_known)
        known_acc = accuracy_score(y_test_known, y_pred_known)

        # Performance on UNSEEN Zero-Day traffic
        X_zd = zeroday_df[feature_cols]
        y_zd_pred = dt.predict(X_zd)
        y_zd_proba = dt.predict_proba(X_zd)
        zd_conf = np.max(y_zd_proba, axis=1).mean()

        # Binary Anomaly Recall: Did the model classify it as an ATTACK (anything other than Benign)?
        flagged_as_attack = (y_zd_pred != benign_cat_idx).sum()
        binary_anomaly_recall = (flagged_as_attack / len(zeroday_df)) * 100

        # Detailed breakdown per holdout attack
        for attack_name in holdout_attacks:
            sub_mask = (zeroday_df["label"] == attack_name).values
            n_samples = sub_mask.sum()
            if n_samples == 0:
                continue
            sub_preds = y_zd_pred[sub_mask]
            sub_flagged = (sub_preds != benign_cat_idx).sum()
            sub_recall = (sub_flagged / n_samples) * 100

            # Most common misclassified category
            pred_cats, counts = np.unique(sub_preds, return_counts=True)
            top_cat_idx = pred_cats[np.argmax(counts)]
            top_cat_name = cat_encoder.classes_[top_cat_idx]

            logger.info(
                f"    -> {attack_name:25s} ({n_samples:,} flows): "
                f"Binary Recall = {sub_recall:6.2f}%, Masquerades as: '{top_cat_name}'"
            )

            results.append({
                "Configuration": config_name,
                "Heldout Attack": attack_name,
                "Samples": n_samples,
                "Binary Detection Recall (%)": f"{sub_recall:.2f}",
                "Dominant Misclassification": top_cat_name,
                "Known Test Acc (%)": f"{known_acc*100:.2f}",
                "Avg Confidence": f"{zd_conf:.3f}",
            })

    results_df = pd.DataFrame(results)
    results_csv = tables_dir / "zero_day_generalization_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved zero-day results to {results_csv}")

    # Plot Zero-Day Detection Rates
    fig, ax = plt.subplots(figsize=(11, 5))
    attacks = results_df["Heldout Attack"]
    recalls = results_df["Binary Detection Recall (%)"].astype(float)
    colors = ["#2b5c8f" if r >= 90 else ("#d97724" if r >= 70 else "#c93b2b") for r in recalls]

    bars = ax.bar(attacks, recalls, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(90, color="green", linestyle="--", linewidth=1, label="90% Anomaly Threshold")
    ax.axhline(50, color="red", linestyle=":", linewidth=1, label="Random Chance (50%)")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Binary Anomaly Detection Recall (%)", fontsize=11, fontweight="bold")
    ax.set_title("Zero-Day Attack Generalization: Anomaly Detection on Unseen Threats", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(rotation=25, ha="right")
    ax.legend(loc="lower right")

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    chart_path = figures_dir / "zero_day_detection_rates.png"
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {chart_path}")

    print("\n" + "=" * 80)
    print("  ZERO-DAY GENERALIZATION EXPERIMENT SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    print(f"\nArtifacts saved successfully:")
    print(f"  - Results Table: {results_csv}")
    print(f"  - Chart:         {chart_path}")


if __name__ == "__main__":
    main()
