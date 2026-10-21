#!/usr/bin/env python3
"""
04_feature_reduction_edge_profiling.py — Phase 2.1: Lightweight Optimization

This script implements Phase 2.1:
1. Filters redundant collinear features (Pearson correlation > 0.95).
2. Computes feature importance rankings using Mutual Information & Tree Gini.
3. Evaluates models across 4 feature subsets:
   - Full (40 features)
   - Top-25 features
   - Top-15 features (Target Edge Model)
   - Top-10 features (Ultra-constrained Edge Model)
4. Profiles hard edge metrics: Accuracy, Macro F1, Inference Latency (µs/sample), Model Size (KB).
5. Exports Pareto frontier plots and saves the edge-optimized model:
   models/decision_tree_top15.joblib.

Usage:
    python notebooks/04_feature_reduction_edge_profiling.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import mutual_info_classif

from src.feature_engineering import filter_correlated_features
from src.utils import logger, load_config, set_seed, ensure_dirs


def save_pareto_plot(results_df, metric_col, title, ylabel, save_path, ylim=None):
    """Save Pareto tradeoff line chart."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, color, marker in [("Decision Tree", "#1f77b4", "o"), ("Random Forest", "#ff7f0e", "s")]:
        sub = results_df[results_df["Model"].str.contains(model_name)]
        if not sub.empty:
            sub = sub.sort_values(by="Num Features")
            ax.plot(
                sub["Num Features"],
                sub[metric_col].astype(float),
                label=model_name,
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=8,
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    f"{float(row[metric_col]):.2f}",
                    (row["Num Features"], float(row[metric_col])),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_xlabel("Number of Features", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks([10, 15, 25, 40])
    if ylim:
        ax.set_ylim(ylim)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {save_path}")


def main():
    set_seed(42)
    config = load_config("configs/experiment_config.yaml")
    ensure_dirs(config)

    print("=" * 80)
    print("  PHASE 2.1: LIGHTWEIGHT FEATURE REDUCTION & EDGE PROFILING")
    print("=" * 80)

    # 1. Load Parquet Splits
    splits_dir = Path("data/splits")
    train_parquet = splits_dir / "train.parquet"
    test_parquet = splits_dir / "test.parquet"

    if not train_parquet.exists() or not test_parquet.exists():
        raise FileNotFoundError(
            f"Preprocessed splits not found in {splits_dir}. Run 02_preprocessing.py first!"
        )

    logger.info("Loading preprocessed parquet splits...")
    t0 = time.time()
    train_df = pd.read_parquet(train_parquet)
    test_df = pd.read_parquet(test_parquet)
    logger.info(f"Loaded train ({len(train_df):,} rows) and test ({len(test_df):,} rows) in {time.time()-t0:.2f}s")

    feature_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if "label" not in c]
    logger.info(f"Total initial features: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df["label_category_encoded"]
    X_test = test_df[feature_cols]
    y_test = test_df["label_category_encoded"]

    # 2. Multicollinearity Filtering (Pearson correlation > 0.95)
    logger.info("Computing Pearson correlation matrix and filtering collinear features...")
    X_train_filtered, dropped_corr = filter_correlated_features(X_train, threshold=0.95)
    logger.info(f"Dropped {len(dropped_corr)} collinear features: {dropped_corr}")

    # 3. Feature Importance Ranking via Mutual Information & Tree Gini
    logger.info("Computing Mutual Information scores (using 50,000 stratified samples)...")
    sample_size = min(50000, len(X_train))
    sample_indices = train_df.groupby("label_category_encoded", group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) / len(train_df) * sample_size)), random_state=42)
    ).index

    X_sample = X_train.loc[sample_indices]
    y_sample = y_train.loc[sample_indices]

    mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

    # Train a baseline DT to get Gini importance
    dt_temp = DecisionTreeClassifier(max_depth=20, random_state=42)
    dt_temp.fit(X_train, y_train)
    gini_series = pd.Series(dt_temp.feature_importances_, index=feature_cols).sort_values(ascending=False)

    ranking_df = pd.DataFrame({
        "Mutual_Information": mi_series,
        "Gini_Importance": gini_series.reindex(mi_series.index)
    })
    ranking_df["Composite_Rank"] = (
        ranking_df["Mutual_Information"].rank(ascending=False) +
        ranking_df["Gini_Importance"].rank(ascending=False)
    ).rank()
    ranking_df = ranking_df.sort_values(by="Composite_Rank")

    tables_dir = Path("results/tables")
    figures_dir = Path("results/figures")
    models_dir = Path("models")

    ranking_path = tables_dir / "feature_importance_ranking.csv"
    ranking_df.to_csv(ranking_path)
    logger.info(f"Saved feature importance rankings to {ranking_path}")

    sorted_features = ranking_df.index.tolist()

    # 4. Benchmarking Feature Tiers
    feature_subsets = {
        "Full (40)": sorted_features[:40],
        "Top-25": sorted_features[:25],
        "Top-15 (Edge Target)": sorted_features[:15],
        "Top-10 (Ultra-Constrained)": sorted_features[:10],
    }

    results = []

    for tier_name, selected_features in feature_subsets.items():
        n_feats = len(selected_features)
        print("\n" + "-" * 75)
        logger.info(f"Benchmarking Tier: {tier_name} ({n_feats} features)...")
        logger.info(f"  Features: {selected_features[:5]} ... (+{n_feats-5} more)")

        X_tr = X_train[selected_features]
        X_te = X_test[selected_features]

        # A. Decision Tree
        dt = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=30,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        t0_train = time.perf_counter()
        dt.fit(X_tr, y_train)
        t_train_dt = time.perf_counter() - t0_train

        # Save top-15 model specifically for Edge deployment
        if n_feats == 15:
            joblib.dump(dt, models_dir / "decision_tree_top15.joblib")
            logger.info("  Saved models/decision_tree_top15.joblib (Optimized Edge Model)")

        # Save model temporarily to measure disk size
        temp_model_path = models_dir / "temp_eval.joblib"
        joblib.dump(dt, temp_model_path)
        dt_size_kb = os.path.getsize(temp_model_path) / 1024
        temp_model_path.unlink()

        t0_infer = time.perf_counter()
        y_pred_dt = dt.predict(X_te)
        t_infer_dt = time.perf_counter() - t0_infer
        latency_dt_us = (t_infer_dt / len(X_te)) * 1_000_000

        acc_dt = accuracy_score(y_test, y_pred_dt)
        f1_macro_dt = f1_score(y_test, y_pred_dt, average="macro", zero_division=0)
        f1_weighted_dt = f1_score(y_test, y_pred_dt, average="weighted", zero_division=0)

        results.append({
            "Tier": tier_name,
            "Num Features": n_feats,
            "Model": "Decision Tree",
            "Accuracy (%)": f"{acc_dt * 100:.2f}",
            "Macro F1 (%)": f"{f1_macro_dt * 100:.2f}",
            "Weighted F1 (%)": f"{f1_weighted_dt * 100:.2f}",
            "Latency (µs)": f"{latency_dt_us:.3f}",
            "Model Size (KB)": f"{dt_size_kb:.1f}",
            "Train Time (s)": f"{t_train_dt:.2f}",
        })

        # B. Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            criterion="gini",
            random_state=42,
            n_jobs=-1
        )
        t0_train_rf = time.perf_counter()
        rf.fit(X_tr, y_train)
        t_train_rf = time.perf_counter() - t0_train_rf

        joblib.dump(rf, temp_model_path)
        rf_size_kb = os.path.getsize(temp_model_path) / 1024
        temp_model_path.unlink()

        t0_infer_rf = time.perf_counter()
        y_pred_rf = rf.predict(X_te)
        t_infer_rf = time.perf_counter() - t0_infer_rf
        latency_rf_us = (t_infer_rf / len(X_te)) * 1_000_000

        acc_rf = accuracy_score(y_test, y_pred_rf)
        f1_macro_rf = f1_score(y_test, y_pred_rf, average="macro", zero_division=0)
        f1_weighted_rf = f1_score(y_test, y_pred_rf, average="weighted", zero_division=0)

        results.append({
            "Tier": tier_name,
            "Num Features": n_feats,
            "Model": "Random Forest",
            "Accuracy (%)": f"{acc_rf * 100:.2f}",
            "Macro F1 (%)": f"{f1_macro_rf * 100:.2f}",
            "Weighted F1 (%)": f"{f1_weighted_rf * 100:.2f}",
            "Latency (µs)": f"{latency_rf_us:.3f}",
            "Model Size (KB)": f"{rf_size_kb:.1f}",
            "Train Time (s)": f"{t_train_rf:.2f}",
        })

        logger.info(
            f"  DT -> Acc: {acc_dt*100:.2f}%, F1: {f1_weighted_dt*100:.2f}%, Latency: {latency_dt_us:.3f} µs, Size: {dt_size_kb:.1f} KB"
        )
        logger.info(
            f"  RF -> Acc: {acc_rf*100:.2f}%, F1: {f1_weighted_rf*100:.2f}%, Latency: {latency_rf_us:.3f} µs, Size: {rf_size_kb/1024:.1f} MB"
        )

    # 5. Save Benchmarks Table
    benchmarks_df = pd.DataFrame(results)
    benchmarks_csv = tables_dir / "feature_reduction_benchmarks.csv"
    benchmarks_df.to_csv(benchmarks_csv, index=False)
    logger.info(f"Saved feature reduction benchmarks to {benchmarks_csv}")

    # 6. Generate Pareto Frontier Visualizations
    save_pareto_plot(
        benchmarks_df,
        metric_col="Accuracy (%)",
        title="Pareto Frontier: Accuracy vs. Feature Count",
        ylabel="Test Accuracy (%)",
        save_path=figures_dir / "accuracy_vs_feature_count.png",
        ylim=(95, 100)
    )

    save_pareto_plot(
        benchmarks_df,
        metric_col="Latency (µs)",
        title="Inference Latency vs. Feature Count (Lower is Better)",
        ylabel="Inference Latency (µs / sample)",
        save_path=figures_dir / "latency_vs_feature_count.png"
    )

    print("\n" + "=" * 80)
    print("  FEATURE REDUCTION & EDGE PROFILING SUMMARY")
    print("=" * 80)
    print(benchmarks_df.to_string(index=False))
    print("=" * 80)
    print(f"\nDeliverables Generated:")
    print(f"  - Top-15 Edge Model:  models/decision_tree_top15.joblib")
    print(f"  - Feature Rankings:   {ranking_path}")
    print(f"  - Benchmark Table:    {benchmarks_csv}")
    print(f"  - Pareto Curves:      {figures_dir / 'accuracy_vs_feature_count.png'}")


if __name__ == "__main__":
    main()
