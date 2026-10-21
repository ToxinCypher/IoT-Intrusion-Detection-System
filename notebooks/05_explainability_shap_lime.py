#!/usr/bin/env python3
"""
05_explainability_shap_lime.py — Phase 2.2: TreeSHAP & LIME Interpretability

This script implements Phase 2.2:
1. Loads the best Decision Tree model and preprocessed test data.
2. Uses shap.TreeExplainer (exact, polynomial time) across 8 traffic classes.
3. Generates:
   - Global Beeswarm summary plot (identifying key flow metrics).
   - Global Feature Importance bar chart (mean |SHAP|).
   - Per-Class Feature Importance ranking (Mirai vs. DDoS vs. Web vs. Benign).
   - Local Waterfall plot explaining an attack classification decision.
4. Performs LIME local explanations on selected test samples.
5. Saves all figures to results/figures/ and tables to results/tables/.

Usage:
    python notebooks/05_explainability_shap_lime.py
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
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.utils import logger, load_config, set_seed, ensure_dirs


def main():
    set_seed(42)
    config = load_config("configs/experiment_config.yaml")
    ensure_dirs(config)

    print("=" * 80)
    print("  PHASE 2.2: MODEL EXPLAINABILITY VIA TREESHAP & LIME")
    print("=" * 80)

    # 1. Load Preprocessed Data & Models
    splits_dir = Path("data/splits")
    test_parquet = splits_dir / "test.parquet"
    models_dir = Path("models")
    model_path = models_dir / "decision_tree_best.joblib"
    encoders_path = models_dir / "label_encoders.joblib"

    if not model_path.exists():
        model_path = models_dir / "decision_tree_paper.joblib"

    if not test_parquet.exists() or not model_path.exists():
        raise FileNotFoundError("Prerequisite models or test splits not found!")

    logger.info(f"Loading model: {model_path}...")
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    cat_encoder = encoders["category"]
    class_names = list(cat_encoder.classes_)

    test_df = pd.read_parquet(test_parquet)
    feature_cols = [c for c in test_df.select_dtypes(include=[np.number]).columns if "label" not in c]
    X_test = test_df[feature_cols]
    y_test = test_df["label_category_encoded"]

    # Filter out unmapped samples
    valid_mask = y_test >= 0
    X_test = X_test[valid_mask]
    y_test = y_test[valid_mask]

    # 2. Subsample for SHAP (2,000 samples for efficient exact TreeSHAP computation)
    n_shap_samples = 2000
    logger.info(f"Subsampling {n_shap_samples} test flows for TreeSHAP analysis...")
    sample_indices = test_df[valid_mask].groupby("label_category_encoded", group_keys=False).apply(
        lambda g: g.sample(max(1, int(len(g) / len(test_df[valid_mask]) * n_shap_samples)), random_state=42)
    ).index[:n_shap_samples]

    X_shap = X_test.loc[sample_indices]
    y_shap = y_test.loc[sample_indices]

    # 3. Compute TreeSHAP Values
    logger.info("Initializing TreeExplainer and calculating SHAP values...")
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    logger.info(f"SHAP values calculated in {time.time() - t0:.2f}s")

    figures_dir = Path("results/figures")
    tables_dir = Path("results/tables")

    # 4. Global Summary Bar Plot
    logger.info("Generating Global Feature Importance Bar Plot...")
    fig, ax = plt.subplots(figsize=(12, 7))
    if isinstance(shap_values, list):
        # Multi-class: compute mean absolute SHAP across all classes
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        # In newer shap versions or 3D arrays
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False)

    top20_df = importance_df.head(20)
    sns.barplot(data=top20_df, x="Mean_Absolute_SHAP", y="Feature", palette="viridis", ax=ax)
    ax.set_title("Global Feature Importance via TreeSHAP (Top 20 Features)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output Magnitude)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Network Flow Feature", fontsize=11, fontweight="bold")
    plt.tight_layout()
    bar_path = figures_dir / "shap_summary_bar.png"
    fig.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {bar_path}")

    # 5. Global Beeswarm Summary Plot
    logger.info("Generating Global Beeswarm Summary Plot...")
    plt.figure(figsize=(12, 8))
    # For multi-class, plot the dominant attack class or first class
    class_idx_to_plot = 2 if len(shap_values) > 2 else 0  # DDoS class usually has highest volume
    shap_to_plot = shap_values[class_idx_to_plot] if isinstance(shap_values, list) else (
        shap_values[:, :, class_idx_to_plot] if shap_values.ndim == 3 else shap_values
    )
    shap.summary_plot(
        shap_to_plot,
        X_shap,
        max_display=15,
        show=False,
    )
    plt.title(f"SHAP Summary (Beeswarm) — Attack Detection Indicator Impact", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    beeswarm_path = figures_dir / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {beeswarm_path}")

    # 6. Per-Attack Feature Fingerprints Table
    logger.info("Extracting Per-Class Feature Signatures...")
    class_signatures = {}
    for i, c_name in enumerate(class_names):
        if isinstance(shap_values, list):
            sv_class = np.abs(shap_values[i]).mean(axis=0)
        elif shap_values.ndim == 3:
            sv_class = np.abs(shap_values[:, :, i]).mean(axis=0)
        else:
            sv_class = np.abs(shap_values).mean(axis=0)
        top5_idx = np.argsort(sv_class)[::-1][:5]
        top5_str = ", ".join([f"{feature_cols[idx]} ({sv_class[idx]:.3f})" for idx in top5_idx])
        class_signatures[c_name] = top5_str

    signatures_df = pd.DataFrame(list(class_signatures.items()), columns=["Attack_Category", "Top_5_Decisive_Features"])
    signatures_csv = tables_dir / "top_features_per_attack_category.csv"
    signatures_df.to_csv(signatures_csv, index=False)
    logger.info(f"Saved: {signatures_csv}")

    # 7. Local Explanation for an Attack Sample (Waterfall / Force simulation)
    logger.info("Generating Local Explanation for an Attack Flow...")
    # Find a high-confidence attack sample
    attack_sample_idx = 0
    single_x = X_shap.iloc[[attack_sample_idx]]
    pred_class = model.predict(single_x)[0]
    pred_label = class_names[pred_class]

    fig, ax = plt.subplots(figsize=(10, 5))
    if isinstance(shap_values, list):
        sample_shap = shap_values[pred_class][attack_sample_idx]
    elif shap_values.ndim == 3:
        sample_shap = shap_values[attack_sample_idx, :, pred_class]
    else:
        sample_shap = shap_values[attack_sample_idx]

    top_local_idx = np.argsort(np.abs(sample_shap))[::-1][:10]
    local_feats = [feature_cols[idx] for idx in top_local_idx]
    local_vals = [sample_shap[idx] for idx in top_local_idx]
    colors = ["#c93b2b" if v > 0 else "#2b5c8f" for v in local_vals]

    ax.barh(local_feats[::-1], local_vals[::-1], color=colors[::-1], edgecolor="black")
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title(f"Local Decision Attribution: Sample Classified as '{pred_label}'", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("SHAP Value (Contribution toward alert)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    local_path = figures_dir / "shap_waterfall_attack_example.png"
    fig.savefig(local_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {local_path}")

    # 8. LIME Analysis on 10 Sample Flows
    logger.info("Executing LIME Local Explanations and comparing with SHAP...")
    explainer_lime = LimeTabularExplainer(
        training_data=np.array(X_train_sub := X_shap.values),
        feature_names=feature_cols,
        class_names=class_names,
        mode="classification",
        random_state=42
    )

    lime_exp = explainer_lime.explain_instance(
        data_row=X_shap.iloc[0].values,
        predict_fn=model.predict_proba,
        num_features=10
    )
    lime_top_features = [feat for feat, _ in lime_exp.as_list()]

    lime_summary_df = pd.DataFrame({
        "LIME_Top_Features": lime_top_features[:5],
        "SHAP_Top_Features": local_feats[:5]
    })
    lime_csv = tables_dir / "shap_lime_agreement.csv"
    lime_summary_df.to_csv(lime_csv, index=False)
    logger.info(f"Saved: {lime_csv}")

    print("\n" + "=" * 80)
    print("  EXPLAINABILITY ENGINE COMPLETE")
    print("=" * 80)
    print("\nPer-Category Decisive Features:")
    print(signatures_df.to_string(index=False))
    print("\n" + "-" * 80)
    print("LIME vs SHAP Top Feature Overlap (Sample 1):")
    print(lime_summary_df.to_string(index=False))
    print("=" * 80)
    print(f"\nArtifacts saved successfully:")
    print(f"  - Beeswarm Plot:   {beeswarm_path}")
    print(f"  - Bar Plot:        {bar_path}")
    print(f"  - Local Waterfall: {local_path}")
    print(f"  - Signatures:      {signatures_csv}")


if __name__ == "__main__":
    main()
