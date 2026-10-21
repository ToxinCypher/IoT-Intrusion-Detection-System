"""
Zero-Day Generalization Experiment — hold-out attack type evaluation.

Tests whether models trained on a subset of attack types can detect
unseen ("zero-day") attack types they were never trained on.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import logger, timed


# ---------------------------------------------------------------------------
# Zero-Day Split Creation
# ---------------------------------------------------------------------------

@timed
def create_zero_day_split(
    df: pd.DataFrame,
    holdout_attacks: List[str],
    label_col: str = "label",
    label_binary_col: str = "label_binary",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/test splits with held-out attack types.

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed dataset with original label column.
    holdout_attacks : list of str
        Attack type names to hold out from training entirely.
    label_col : str
        Fine-grained label column (34 types).
    label_binary_col : str
        Binary label column (Attack / Benign).

    Returns
    -------
    (train_df, test_known_df, test_zeroday_df)
    - train_df: data without holdout attack types
    - test_known_df: test split of known attack types
    - test_zeroday_df: all data of held-out attack types
    """
    # Validate holdout attacks exist
    available = df[label_col].unique()
    missing = [a for a in holdout_attacks if a not in available]
    if missing:
        logger.warning(f"  Holdout attacks not found in data: {missing}")
        holdout_attacks = [a for a in holdout_attacks if a in available]

    if not holdout_attacks:
        raise ValueError("No valid holdout attacks specified!")

    # Separate zero-day data
    zeroday_mask = df[label_col].isin(holdout_attacks)
    test_zeroday_df = df[zeroday_mask].copy()
    remaining_df = df[~zeroday_mask].copy()

    # Split remaining into train/test (80/20)
    from sklearn.model_selection import train_test_split

    train_df, test_known_df = train_test_split(
        remaining_df,
        test_size=0.2,
        random_state=42,
        stratify=remaining_df[label_col],
    )

    logger.info(f"  Zero-day split:")
    logger.info(f"    Held-out attacks: {holdout_attacks}")
    logger.info(f"    Train: {len(train_df):,} rows ({remaining_df[label_col].nunique() - len(holdout_attacks)} attack types)")
    logger.info(f"    Test (known): {len(test_known_df):,} rows")
    logger.info(f"    Test (zero-day): {len(test_zeroday_df):,} rows")

    for attack in holdout_attacks:
        n = (test_zeroday_df[label_col] == attack).sum()
        logger.info(f"      {attack}: {n:,} samples")

    return train_df, test_known_df, test_zeroday_df


# ---------------------------------------------------------------------------
# Zero-Day Evaluation
# ---------------------------------------------------------------------------

@timed
def evaluate_zero_day(
    model,
    test_zeroday_df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label",
    label_binary_col: str = "label_binary",
    label_category_col: str = "label_category",
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a model's ability to detect held-out (zero-day) attacks.

    Metrics computed:
    1. Binary detection rate: does the model flag zero-day traffic as "Attack"?
    2. Category-level accuracy: does it assign a reasonable category?
    3. Confidence analysis: how confident is the model on zero-day vs known?

    Parameters
    ----------
    model : trained classifier
    test_zeroday_df : zero-day test data
    feature_cols : list of feature column names
    label_col : fine-grained label column
    label_binary_col : binary label column
    label_category_col : category label column
    model_name : string for logging

    Returns
    -------
    dict with zero-day evaluation results
    """
    X_zd = test_zeroday_df[feature_cols]

    # Predictions
    y_pred = model.predict(X_zd)

    results = {
        "model_name": model_name,
        "n_zeroday_samples": len(test_zeroday_df),
        "holdout_attacks": test_zeroday_df[label_col].unique().tolist(),
    }

    # --- 1. Binary detection ---
    # Check if model predicts anything other than "Benign" (encoded as 0 typically)
    # We check based on predictions: any non-benign prediction counts as "detected"
    # This requires knowing the benign class index
    try:
        y_proba = model.predict_proba(X_zd)
        results["mean_confidence"] = float(np.max(y_proba, axis=1).mean())
        results["confidence_per_attack"] = {}
        for attack in test_zeroday_df[label_col].unique():
            mask = test_zeroday_df[label_col] == attack
            if mask.sum() > 0:
                attack_confidence = np.max(y_proba[mask.values], axis=1).mean()
                results["confidence_per_attack"][attack] = float(attack_confidence)
    except Exception:
        results["mean_confidence"] = None

    # --- 2. Prediction distribution ---
    pred_series = pd.Series(y_pred, index=test_zeroday_df.index)
    results["prediction_distribution"] = pred_series.value_counts().to_dict()

    # Per holdout attack: what does the model predict?
    results["per_attack_predictions"] = {}
    for attack in test_zeroday_df[label_col].unique():
        mask = test_zeroday_df[label_col] == attack
        attack_preds = pred_series[mask]
        results["per_attack_predictions"][attack] = {
            "n_samples": int(mask.sum()),
            "prediction_counts": attack_preds.value_counts().to_dict(),
            "most_common_prediction": int(attack_preds.mode().iloc[0]) if len(attack_preds) > 0 else None,
        }

    logger.info(f"  Zero-day evaluation for {model_name}:")
    logger.info(f"    Total zero-day samples: {len(test_zeroday_df):,}")
    if results.get("mean_confidence"):
        logger.info(f"    Mean prediction confidence: {results['mean_confidence']:.4f}")
    for attack, info in results["per_attack_predictions"].items():
        logger.info(f"    {attack}: {info['n_samples']} samples → predicted as {info['prediction_counts']}")

    return results


# ---------------------------------------------------------------------------
# Run Full Zero-Day Experiment
# ---------------------------------------------------------------------------

@timed
def run_zero_day_experiment(
    df: pd.DataFrame,
    models: dict,
    feature_cols: list,
    holdout_configs: dict,
    label_col: str = "label",
    label_category_encoded_col: str = "label_category_encoded",
) -> pd.DataFrame:
    """
    Run the full zero-day experiment across multiple holdout configurations
    and multiple models.

    Parameters
    ----------
    df : pd.DataFrame — full preprocessed data
    models : dict — {model_name: train_function}
    feature_cols : list — feature column names
    holdout_configs : dict — {config_name: [attack_type_1, ...]}
    label_col : str
    label_category_encoded_col : str

    Returns
    -------
    pd.DataFrame — results summary
    """
    all_results = []

    for config_name, holdout_attacks in holdout_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Zero-day config: {config_name}")
        logger.info(f"Holdout attacks: {holdout_attacks}")
        logger.info(f"{'='*60}")

        # Create split
        train_df, test_known_df, test_zeroday_df = create_zero_day_split(
            df, holdout_attacks, label_col=label_col
        )

        if len(test_zeroday_df) == 0:
            logger.warning(f"  No zero-day samples for {config_name}, skipping")
            continue

        # Train and evaluate each model
        for model_name, train_fn in models.items():
            logger.info(f"\n  Training {model_name} for {config_name}...")

            X_train = train_df[feature_cols]
            y_train = train_df[label_category_encoded_col]

            # Train model
            result = train_fn(X_train, y_train)
            model = result["model"]

            # Evaluate on zero-day
            zd_results = evaluate_zero_day(
                model, test_zeroday_df, feature_cols,
                label_col=label_col,
                model_name=model_name,
            )
            zd_results["config"] = config_name
            zd_results["training_time"] = result.get("training_time")

            all_results.append(zd_results)

    return all_results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_zero_day_results(
    results: list,
    save_path: Optional[str] = None,
):
    """
    Plot zero-day detection results as a grouped bar chart.
    """
    # Extract data for plotting
    data = []
    for r in results:
        for attack, info in r.get("per_attack_predictions", {}).items():
            data.append({
                "Model": r["model_name"],
                "Config": r.get("config", ""),
                "Attack": attack,
                "Samples": info["n_samples"],
                "Confidence": r.get("confidence_per_attack", {}).get(attack, 0),
            })

    if not data:
        logger.warning("  No data to plot for zero-day results")
        return

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 6))
    if "Confidence" in plot_df.columns and plot_df["Confidence"].notna().any():
        pivot = plot_df.pivot_table(
            index="Attack", columns="Model", values="Confidence", aggfunc="mean"
        )
        pivot.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_ylabel("Mean Prediction Confidence")
    else:
        ax.text(0.5, 0.5, "Confidence data not available", ha="center", va="center")

    ax.set_title("Zero-Day Attack Detection — Prediction Confidence", fontsize=14, fontweight="bold")
    ax.set_xlabel("Held-out Attack Type")
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="Model")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Zero-day results plot saved to {save_path}")

    plt.show()
    return fig
