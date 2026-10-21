"""
Evaluation — metrics computation, confusion matrices, comparison tables.
"""

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import logger, timed


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@timed
def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_names: Optional[list] = None,
    model_name: str = "Model",
) -> dict:
    """
    Comprehensive evaluation of a trained classifier.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_test : test features
    y_test : test labels (encoded)
    label_names : list of class names (for display)
    model_name : str for logging

    Returns
    -------
    dict with all computed metrics
    """
    logger.info(f"Evaluating {model_name}...")

    # Predictions
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start
    inference_per_sample_ms = (inference_time / len(X_test)) * 1000

    # Core metrics
    results = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "inference_time_total_s": inference_time,
        "inference_per_sample_ms": inference_per_sample_ms,
        "n_test_samples": len(y_test),
    }

    # Per-class report
    report = classification_report(
        y_test, y_pred, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    results["classification_report"] = report
    results["per_class_f1"] = {
        k: v["f1-score"] for k, v in report.items()
        if k not in ["accuracy", "macro avg", "weighted avg"]
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    results["confusion_matrix"] = cm

    # ROC-AUC (one-vs-rest) — only if model supports predict_proba
    try:
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        results["roc_auc_macro"] = roc_auc
    except Exception:
        results["roc_auc_macro"] = None
        logger.warning(f"  ROC-AUC not available for {model_name}")

    # Model size
    try:
        import joblib
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=True) as tmp:
            joblib.dump(model, tmp.name)
            results["model_size_kb"] = os.path.getsize(tmp.name) / 1024
    except Exception:
        results["model_size_kb"] = None

    # Log summary
    logger.info(
        f"  {model_name}: Acc={results['accuracy']:.4f}, "
        f"F1(macro)={results['macro_f1']:.4f}, "
        f"F1(weighted)={results['weighted_f1']:.4f}, "
        f"Latency={results['inference_per_sample_ms']:.4f} ms/sample"
    )

    return results


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def generate_comparison_table(
    results_list: list,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a comparison table from a list of evaluation result dicts.

    Parameters
    ----------
    results_list : list of dicts from evaluate_model()
    save_path : optional CSV path to save the table

    Returns
    -------
    pd.DataFrame — comparison table
    """
    rows = []
    for r in results_list:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "Macro F1": f"{r['macro_f1']:.4f}",
            "Weighted F1": f"{r['weighted_f1']:.4f}",
            "Macro Precision": f"{r['macro_precision']:.4f}",
            "Macro Recall": f"{r['macro_recall']:.4f}",
            "ROC-AUC": f"{r['roc_auc_macro']:.4f}" if r.get("roc_auc_macro") else "N/A",
            "Latency (ms/sample)": f"{r['inference_per_sample_ms']:.4f}",
            "Model Size (KB)": f"{r['model_size_kb']:.1f}" if r.get("model_size_kb") else "N/A",
        })

    table = pd.DataFrame(rows)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        table.to_csv(save_path, index=False)
        logger.info(f"Comparison table saved to {save_path}")

    return table


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None,
):
    """
    Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray — confusion matrix
    class_names : list of class labels
    title : str
    normalize : bool — show percentages instead of raw counts
    figsize : tuple
    save_path : str, optional — path to save the figure
    """
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.show()
    return fig


def plot_per_class_f1(
    per_class_f1: dict,
    title: str = "Per-Class F1-Score",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
):
    """
    Bar chart of per-class F1 scores.
    """
    classes = list(per_class_f1.keys())
    scores = list(per_class_f1.values())

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(classes)), scores, color="steelblue", edgecolor="white")

    # Color bars below threshold
    for bar, score in zip(bars, scores):
        if score < 0.9:
            bar.set_color("tomato")
        elif score < 0.95:
            bar.set_color("orange")

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("F1-Score")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="0.95 threshold")
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Per-class F1 plot saved to {save_path}")

    plt.show()
    return fig
