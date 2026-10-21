"""
Explainability — SHAP and LIME wrapper functions for IoT IDS models.

Primary contribution: understanding which network flow features drive
each attack classification in the CICIoT2023 dataset.
"""

import os
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import logger, timed


# ---------------------------------------------------------------------------
# SHAP Analysis
# ---------------------------------------------------------------------------

@timed
def compute_shap_values(
    model,
    X_sample: pd.DataFrame,
    model_type: str = "tree",
    max_samples: int = 5000,
    random_state: int = 42,
):
    """
    Compute SHAP values for a trained model.

    Parameters
    ----------
    model : fitted estimator
    X_sample : feature DataFrame (will be subsampled if > max_samples)
    model_type : 'tree' (DT/RF/XGBoost) or 'kernel' (any model)
    max_samples : max rows for SHAP computation
    random_state : seed

    Returns
    -------
    (shap_values, explainer, X_used) — SHAP values array, explainer object,
                                        and the actual X used
    """
    import shap

    # Subsample if needed
    if len(X_sample) > max_samples:
        X_used = X_sample.sample(n=max_samples, random_state=random_state)
        logger.info(f"  Subsampled to {max_samples} rows for SHAP")
    else:
        X_used = X_sample

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        logger.info("  Using TreeExplainer (exact, polynomial time)")
    elif model_type == "kernel":
        background = shap.sample(X_used, min(100, len(X_used)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        logger.info("  Using KernelExplainer (approximate, slower)")
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'tree' or 'kernel'.")

    logger.info("  Computing SHAP values (this may take a few minutes)...")
    shap_values = explainer.shap_values(X_used)

    logger.info(f"  SHAP values computed. Shape: {np.array(shap_values).shape}")
    return shap_values, explainer, X_used


# ---------------------------------------------------------------------------
# Global SHAP Plots
# ---------------------------------------------------------------------------

def plot_shap_summary(
    shap_values,
    X_sample: pd.DataFrame,
    class_names: Optional[list] = None,
    plot_type: str = "dot",
    max_display: int = 20,
    save_path: Optional[str] = None,
):
    """
    Generate SHAP summary plot (beeswarm or bar).

    Parameters
    ----------
    shap_values : SHAP values (list of arrays for multi-class)
    X_sample : feature DataFrame used for SHAP
    class_names : list of class names
    plot_type : 'dot' (beeswarm) or 'bar'
    max_display : number of features to show
    save_path : path to save figure
    """
    import shap

    plt.figure(figsize=(12, 8))

    if isinstance(shap_values, list):
        # Multi-class: summary across all classes
        shap.summary_plot(
            shap_values,
            X_sample,
            plot_type=plot_type,
            max_display=max_display,
            class_names=class_names,
            show=False,
        )
    else:
        shap.summary_plot(
            shap_values,
            X_sample,
            plot_type=plot_type,
            max_display=max_display,
            show=False,
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  SHAP summary plot saved to {save_path}")

    plt.show()


def plot_shap_per_class(
    shap_values: list,
    X_sample: pd.DataFrame,
    class_names: list,
    top_k: int = 10,
    save_dir: Optional[str] = None,
):
    """
    Generate a SHAP summary plot for each attack class individually.

    Parameters
    ----------
    shap_values : list of arrays (one per class)
    X_sample : feature DataFrame
    class_names : list of class names
    top_k : number of top features per class
    save_dir : directory to save individual plots
    """
    import shap

    for i, class_name in enumerate(class_names):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[i],
            X_sample,
            plot_type="dot",
            max_display=top_k,
            show=False,
        )
        plt.title(f"SHAP Feature Importance — {class_name}", fontsize=13)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fpath = os.path.join(save_dir, f"shap_summary_{class_name}.png")
            plt.savefig(fpath, dpi=150, bbox_inches="tight")
            logger.info(f"  Saved {fpath}")

        plt.show()


# ---------------------------------------------------------------------------
# SHAP Feature Importance Table
# ---------------------------------------------------------------------------

def compute_shap_importance_table(
    shap_values: list,
    feature_names: list,
    class_names: list,
    top_k: int = 15,
) -> pd.DataFrame:
    """
    Create a table of mean |SHAP| per feature per class.

    Returns
    -------
    pd.DataFrame with features as rows, classes as columns
    """
    importance_dict = {}
    for i, class_name in enumerate(class_names):
        mean_abs_shap = np.abs(shap_values[i]).mean(axis=0)
        importance_dict[class_name] = mean_abs_shap

    table = pd.DataFrame(importance_dict, index=feature_names)
    table["Overall"] = table.mean(axis=1)
    table = table.sort_values("Overall", ascending=False)

    logger.info(f"  SHAP importance table: {table.shape[0]} features × {table.shape[1]} classes")
    return table.head(top_k)


# ---------------------------------------------------------------------------
# Local SHAP Explanations
# ---------------------------------------------------------------------------

def plot_shap_force(
    explainer,
    shap_values,
    X_sample: pd.DataFrame,
    idx: int,
    class_idx: int = 0,
    save_path: Optional[str] = None,
):
    """
    Generate a SHAP force plot for a single prediction.

    Parameters
    ----------
    explainer : SHAP explainer object
    shap_values : SHAP values (list for multi-class)
    X_sample : feature DataFrame
    idx : row index in X_sample to explain
    class_idx : which class's SHAP values to show (for multi-class)
    save_path : path to save as HTML
    """
    import shap

    if isinstance(shap_values, list):
        sv = shap_values[class_idx][idx]
        ev = explainer.expected_value[class_idx]
    else:
        sv = shap_values[idx]
        ev = explainer.expected_value

    force = shap.force_plot(ev, sv, X_sample.iloc[idx], matplotlib=True, show=False)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Force plot saved to {save_path}")

    plt.show()


def plot_shap_waterfall(
    explainer,
    shap_values,
    X_sample: pd.DataFrame,
    idx: int,
    class_idx: int = 0,
    max_display: int = 15,
    save_path: Optional[str] = None,
):
    """
    Generate a SHAP waterfall plot for a single prediction.
    """
    import shap

    if isinstance(shap_values, list):
        sv = shap_values[class_idx][idx]
        ev = explainer.expected_value[class_idx]
    else:
        sv = shap_values[idx]
        ev = explainer.expected_value

    explanation = shap.Explanation(
        values=sv,
        base_values=ev,
        data=X_sample.iloc[idx].values,
        feature_names=X_sample.columns.tolist(),
    )

    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(explanation, max_display=max_display, show=False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Waterfall plot saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# SHAP Dependence Plots
# ---------------------------------------------------------------------------

def plot_shap_dependence(
    shap_values,
    X_sample: pd.DataFrame,
    feature: str,
    interaction_feature: Optional[str] = "auto",
    class_idx: int = 0,
    save_path: Optional[str] = None,
):
    """
    Generate a SHAP dependence plot for a specific feature.
    """
    import shap

    sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature,
        sv,
        X_sample,
        interaction_index=interaction_feature,
        show=False,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  Dependence plot saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# LIME Analysis
# ---------------------------------------------------------------------------

@timed
def compute_lime_explanations(
    model,
    X_train: pd.DataFrame,
    X_explain: pd.DataFrame,
    class_names: Optional[list] = None,
    num_features: int = 15,
    n_samples: int = 100,
    random_state: int = 42,
) -> list:
    """
    Generate LIME explanations for a set of samples.

    Parameters
    ----------
    model : fitted estimator with predict_proba
    X_train : training data (for LIME's background distribution)
    X_explain : samples to explain
    class_names : list of class names
    num_features : number of features in each explanation
    n_samples : number of samples to explain (subsampled from X_explain)
    random_state : seed

    Returns
    -------
    list of lime.explanation.Explanation objects
    """
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode="classification",
        random_state=random_state,
    )

    # Subsample
    if len(X_explain) > n_samples:
        X_explain = X_explain.sample(n=n_samples, random_state=random_state)

    explanations = []
    for i in range(len(X_explain)):
        exp = explainer.explain_instance(
            X_explain.iloc[i].values,
            model.predict_proba,
            num_features=num_features,
        )
        explanations.append(exp)

        if (i + 1) % 20 == 0:
            logger.info(f"  LIME: explained {i + 1}/{len(X_explain)} samples")

    logger.info(f"  Generated {len(explanations)} LIME explanations")
    return explanations


def compare_shap_lime_rankings(
    shap_importance: pd.Series,
    lime_importance: pd.Series,
    top_k: int = 15,
) -> dict:
    """
    Compare SHAP and LIME feature importance rankings.

    Parameters
    ----------
    shap_importance : Series (feature → mean |SHAP|)
    lime_importance : Series (feature → mean |LIME weight|)
    top_k : number of top features to compare

    Returns
    -------
    dict with 'spearman_rho', 'top_k_overlap', 'rankings_table'
    """
    from scipy.stats import spearmanr

    # Align features
    common = shap_importance.index.intersection(lime_importance.index)
    shap_rank = shap_importance[common].rank(ascending=False)
    lime_rank = lime_importance[common].rank(ascending=False)

    # Spearman correlation
    rho, pvalue = spearmanr(shap_rank, lime_rank)

    # Top-k overlap
    shap_top = set(shap_importance.nlargest(top_k).index)
    lime_top = set(lime_importance.nlargest(top_k).index)
    overlap = shap_top & lime_top

    # Rankings table
    rankings = pd.DataFrame({
        "SHAP_Rank": shap_rank,
        "LIME_Rank": lime_rank,
        "SHAP_Value": shap_importance[common],
        "LIME_Value": lime_importance[common],
    }).sort_values("SHAP_Rank")

    result = {
        "spearman_rho": rho,
        "spearman_pvalue": pvalue,
        "top_k_overlap_count": len(overlap),
        "top_k_overlap_features": sorted(overlap),
        "rankings_table": rankings,
    }

    logger.info(
        f"  SHAP vs LIME: Spearman ρ={rho:.4f} (p={pvalue:.4f}), "
        f"Top-{top_k} overlap: {len(overlap)}/{top_k}"
    )

    return result
