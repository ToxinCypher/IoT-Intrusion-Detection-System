"""
Feature engineering — selection, variance filtering, correlation filtering.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold

from src.utils import logger, timed


# ---------------------------------------------------------------------------
# Variance-based filtering
# ---------------------------------------------------------------------------

@timed
def select_features_variance(
    X: pd.DataFrame,
    threshold: float = 0.01,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop features with variance below a threshold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    threshold : float
        Minimum variance to keep a feature.

    Returns
    -------
    (filtered_X, dropped_columns)
    """
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)

    kept_mask = selector.get_support()
    kept_cols = X.columns[kept_mask].tolist()
    dropped_cols = X.columns[~kept_mask].tolist()

    if dropped_cols:
        logger.info(
            f"  Variance filter (threshold={threshold}): "
            f"dropped {len(dropped_cols)} features: {dropped_cols}"
        )
    else:
        logger.info(f"  Variance filter: all {len(kept_cols)} features passed")

    return X[kept_cols], dropped_cols


# ---------------------------------------------------------------------------
# Correlation-based filtering
# ---------------------------------------------------------------------------

@timed
def filter_correlated_features(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove one of each pair of features with Pearson correlation > threshold.

    The feature with lower mean absolute correlation with all other features
    is kept (i.e., we drop the more redundant one).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    threshold : float
        Correlation threshold (default 0.95).

    Returns
    -------
    (filtered_X, dropped_columns)
    """
    corr_matrix = X.corr().abs()

    # Upper triangle (avoid self-correlation and double-counting)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    # Find columns to drop
    to_drop = set()
    for col in upper.columns:
        correlated_features = upper.index[upper[col] > threshold].tolist()
        if correlated_features:
            # Drop the feature with higher mean correlation to all others
            for feat in correlated_features:
                mean_corr_col = corr_matrix[col].mean()
                mean_corr_feat = corr_matrix[feat].mean()
                if mean_corr_feat >= mean_corr_col:
                    to_drop.add(feat)
                else:
                    to_drop.add(col)

    dropped_cols = sorted(to_drop)
    kept_cols = [c for c in X.columns if c not in to_drop]

    logger.info(
        f"  Correlation filter (threshold={threshold}): "
        f"dropped {len(dropped_cols)} features, kept {len(kept_cols)}"
    )
    if dropped_cols:
        logger.debug(f"  Dropped correlated features: {dropped_cols}")

    return X[kept_cols], dropped_cols


# ---------------------------------------------------------------------------
# Mutual Information feature selection
# ---------------------------------------------------------------------------

@timed
def select_features_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    k: int = 30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select top-k features by mutual information with the target.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels (encoded).
    k : int
        Number of top features to select.
    random_state : int
        Seed.

    Returns
    -------
    (filtered_X, mi_scores) — filtered DataFrame and Series of MI scores
    """
    k = min(k, X.shape[1])

    logger.info(f"  Computing mutual information for {X.shape[1]} features...")
    mi_scores = mutual_info_classif(X, y, random_state=random_state, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    top_features = mi_series.head(k).index.tolist()

    logger.info(f"  Selected top-{k} features by MI:")
    for i, (feat, score) in enumerate(mi_series.head(k).items()):
        logger.info(f"    {i + 1:2d}. {feat:35s}  MI={score:.4f}")

    return X[top_features], mi_series


# ---------------------------------------------------------------------------
# Full feature engineering pipeline
# ---------------------------------------------------------------------------

@timed
def run_feature_engineering(
    X: pd.DataFrame,
    y: pd.Series,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    mi_top_k: int = 30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    Run the full feature engineering pipeline.

    1. Variance filter
    2. Correlation filter
    3. Mutual information selection

    Returns
    -------
    (X_selected, info_dict) — filtered features and metadata about the process
    """
    info = {"original_features": X.columns.tolist()}

    # Step 1: Variance filter
    X, dropped_var = select_features_variance(X, threshold=variance_threshold)
    info["dropped_variance"] = dropped_var

    # Step 2: Correlation filter
    X, dropped_corr = filter_correlated_features(X, threshold=correlation_threshold)
    info["dropped_correlation"] = dropped_corr

    # Step 3: Mutual information
    X, mi_scores = select_features_mutual_info(X, y, k=mi_top_k, random_state=random_state)
    info["mi_scores"] = mi_scores
    info["selected_features"] = X.columns.tolist()

    logger.info(
        f"  Feature engineering complete: {len(info['original_features'])} → {X.shape[1]} features"
    )

    return X, info
