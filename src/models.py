import time
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.utils import logger, timed


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

@timed
def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[dict] = None,
    cv: int = 5,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> dict:
    """
    Train a Decision Tree with GridSearchCV.

    Returns
    -------
    dict with keys: model, best_params, cv_results, training_time
    """
    if param_grid is None:
        param_grid = {
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        }

    dt = DecisionTreeClassifier(
        random_state=random_state,
        class_weight=class_weight,
    )

    logger.info(f"Training Decision Tree with GridSearchCV ({cv}-fold)...")
    start = time.perf_counter()

    search = GridSearchCV(
        dt, param_grid, cv=cv, scoring="f1_macro",
        n_jobs=-1, verbose=1, refit=True,
    )
    search.fit(X_train, y_train)

    elapsed = time.perf_counter() - start

    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  Best CV F1 (macro): {search.best_score_:.4f}")

    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "cv_results": pd.DataFrame(search.cv_results_),
        "training_time": elapsed,
    }


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

@timed
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_dist: Optional[dict] = None,
    cv: int = 3,
    n_iter: int = 20,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> dict:
    """
    Train a Random Forest with RandomizedSearchCV.

    Returns
    -------
    dict with keys: model, best_params, cv_results, training_time
    """
    if param_dist is None:
        param_dist = {
            "n_estimators": [100, 200, 500],
            "max_depth": [15, 20, None],
            "min_samples_split": [2, 5],
            "max_features": ["sqrt", "log2"],
        }

    rf = RandomForestClassifier(
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )

    logger.info(f"Training Random Forest with RandomizedSearchCV ({cv}-fold, {n_iter} iter)...")
    start = time.perf_counter()

    search = RandomizedSearchCV(
        rf, param_dist, n_iter=n_iter, cv=cv, scoring="f1_macro",
        n_jobs=-1, verbose=1, random_state=random_state, refit=True,
    )
    search.fit(X_train, y_train)

    elapsed = time.perf_counter() - start

    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  Best CV F1 (macro): {search.best_score_:.4f}")

    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "cv_results": pd.DataFrame(search.cv_results_),
        "training_time": elapsed,
    }


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

@timed
def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_dist: Optional[dict] = None,
    cv: int = 3,
    n_iter: int = 15,
    random_state: int = 42,
) -> dict:
    """
    Train XGBoost with RandomizedSearchCV.

    Returns
    -------
    dict with keys: model, best_params, cv_results, training_time
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("xgboost not installed. Run: pip install xgboost")
        raise

    n_classes = y_train.nunique()

    if param_dist is None:
        param_dist = {
            "n_estimators": [100, 200, 500],
            "max_depth": [6, 10, 15],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
        }

    xgb = XGBClassifier(
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )

    logger.info(f"Training XGBoost with RandomizedSearchCV ({cv}-fold, {n_iter} iter)...")
    start = time.perf_counter()

    search = RandomizedSearchCV(
        xgb, param_dist, n_iter=n_iter, cv=cv, scoring="f1_macro",
        n_jobs=-1, verbose=1, random_state=random_state, refit=True,
    )
    search.fit(X_train, y_train)

    elapsed = time.perf_counter() - start

    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  Best CV F1 (macro): {search.best_score_:.4f}")

    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "cv_results": pd.DataFrame(search.cv_results_),
        "training_time": elapsed,
    }


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

@timed
def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[dict] = None,
    cv: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Train Logistic Regression with GridSearchCV.

    Note: Requires scaled features for best performance.
    """
    if param_grid is None:
        param_grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "max_iter": [1000],
            "solver": ["lbfgs"],
        }

    lr = LogisticRegression(
        random_state=random_state,
        class_weight="balanced",
        multi_class="multinomial",
    )

    logger.info(f"Training Logistic Regression with GridSearchCV ({cv}-fold)...")
    start = time.perf_counter()

    search = GridSearchCV(
        lr, param_grid, cv=cv, scoring="f1_macro",
        n_jobs=-1, verbose=1, refit=True,
    )
    search.fit(X_train, y_train)

    elapsed = time.perf_counter() - start

    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  Best CV F1 (macro): {search.best_score_:.4f}")

    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "cv_results": pd.DataFrame(search.cv_results_),
        "training_time": elapsed,
    }


# ---------------------------------------------------------------------------
# K-Nearest Neighbors
# ---------------------------------------------------------------------------

@timed
def train_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[dict] = None,
    cv: int = 5,
) -> dict:
    """
    Train KNN with GridSearchCV.

    Note: Requires scaled features for best performance.
    """
    if param_grid is None:
        param_grid = {
            "n_neighbors": [3, 5, 10],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        }

    knn = KNeighborsClassifier(n_jobs=-1)

    logger.info(f"Training KNN with GridSearchCV ({cv}-fold)...")
    start = time.perf_counter()

    search = GridSearchCV(
        knn, param_grid, cv=cv, scoring="f1_macro",
        n_jobs=-1, verbose=1, refit=True,
    )
    search.fit(X_train, y_train)

    elapsed = time.perf_counter() - start

    logger.info(f"  Best params: {search.best_params_}")
    logger.info(f"  Best CV F1 (macro): {search.best_score_:.4f}")

    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "cv_results": pd.DataFrame(search.cv_results_),
        "training_time": elapsed,
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(model, filepath: str):
    """Save a trained model to disk using joblib."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    size_kb = os.path.getsize(filepath) / 1024
    logger.info(f"  Model saved: {filepath} ({size_kb:.1f} KB)")


def load_model(filepath: str):
    """Load a trained model from disk."""
    model = joblib.load(filepath)
    logger.info(f"  Model loaded: {filepath}")
    return model
