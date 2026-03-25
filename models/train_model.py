"""
models/train_model.py
----------------------
Model training pipeline for Credit Card Fraud Detection.

Trains multiple ML models, compares performance, selects the best,
and persists it to disk using joblib.

Usage:
    python models/train_model.py --data data/creditcard.csv

Models Trained:
    - Logistic Regression      (baseline, fast, interpretable)
    - Random Forest            (ensemble, robust, feature importance)
    - Gradient Boosting        (high performance, slower)
    - Isolation Forest         (anomaly detection, unsupervised baseline)
"""

import os
import sys
import argparse
import logging
import warnings
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)
from sklearn.exceptions import ConvergenceWarning

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.preprocessing import run_preprocessing_pipeline
from utils.evaluation import compute_metrics, print_classification_report, build_comparison_table

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------

def get_supervised_models(fast: bool = False, rf_n_estimators: int = 200, gb_n_estimators: int = 150) -> dict:
    """Return a dictionary of supervised classifiers with tuned hyperparameters.

    All models are configured for:
    - Speed/accuracy trade-off suitable for production use
    - Reproducible results via random_state
    - class_weight='balanced' to handle residual imbalance after SMOTE

    The `fast` flag reduces model complexity for quick iterations.
    """

    if fast:
        # Faster training for local experimentation (less accurate)
        rf_n_estimators = min(rf_n_estimators, 50)
        gb_n_estimators = min(gb_n_estimators, 30)

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=gb_n_estimators,
            learning_rate=0.1,
            max_depth=5,
            max_leaf_nodes=31,
            early_stopping=True,
            random_state=42,
            verbose=1,
        ),
    }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_and_evaluate(models: dict, X_train, y_train, X_test, y_test) -> dict:
    """
    Train each model and collect evaluation metrics.

    Args:
        models: Dict of model_name -> sklearn estimator
        X_train: SMOTE-balanced training features
        y_train: SMOTE-balanced training labels
        X_test: Original (unbalanced) test features
        y_test: Original test labels

    Returns:
        Dict mapping model_name -> {metrics, trained_model, y_pred, y_prob}
    """
    results = {}

    for name, model in models.items():
        logger.info("Training: %s ...", name)

        # Fit
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        print_classification_report(y_test, y_pred, model_name=name)

        results[name] = {
            "model": model,
            "metrics": metrics,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
        logger.info("%s | ROC-AUC: %.4f | Recall: %.4f | F1: %.4f",
                    name, metrics["roc_auc"], metrics["recall"], metrics["f1_score"])

    return results


def train_isolation_forest(X_train_raw, X_test, y_test) -> dict:
    """
    Train Isolation Forest as an unsupervised anomaly detector.

    Note: Isolation Forest doesn't use labels during training.
    It predicts -1 for anomalies and 1 for normal points.
    We remap to 1/0 for compatibility with our binary classification metrics.

    Args:
        X_train_raw: Training features (without SMOTE — unsupervised)
        X_test: Test features
        y_test: True test labels (for evaluation only)

    Returns:
        Dict with metrics and model
    """
    logger.info("Training: Isolation Forest (unsupervised anomaly detection) ...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.001,  # Expected fraud rate ~0.17%
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train_raw)

    # Remap: -1 (anomaly) → 1 (fraud), 1 (normal) → 0
    raw_pred = iso.predict(X_test)
    y_pred = np.where(raw_pred == -1, 1, 0)

    # Score: negative anomaly score (lower = more anomalous → higher fraud prob)
    scores = -iso.score_samples(X_test)
    # Normalize to [0, 1] range
    y_prob = (scores - scores.min()) / (scores.max() - scores.min())

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_classification_report(y_test, y_pred, model_name="Isolation Forest")
    logger.info("Isolation Forest | ROC-AUC: %.4f | Recall: %.4f | F1: %.4f",
                metrics["roc_auc"], metrics["recall"], metrics["f1_score"])

    return {
        "model": iso,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------

def select_best_model(results: dict) -> tuple:
    """
    Select the best model based on ROC-AUC score.

    For fraud detection, ROC-AUC is the primary selection criterion because:
    - It measures discriminative ability across all thresholds
    - It's robust to class imbalance
    - It balances FPR vs TPR trade-off

    Args:
        results: Training results dict

    Returns:
        Tuple of (best_model_name, best_model_object)
    """
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best_model = results[best_name]["model"]
    best_auc = results[best_name]["metrics"]["roc_auc"]
    logger.info("Best model: %s (ROC-AUC = %.4f)", best_name, best_auc)
    return best_name, best_model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, feature_names: list, best_name: str, metrics: dict,
               model_path: str = "models/fraud_model.pkl",
               meta_path: str = "models/model_metadata.json"):
    """
    Persist the trained model and its metadata to disk.

    Saves:
    - fraud_model.pkl: The sklearn estimator (joblib format)
    - model_metadata.json: Feature names, model name, performance metrics

    Args:
        model: Trained sklearn estimator
        feature_names: List of feature column names used during training
        best_name: Name of the selected model
        metrics: Performance metrics dict
        model_path: Output path for the .pkl file
        meta_path: Output path for the metadata JSON
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model binary
    joblib.dump(model, model_path)
    logger.info("Model saved: %s", model_path)

    # Save metadata for dashboard consumption
    metadata = {
        "model_name": best_name,
        "feature_names": feature_names,
        "metrics": metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved: %s", meta_path)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main(
    data_path: str,
    fast: bool = False,
    rf_n_estimators: int = 200,
    gb_n_estimators: int = 150,
    sample_frac: float = 1.0,
):
    """Full training pipeline.

    1. Preprocess data (load → engineer → split → SMOTE)
    2. Train all models
    3. Print comparison table
    4. Select best model (ROC-AUC)
    5. Save model and metadata
    """
    logger.info("=" * 60)
    logger.info("  Credit Card Fraud Detection — Training Pipeline")
    logger.info("=" * 60)

    # Step 1 — Preprocessing
    pipeline = run_preprocessing_pipeline(data_path, sample_frac=sample_frac)
    X_train_res = pipeline["X_train_res"]
    y_train_res = pipeline["y_train_res"]
    X_test      = pipeline["X_test"]
    y_test      = pipeline["y_test"]
    feature_names = pipeline["feature_names"]

    # Keep raw (pre-SMOTE) training data for Isolation Forest
    # We reuse X_train_res without SMOTE labels for unsupervised training
    X_train_raw = X_train_res[y_train_res == 0]  # Only normal transactions

    # Step 2 — Train supervised models
    supervised_models = get_supervised_models(
        fast=fast,
        rf_n_estimators=rf_n_estimators,
        gb_n_estimators=gb_n_estimators,
    )
    results = train_and_evaluate(supervised_models, X_train_res, y_train_res, X_test, y_test)

    # Step 3 — Train Isolation Forest (unsupervised)
    iso_result = train_isolation_forest(X_train_raw, X_test, y_test)
    results["Isolation Forest"] = iso_result

    # Step 4 — Comparison table
    metrics_dict = {name: res["metrics"] for name, res in results.items()}
    comparison_df = build_comparison_table(metrics_dict)
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON TABLE (sorted by ROC-AUC)")
    print("=" * 70)
    print(comparison_df.to_string())
    print("=" * 70)

    # Step 5 — Select best supervised model
    supervised_results = {k: v for k, v in results.items() if k != "Isolation Forest"}
    best_name, best_model = select_best_model(supervised_results)

    # Step 6 — Save
    best_metrics = results[best_name]["metrics"]
    save_model(best_model, feature_names, best_name, best_metrics)

    logger.info("Training pipeline complete.")
    logger.info("Best model: %s | ROC-AUC: %.4f", best_name, best_metrics["roc_auc"])

    return results, best_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/creditcard.csv",
        help="Path to the creditcard.csv dataset",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a faster (lower-complexity) training cycle for development/debugging",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=200,
        help="Number of trees to use for the Random Forest model",
    )
    parser.add_argument(
        "--gb-estimators",
        type=int,
        default=150,
        help="Number of boosting iterations to use for the Gradient Boosting model",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        help="Fraction of training data to use (1.0 = full). Useful for fast dev runs",
    )

    args = parser.parse_args()
    main(
        args.data,
        fast=args.fast,
        rf_n_estimators=args.rf_estimators,
        gb_n_estimators=args.gb_estimators,
        sample_frac=args.sample_frac,
    )
