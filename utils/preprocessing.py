"""
utils/preprocessing.py
-----------------------
Data preprocessing pipeline for Credit Card Fraud Detection.

Responsibilities:
- Load the raw dataset
- Feature engineering and scaling
- Handle severe class imbalance using SMOTE
- Train/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Dataset Loading
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the credit card fraud CSV dataset.

    Args:
        filepath: Path to creditcard.csv

    Returns:
        Raw DataFrame
    """
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Fraud cases: {df['Class'].sum()} / {len(df)} ({df['Class'].mean()*100:.4f}%)")
    return df


# ---------------------------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering on the raw dataset.

    - Scale 'Amount' using StandardScaler (V1–V28 are already PCA-transformed)
    - Scale 'Time' to hours for interpretability
    - Drop original 'Amount' column after scaling

    Args:
        df: Raw DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # Scale the 'Amount' feature — PCA features are already scaled
    scaler = StandardScaler()
    df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])

    # Convert Time (seconds) to hours for readability
    df["Hour"] = (df["Time"] / 3600) % 24

    # Drop raw Amount and Time — replaced by scaled versions
    df.drop(columns=["Amount", "Time"], inplace=True)

    logger.info("Feature engineering complete. Final shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 3. Feature/Label Separation
# ---------------------------------------------------------------------------

def split_features_labels(df: pd.DataFrame):
    """
    Separate features (X) from the target label (y).

    Args:
        df: Preprocessed DataFrame

    Returns:
        X (features), y (labels)
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]
    logger.info("Features: %d columns | Positive (fraud) samples: %d", X.shape[1], y.sum())
    return X, y


# ---------------------------------------------------------------------------
# 4. Train/Test Split
# ---------------------------------------------------------------------------

def get_train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split to preserve class ratio.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        random_state: Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,         # preserve fraud ratio in both splits
        random_state=random_state
    )
    logger.info(
        "Train: %d samples (fraud: %d) | Test: %d samples (fraud: %d)",
        len(X_train), y_train.sum(), len(X_test), y_test.sum()
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. SMOTE Oversampling
# ---------------------------------------------------------------------------

def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE to handle extreme class imbalance on the training set only.

    SMOTE (Synthetic Minority Oversampling TEchnique) generates synthetic
    fraud samples by interpolating between existing minority-class neighbors.

    IMPORTANT: SMOTE is applied ONLY to training data to prevent data leakage.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Seed for reproducibility

    Returns:
        X_resampled, y_resampled — balanced training set
    """
    logger.info("Applying SMOTE. Original distribution: %s", dict(y_train.value_counts()))
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info("After SMOTE: %d samples | Fraud: %d | Normal: %d",
                len(X_resampled), y_resampled.sum(), (y_resampled == 0).sum())
    return X_resampled, y_resampled


# ---------------------------------------------------------------------------
# 6. Full Pipeline (convenience function)
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(filepath: str, test_size: float = 0.2, sample_frac: float = 1.0, random_state: int = 42):
    """End-to-end preprocessing pipeline.

    Steps:
      1. Load dataset
      2. Engineer features (scale Amount, convert Time)
      3. Split features and labels
      4. Train/test split (stratified)
      5. Optional sample down training data (before SMOTE)
      6. Apply SMOTE to training data

    Args:
        filepath: Path to creditcard.csv
        test_size: Proportion of dataset for test split
        sample_frac: Fraction of training data to use (1.0 = full). Useful for quick iterations.
        random_state: Seed for reproducibility

    Returns:
        dict with keys:
            X_train_res, y_train_res  — SMOTE-balanced training data
            X_test, y_test            — Original (unbalanced) test data
            feature_names             — List of feature column names
            df                        — Full preprocessed DataFrame
    """
    df = load_dataset(filepath)
    df = engineer_features(df)
    X, y = split_features_labels(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Optionally subsample training set for faster experimentation
    if sample_frac < 1.0:
        logger.info("Subsampling training data to %.1f%% of original", sample_frac * 100)
        X_train = X_train.sample(frac=sample_frac, random_state=random_state)
        y_train = y_train.loc[X_train.index]
        logger.info("After subsampling: %d training samples (fraud: %d)", len(X_train), y_train.sum())

    X_train_res, y_train_res = apply_smote(X_train, y_train)

    return {
        "X_train_res": X_train_res,
        "y_train_res": y_train_res,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "df": df,
    }
