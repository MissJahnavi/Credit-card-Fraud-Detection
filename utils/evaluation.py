"""
utils/evaluation.py
-------------------
Model evaluation utilities for the Credit Card Fraud Detection system.

Provides:
- Confusion matrix visualization
- Classification report
- ROC curve
- Precision-Recall curve (critical for imbalanced datasets)
- Feature importance chart
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Comprehensive Metrics Summary
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """
    Compute all relevant classification metrics for fraud detection.

    For imbalanced datasets, ROC-AUC and Recall are the most critical metrics
    because the cost of missing a fraud (false negative) is very high.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted binary labels
        y_prob: Predicted fraud probabilities

    Returns:
        Dictionary of metric names to values
    """
    return {
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "precision":       round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":          round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":         round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision":   round(average_precision_score(y_true, y_prob), 4),
    }


def print_classification_report(y_true, y_pred, model_name: str = "Model"):
    """Print a detailed classification report to console."""
    print(f"\n{'='*60}")
    print(f"  Classification Report — {model_name}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"]))


# ---------------------------------------------------------------------------
# 2. Confusion Matrix — Matplotlib version (for saving) & Plotly (dashboard)
# ---------------------------------------------------------------------------

def plot_confusion_matrix_mpl(y_true, y_pred, model_name: str = "Model", save_path: str = None):
    """
    Plot confusion matrix using seaborn/matplotlib.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Title label
        save_path: If provided, saves the figure to this path
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Confusion matrix saved to %s", save_path)
    plt.show()
    return fig


def plotly_confusion_matrix(y_true, y_pred, model_name: str = "Model") -> go.Figure:
    """
    Interactive Plotly confusion matrix heatmap for the Streamlit dashboard.

    Returns:
        Plotly Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Normal", "Fraud"]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 18},
        showscale=True,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(side="bottom"),
        font=dict(size=13),
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# 3. ROC Curve
# ---------------------------------------------------------------------------

def plotly_roc_curve(y_true, y_prob, model_name: str = "Model") -> go.Figure:
    """
    Interactive ROC curve using Plotly.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted fraud probabilities
        model_name: Legend label

    Returns:
        Plotly Figure object
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"{model_name} (AUC = {auc:.4f})",
        line=dict(color="#2563EB", width=3),
    ))

    # Diagonal — random classifier baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Baseline",
        line=dict(color="gray", width=2, dash="dash"),
    ))

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        height=430,
        font=dict(size=13),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Precision-Recall Curve
# ---------------------------------------------------------------------------

def plotly_precision_recall_curve(y_true, y_prob, model_name: str = "Model") -> go.Figure:
    """
    Precision-Recall curve — more informative than ROC for imbalanced datasets.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted fraud probabilities
        model_name: Legend label

    Returns:
        Plotly Figure object
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode="lines",
        name=f"{model_name} (AP = {avg_prec:.4f})",
        line=dict(color="#16A34A", width=3),
        fill="tozeroy",
        fillcolor="rgba(22, 163, 74, 0.1)",
    ))

    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision",
        height=430,
        font=dict(size=13),
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Feature Importance
# ---------------------------------------------------------------------------

def plotly_feature_importance(
    feature_names: list,
    importances: np.ndarray,
    top_n: int = 15,
    model_name: str = "Model"
) -> go.Figure:
    """
    Horizontal bar chart of top-N most important features.

    Args:
        feature_names: List of feature column names
        importances: Array of feature importance scores
        top_n: Number of top features to display
        model_name: Chart title label

    Returns:
        Plotly Figure object
    """
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)

    # Reverse for horizontal bar chart readability
    df_imp = df_imp.iloc[::-1]

    fig = go.Figure(go.Bar(
        x=df_imp["Importance"],
        y=df_imp["Feature"],
        orientation="h",
        marker=dict(
            color=df_imp["Importance"],
            colorscale="Viridis",
            showscale=True,
        ),
    ))
    fig.update_layout(
        title=f"Top {top_n} Feature Importances — {model_name}",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500,
        font=dict(size=12),
        margin=dict(l=120),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Model Comparison Table
# ---------------------------------------------------------------------------

def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Build a formatted comparison DataFrame from multiple model results.

    Args:
        results: Dict mapping model_name -> metrics dict

    Returns:
        DataFrame suitable for display
    """
    rows = []
    for name, metrics in results.items():
        rows.append({"Model": name, **metrics})
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("roc_auc", ascending=False)
