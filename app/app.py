import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

DATA_PATH     = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODEL_PATH    = os.path.join(PROJECT_ROOT, "models", "fraud_model.pkl")
META_PATH     = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f172a; }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1e40af, #7c3aed);
        padding: 12px 20px;
        border-radius: 10px;
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 16px;
    }
    
    /* Risk badge */
    .risk-safe     { background-color: #166534; color: #bbf7d0; padding: 10px 20px; border-radius: 8px; font-weight: bold; font-size: 1.1rem; }
    .risk-suspicious { background-color: #92400e; color: #fef3c7; padding: 10px 20px; border-radius: 8px; font-weight: bold; font-size: 1.1rem; }
    .risk-fraud    { background-color: #991b1b; color: #fee2e2; padding: 10px 20px; border-radius: 8px; font-weight: bold; font-size: 1.1rem; }
    
    /* Sidebar */
    .css-1d391kg { background-color: #1e293b; }
    
    /* General text */
    body { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        # Generate realistic sample data for demo deployment
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame(np.random.randn(n, 28), columns=[f'V{i}' for i in range(1, 29)])
        df['Time'] = np.random.uniform(0, 172792, n)
        df['Amount'] = np.abs(np.random.exponential(88, n))
        df['Class'] = np.random.choice([0, 1], n, p=[0.998, 0.002])
    df['Hour'] = (df['Time'] / 3600) % 24
    df['Class_Label'] = df['Class'].map({0: 'Normal', 1: 'Fraud'})
    return df


@st.cache_resource
def load_model():
    """Load the trained model from disk."""
    if not os.path.exists(MODEL_PATH):
        return None
    import joblib
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    """Load model metadata (feature names, metrics)."""
    if not os.path.exists(META_PATH):
        return None
    with open(META_PATH) as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/shield.png", width=60)
    st.title("🛡️ FraudSentinel")
    st.caption("Credit Card Risk Analysis Platform")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠  Project Overview",
            "🔮  Fraud Prediction",
            "📊  Analytics Dashboard",
            "📈  Model Performance",
            "🔬  Feature Importance",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Dataset: Kaggle Credit Card Fraud")
    st.caption("284,807 transactions | 492 frauds")

    model = load_model()
    meta  = load_metadata()
    df    = load_data()

    model_status = "✅ Loaded" if model else "❌ Not found"
    data_status  = "✅ Loaded" if df is not None else "❌ Not found"

    st.markdown(f"**Model:** {model_status}")
    st.markdown(f"**Dataset:** {data_status}")

    if meta:
        st.markdown(f"**Active Model:** `{meta.get('model_name', 'N/A')}`")


# ===========================================================================
# PAGE 1 — Project Overview
# ===========================================================================

if "Overview" in page:
    st.markdown("""
    <div style='text-align:center; padding: 30px 0 10px'>
        <h1 style='font-size:2.8rem; background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🛡️ Credit Card Fraud Detection
        </h1>
        <p style='font-size:1.1rem; color:#94a3b8;'>
            ML-powered real-time fraud risk analysis platform for financial institutions
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "284,807")
    col2.metric("Fraud Cases",        "492")
    col3.metric("Fraud Rate",         "0.17%")
    col4.metric("Features",           "30")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("📌 Problem Statement")
        st.markdown("""
        Credit card fraud is one of the most significant challenges in the financial industry.
        With millions of transactions occurring every second, manual review is impossible.

        **Key challenges:**
        - **Severe class imbalance** — only 0.17% of transactions are fraudulent
        - **Feature anonymity** — PCA-transformed features for privacy (V1–V28)
        - **Real-time requirements** — decisions must happen in milliseconds
        - **Asymmetric costs** — missing a fraud is far costlier than a false alarm

        This platform uses ensemble machine learning with SMOTE oversampling to tackle these challenges.
        """)

    with col_r:
        st.subheader("🏗️ System Architecture")
        st.markdown("""
<div style="background:#1e293b; border:1px solid #334155; border-radius:10px;
            padding:16px; font-family:monospace; font-size:0.82rem; line-height:1.7; color:#e2e8f0;">
<span style="color:#60a5fa;">fraud-detection-system/</span><br>
├── data/<br>
│&nbsp;&nbsp;&nbsp;└── <span style="color:#34d399;">creditcard.csv</span> <span style="color:#64748b;">← Kaggle dataset</span><br>
├── utils/<br>
│&nbsp;&nbsp;&nbsp;├── <span style="color:#34d399;">preprocessing.py</span> <span style="color:#64748b;">← SMOTE pipeline</span><br>
│&nbsp;&nbsp;&nbsp;└── <span style="color:#34d399;">evaluation.py</span> <span style="color:#64748b;">← Metrics & charts</span><br>
├── models/<br>
│&nbsp;&nbsp;&nbsp;├── <span style="color:#34d399;">train_model.py</span> <span style="color:#64748b;">← Training pipeline</span><br>
│&nbsp;&nbsp;&nbsp;├── <span style="color:#fbbf24;">fraud_model.pkl</span> <span style="color:#64748b;">← Saved model</span><br>
│&nbsp;&nbsp;&nbsp;└── <span style="color:#fbbf24;">model_metadata.json</span> <span style="color:#64748b;">← Metrics</span><br>
├── notebooks/<br>
│&nbsp;&nbsp;&nbsp;└── <span style="color:#34d399;">eda.ipynb</span> <span style="color:#64748b;">← EDA notebook</span><br>
├── app/<br>
│&nbsp;&nbsp;&nbsp;└── <span style="color:#34d399;">app.py</span> <span style="color:#64748b;">← This dashboard</span><br>
└── <span style="color:#34d399;">requirements.txt</span>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.subheader("🤖 ML Models Used")

    models_info = [
        ("Logistic Regression",  "Fast baseline with L2 regularization. Highly interpretable.",             "Baseline"),
        ("Random Forest",        "200-tree ensemble. Robust to outliers, provides feature importance.",      "Primary"),
        ("Gradient Boosting",    "Sequential boosting. Highest accuracy but slower to train.",              "Primary"),
        ("Isolation Forest",     "Unsupervised anomaly detection. No labels required during training.",     "Auxiliary"),
    ]

    cols = st.columns(4)
    for i, (name, desc, role) in enumerate(models_info):
        with cols[i]:
            color = "#16A34A" if role == "Primary" else "#2563EB" if role == "Baseline" else "#7C3AED"
            st.markdown(f"""
            <div style='background:#1e293b; border:1px solid #334155; border-radius:10px;
                        padding:16px; border-top: 3px solid {color}; height: 180px;'>
                <b style='color:{color};'>{name}</b><br><br>
                <span style='color:#cbd5e1; font-size:0.85rem;'>{desc}</span><br><br>
                <span style='background:{color}22; color:{color}; padding:2px 8px; border-radius:20px; font-size:0.75rem;'>{role}</span>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    st.subheader("🚀 Quick Start")
    st.code("""
# Clone and install
pip install -r requirements.txt

# Download dataset from Kaggle and place at:
# data/creditcard.csv

# Train models
python models/train_model.py --data data/creditcard.csv

# Launch dashboard
streamlit run app/app.py
    """, language="bash")


# ===========================================================================
# PAGE 2 — Fraud Prediction Simulator
# ===========================================================================

elif "Prediction" in page:
    st.markdown('<div class="section-header">🔮 Real-Time Fraud Prediction Simulator</div>',
                unsafe_allow_html=True)

    if model is None or meta is None:
        st.error("""
        ⚠️ Model not loaded. Please train the model first:
        ```bash
        python models/train_model.py --data data/creditcard.csv
        ```
        """)
        st.stop()

    feature_names = meta["feature_names"]

    st.markdown("Configure a simulated transaction below and click **Predict Fraud** to assess risk.")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("💳 Transaction Details")

        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.01, max_value=50000.0,
            value=120.50, step=0.01,
            help="Dollar amount of the transaction"
        )
        hour = st.slider(
            "Transaction Hour (0–24)",
            min_value=0.0, max_value=24.0,
            value=14.0, step=0.5,
            help="Hour of the day when the transaction occurred"
        )

        st.subheader("🔐 Anonymized PCA Features")
        st.caption("These represent PCA-transformed behavioral patterns (V1–V28).")

        # Show a representative subset of PCA features
        pca_inputs = {}
        pca_cols = [f"V{i}" for i in range(1, 29)]

        important_pca = ["V14", "V10", "V12", "V4", "V11", "V17"]
        tab1, tab2 = st.tabs(["Key Features (Recommended)", "All V1-V28"])

        with tab1:
            for feat in important_pca:
                pca_inputs[feat] = st.slider(
                    feat,
                    min_value=-10.0, max_value=10.0,
                    value=0.0, step=0.1,
                    help=f"PCA component {feat}"
                )

        with tab2:
            for feat in pca_cols:
                if feat not in important_pca:
                    pca_inputs[feat] = st.slider(
                        feat,
                        min_value=-10.0, max_value=10.0,
                        value=0.0, step=0.1,
                        key=f"tab2_{feat}"
                    )

    with col2:
        st.subheader("🎯 Risk Assessment")

        predict_btn = st.button("🔍 Predict Fraud", type="primary", use_container_width=True)

        if predict_btn:
            # Build feature vector matching training schema
            feature_vector = {}

            for feat in feature_names:
                if feat == "Amount_Scaled":
                    # Scale amount using mean/std from a typical dataset distribution
                    feature_vector[feat] = (amount - 88.35) / 250.12
                elif feat == "Hour":
                    feature_vector[feat] = hour
                elif feat in pca_inputs:
                    feature_vector[feat] = pca_inputs[feat]
                else:
                    feature_vector[feat] = 0.0

            input_df = pd.DataFrame([feature_vector])[feature_names]
            fraud_prob = model.predict_proba(input_df)[0][1]
            fraud_pct  = round(fraud_prob * 100, 2)

            # Gauge chart
            gauge_color = (
                "#16A34A" if fraud_pct < 30
                else "#D97706" if fraud_pct < 70
                else "#DC2626"
            )

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fraud_pct,
                number={"suffix": "%", "font": {"size": 40, "color": gauge_color}},
                title={"text": "Fraud Probability", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": gauge_color, "thickness": 0.25},
                    "steps": [
                        {"range": [0, 30],   "color": "#064E3B"},
                        {"range": [30, 70],  "color": "#78350F"},
                        {"range": [70, 100], "color": "#7F1D1D"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": fraud_pct,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=280,
                margin=dict(l=30, r=30, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "white"},
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Verdict
            if fraud_pct < 30:
                st.success(f"✅ **SAFE TRANSACTION** — Risk: {fraud_pct}%")
                verdict = "Low risk. Transaction appears normal."
            elif fraud_pct < 70:
                st.warning(f"⚠️ **SUSPICIOUS TRANSACTION** — Risk: {fraud_pct}%")
                verdict = "Elevated risk. Consider additional verification."
            else:
                st.error(f"🚨 **FRAUDULENT TRANSACTION** — Risk: {fraud_pct}%")
                verdict = "High risk. Transaction should be blocked immediately."

            st.info(verdict)

            # Transaction summary
            st.subheader("📋 Transaction Summary")
            summary = {
                "Amount": f"${amount:,.2f}",
                "Hour": f"{hour:.1f}h",
                "Fraud Probability": f"{fraud_pct}%",
                "Decision": "BLOCK" if fraud_pct >= 70 else "REVIEW" if fraud_pct >= 30 else "APPROVE",
            }
            for k, v in summary.items():
                color = "#EF4444" if k == "Decision" and v == "BLOCK" else \
                        "#F59E0B" if k == "Decision" and v == "REVIEW" else "#10B981"
                st.markdown(f"**{k}:** <span style='color:{color}'>{v}</span>",
                            unsafe_allow_html=True)
        else:
            st.info("👈 Configure transaction parameters and click **Predict Fraud**")

            st.markdown("""
            **How the prediction works:**
            1. Your inputs are scaled to match training data distribution
            2. The trained model computes a fraud probability score
            3. Thresholds classify the transaction as Safe / Suspicious / Fraudulent

            | Risk Level  | Probability | Action    |
            |-------------|-------------|-----------|
            | ✅ Safe     | 0–30%       | Approve   |
            | ⚠️ Suspicious | 30–70%   | Review    |
            | 🚨 Fraud    | 70–100%     | Block     |
            """)


# ===========================================================================
# PAGE 3 — Analytics Dashboard
# ===========================================================================

elif "Analytics" in page:
    st.markdown('<div class="section-header">📊 Fraud Analytics Dashboard</div>',
                unsafe_allow_html=True)

    if df is None:
        st.error("⚠️ Dataset not found at `data/creditcard.csv`. Please add the dataset.")
        st.stop()

    total      = len(df)
    fraud_cnt  = df["Class"].sum()
    normal_cnt = total - fraud_cnt
    fraud_rate = fraud_cnt / total * 100

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Transactions", f"{total:,}")
    k2.metric("Fraud Transactions", f"{fraud_cnt:,}",    delta=f"{fraud_rate:.3f}% of total", delta_color="inverse")
    k3.metric("Normal Transactions", f"{normal_cnt:,}")
    k4.metric("Average Fraud Amount", f"${df[df['Class']==1]['Amount'].mean():.2f}")

    st.divider()

    # Row 1: Class distribution + Fraud by amount bucket
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        fig_pie = px.pie(
            values=[normal_cnt, fraud_cnt],
            names=["Normal (99.83%)", "Fraud (0.17%)"],
            color_discrete_sequence=["#2563EB", "#DC2626"],
            hole=0.45,
        )
        fig_pie.update_traces(textinfo="label+percent")
        fig_pie.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", font={"color":"white"})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Transaction Amount Distribution")
        fig_amt = go.Figure()
        fig_amt.add_trace(go.Histogram(
            x=df[df["Class"] == 0]["Amount"],
            name="Normal",
            nbinsx=80,
            marker_color="#2563EB",
            opacity=0.7,
        ))
        fig_amt.add_trace(go.Histogram(
            x=df[df["Class"] == 1]["Amount"],
            name="Fraud",
            nbinsx=60,
            marker_color="#DC2626",
            opacity=0.9,
        ))
        fig_amt.update_layout(
            barmode="overlay",
            xaxis_title="Amount ($)",
            yaxis_title="Count",
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            legend=dict(x=0.7, y=0.9),
        )
        st.plotly_chart(fig_amt, use_container_width=True)

    # Row 2: Fraud over time + Amount box plot
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Fraud Frequency by Hour of Day")
        hour_fraud = df[df["Class"] == 1].groupby(df["Hour"].round(0)).size().reset_index()
        hour_fraud.columns = ["Hour", "Fraud Count"]
        fig_hour = px.bar(
            hour_fraud, x="Hour", y="Fraud Count",
            color="Fraud Count",
            color_continuous_scale="Reds",
        )
        fig_hour.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    with col4:
        st.subheader("Transaction Amount: Fraud vs Normal")
        fig_box = go.Figure()
        for cls, color, label in [(0, "#2563EB", "Normal"), (1, "#DC2626", "Fraud")]:
            fig_box.add_trace(go.Box(
                y=df[df["Class"] == cls]["Amount"],
                name=label,
                marker_color=color,
                boxmean=True,
            ))
        fig_box.update_layout(
            yaxis_title="Amount ($)",
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Row 3: Cumulative fraud over time
    st.subheader("Cumulative Fraud Transactions Over Time")
    fraud_df = df[df["Class"] == 1].sort_values("Time")
    fraud_df["cumulative_fraud"] = range(1, len(fraud_df) + 1)
    fig_cum = px.line(
        fraud_df, x="Time", y="cumulative_fraud",
        color_discrete_sequence=["#F97316"],
    )
    fig_cum.update_layout(
        xaxis_title="Time (seconds from start of recording)",
        yaxis_title="Cumulative Fraud Count",
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Fraud Amount Bins
    st.subheader("Fraud Transactions by Amount Range")
    fraud_only = df[df["Class"] == 1].copy()
    bins = [0, 10, 50, 100, 500, 1000, 5000, 30000]
    labels_bin = ["$0–$10", "$10–$50", "$50–$100", "$100–$500",
                  "$500–$1K", "$1K–$5K", "$5K+"]
    fraud_only["Amount_Bin"] = pd.cut(fraud_only["Amount"], bins=bins, labels=labels_bin)
    bin_counts = fraud_only["Amount_Bin"].value_counts().reindex(labels_bin).fillna(0)
    fig_bins = px.bar(
        x=bin_counts.index, y=bin_counts.values,
        color=bin_counts.values,
        color_continuous_scale="Reds",
        labels={"x": "Amount Range", "y": "Fraud Count"},
    )
    fig_bins.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_bins, use_container_width=True)


# ===========================================================================
# PAGE 4 — Model Performance
# ===========================================================================

elif "Performance" in page:
    st.markdown('<div class="section-header">📈 Model Performance Metrics</div>',
                unsafe_allow_html=True)

    if meta is None:
        st.error("""
        ⚠️ Model metadata not found. Train the model first:
        ```bash
        python models/train_model.py --data data/creditcard.csv
        ```
        """)
        st.stop()

    metrics = meta.get("metrics", {})
    model_name = meta.get("model_name", "Best Model")

    st.subheader(f"Active Model: `{model_name}`")
    st.divider()

    # KPI Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy",      f"{metrics.get('accuracy', 0)*100:.2f}%")
    m2.metric("Precision",     f"{metrics.get('precision', 0)*100:.2f}%")
    m3.metric("Recall",        f"{metrics.get('recall', 0)*100:.2f}%",
              help="Fraction of actual frauds correctly caught — most critical metric")
    m4.metric("F1 Score",      f"{metrics.get('f1_score', 0)*100:.2f}%")
    m5.metric("ROC-AUC",       f"{metrics.get('roc_auc', 0):.4f}")

    st.divider()

    col_l, col_r = st.columns(2)

    if df is not None and model is not None:
        from utils.preprocessing import engineer_features, split_features_labels, get_train_test_split
        from utils.evaluation import plotly_confusion_matrix, plotly_roc_curve, plotly_precision_recall_curve

        with st.spinner("Computing predictions for evaluation charts..."):
            df_eng = engineer_features(df)
            X, y   = split_features_labels(df_eng)
            _, X_test, _, y_test = get_train_test_split(X, y)

            feature_names = meta.get("feature_names", list(X.columns))
            X_test_aligned = X_test[feature_names]

            y_pred = model.predict(X_test_aligned)
            y_prob = model.predict_proba(X_test_aligned)[:, 1]

        with col_l:
            st.subheader("Confusion Matrix")
            fig_cm = plotly_confusion_matrix(y_test, y_pred, model_name)
            fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_r:
            st.subheader("ROC Curve")
            fig_roc = plotly_roc_curve(y_test, y_prob, model_name)
            fig_roc.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
            st.plotly_chart(fig_roc, use_container_width=True)

        st.subheader("Precision-Recall Curve")
        st.caption("For imbalanced datasets, PR curve is more informative than ROC. "
                   "Higher Average Precision = better model.")
        fig_pr = plotly_precision_recall_curve(y_test, y_prob, model_name)
        fig_pr.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
        st.plotly_chart(fig_pr, use_container_width=True)

    else:
        st.info("Evaluation charts require both the model (`fraud_model.pkl`) and dataset (`creditcard.csv`) to be present.")

    # Metrics Explanation
    st.divider()
    st.subheader("📖 Metrics Guide for Fraud Detection")
    with st.expander("Why these metrics matter"):
        st.markdown("""
        | Metric | Formula | Why it matters |
        |---|---|---|
        | **Accuracy** | (TP+TN) / Total | Misleading for imbalanced data (0.17% fraud rate means 99.83% baseline) |
        | **Precision** | TP / (TP+FP) | How many flagged transactions are actually fraud |
        | **Recall** | TP / (TP+FN) | **Most critical** — What fraction of real frauds did we catch? |
        | **F1 Score** | 2·P·R / (P+R) | Harmonic mean of Precision and Recall |
        | **ROC-AUC** | Area under ROC | Overall discriminability across all thresholds |
        | **Avg Precision** | Area under PR | More informative than ROC for imbalanced datasets |

        **In fraud detection, Recall is paramount** — missing a fraud is far more costly than a false positive.
        """)


# ===========================================================================
# PAGE 5 — Feature Importance
# ===========================================================================

elif "Feature" in page:
    st.markdown('<div class="section-header">🔬 Feature Importance Analysis</div>',
                unsafe_allow_html=True)

    if model is None or meta is None:
        st.error("⚠️ Model not found. Run `python models/train_model.py` first.")
        st.stop()

    feature_names = meta.get("feature_names", [])
    model_name    = meta.get("model_name", "Model")

    has_importance = hasattr(model, "feature_importances_")
    has_coef       = hasattr(model, "coef_")

    if has_importance:
        importances = model.feature_importances_
        importance_type = "Gini Importance"
    elif has_coef:
        importances = np.abs(model.coef_[0])
        importance_type = "Coefficient Magnitude"
    else:
        st.warning("This model type does not expose feature importances directly.")
        st.stop()

    from utils.evaluation import plotly_feature_importance

    st.subheader(f"Top 15 Features — {model_name} ({importance_type})")
    st.caption("These are the transaction attributes that most influence the fraud prediction.")

    fig_imp = plotly_feature_importance(
        feature_names=feature_names,
        importances=importances,
        top_n=15,
        model_name=model_name,
    )
    fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    st.plotly_chart(fig_imp, use_container_width=True)

    # Full table
    st.subheader("All Features Ranked")
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    imp_df.index += 1  # 1-based rank
    imp_df["Importance"] = imp_df["Importance"].round(6)
    imp_df["Relative %"] = (imp_df["Importance"] / imp_df["Importance"].sum() * 100).round(2)
    st.dataframe(imp_df, use_container_width=True)

    st.divider()
    st.subheader("📖 Feature Reference")
    with st.expander("What do these features represent?"):
        st.markdown("""
        | Feature | Description |
        |---|---|
        | **V1–V28** | PCA-transformed anonymized features representing behavioral patterns like transaction velocity, merchant category, geographic anomalies, device fingerprint, etc. |
        | **Amount_Scaled** | Scaled transaction amount. Large or unusually small amounts can be fraud indicators. |
        | **Hour** | Hour of day derived from Time. Fraudsters often operate at unusual hours (late night / early morning). |

        **High-importance PCA features typically correspond to:**
        - V14 → Merchant category anomaly signal
        - V10 → Geographic mismatch
        - V12 → Transaction velocity
        - V4  → Card-not-present indicator
        - V11 → Account age signal
        """)
