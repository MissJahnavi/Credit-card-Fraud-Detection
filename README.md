# 🛡️ Credit Card Fraud Detection & Risk Analysis Platform

A production-grade machine learning system for detecting fraudulent credit card transactions, featuring a real-time interactive analytics dashboard built with Streamlit.

---

## 📌 Problem Statement

Credit card fraud costs the global financial industry billions of dollars annually. With millions of transactions per second, manual fraud review is impossible at scale. This platform applies ensemble machine learning with intelligent class imbalance handling to automate fraud detection with high accuracy.

**Key Challenges:**
- Severe class imbalance (0.17% fraud rate)
- PCA-anonymized features (privacy-preserving)
- Real-time prediction requirements
- High cost of false negatives (missed frauds)

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total Records | 284,807 |
| Fraud Cases | 492 (0.17%) |
| Normal Cases | 284,315 (99.83%) |
| Features | V1–V28 (PCA), Time, Amount, Class |
| Missing Values | None |

**Download the dataset** from Kaggle and place it at `data/creditcard.csv`.

---

## 🤖 Machine Learning Models

| Model | Type | Notes |
|---|---|---|
| **Logistic Regression** | Supervised | Interpretable baseline with L2 regularization |
| **Random Forest** | Supervised Ensemble | 200 trees, provides feature importance |
| **Gradient Boosting** | Supervised Ensemble | Sequential boosting, highest accuracy |
| **Isolation Forest** | Unsupervised | Anomaly detection, no labels required |

**Selection Criterion:** ROC-AUC score (best model auto-selected and saved)

---

## 🏗️ Project Architecture

```
fraud-detection-system/
│
├── data/
│   └── creditcard.csv              ← Kaggle dataset (download separately)
│
├── notebooks/
│   └── eda.ipynb                   ← Exploratory Data Analysis notebook
│
├── models/
│   ├── train_model.py              ← Multi-model training pipeline
│   ├── fraud_model.pkl             ← Saved best model (generated after training)
│   └── model_metadata.json        ← Feature names + performance metrics
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py            ← Data loading, scaling, SMOTE, train/test split
│   └── evaluation.py              ← Metrics computation + Plotly visualizations
│
├── app/
│   └── app.py                      ← Streamlit dashboard (5 sections)
│
├── requirements.txt
└── README.md
```

---

## ⚙️ ML Pipeline

```
Raw CSV
   │
   ▼
Feature Engineering
   • Scale Amount → Amount_Scaled (StandardScaler)
   • Convert Time → Hour (seconds → hour of day)
   • Drop raw Amount and Time columns
   │
   ▼
Stratified Train/Test Split (80/20)
   │
   ▼
SMOTE Oversampling (Training Set Only)
   • Generates synthetic minority-class samples
   • Prevents data leakage to test set
   │
   ▼
Model Training (4 models in parallel)
   │
   ▼
Evaluation (ROC-AUC, Recall, F1, Precision)
   │
   ▼
Best Model Selection → Saved as fraud_model.pkl
```

---

## 🚀 Quick Start

### 1. Clone / Download

```bash
git clone <repo-url>
cd fraud-detection-system
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```
data/creditcard.csv
```

### 4. Run Exploratory Data Analysis (Optional)

```bash
jupyter notebook notebooks/eda.ipynb
```

### 5. Train the Models

```bash
python models/train_model.py --data data/creditcard.csv
```

Expected output:
```
[INFO] Loading dataset...
[INFO] Applying SMOTE...
[INFO] Training: Logistic Regression ...
[INFO] Training: Random Forest ...
[INFO] Training: Gradient Boosting ...
[INFO] Training: Isolation Forest ...

MODEL COMPARISON TABLE (sorted by ROC-AUC)
─────────────────────────────────────────────────────────────────────
                     accuracy  precision  recall  f1_score  roc_auc
Gradient Boosting    0.9997    0.9512     0.8163  0.8788    0.9821
Random Forest        0.9996    0.9701     0.7755  0.8621    0.9754
...
```

### 6. Launch Dashboard

```bash
streamlit run app/app.py
```

Open your browser at: **http://localhost:8501**

---

## 📱 Dashboard Sections

| Section | Description |
|---|---|
| 🏠 **Project Overview** | Architecture, model descriptions, dataset summary |
| 🔮 **Fraud Prediction** | Simulate a transaction and get real-time fraud probability |
| 📊 **Analytics Dashboard** | Interactive charts: class distribution, amount analysis, time patterns |
| 📈 **Model Performance** | Confusion matrix, ROC curve, Precision-Recall curve |
| 🔬 **Feature Importance** | Top predictive features ranked by Gini importance |

---

## 📏 Evaluation Metrics

For imbalanced fraud detection, we prioritize:

1. **Recall (Sensitivity)** — What fraction of actual frauds are caught?
   - Missing a fraud = financial loss → minimize false negatives
   
2. **ROC-AUC** — Model's ability to discriminate across all thresholds
   - Primary model selection criterion

3. **Precision-Recall AUC** — More informative than ROC for heavy imbalance

> ⚠️ **Accuracy is misleading here.** A model that predicts "Normal" for every transaction gets 99.83% accuracy — yet catches zero frauds.

---

## 🧠 Key Design Decisions

### Why SMOTE?
- Random oversampling leads to overfitting (duplicating exact samples)
- SMOTE generates interpolated synthetic fraud samples
- Applied **only to training data** to prevent data leakage

### Why class_weight='balanced'?
- After SMOTE, some residual imbalance may remain
- `class_weight='balanced'` adds a secondary safeguard in the loss function

### Why ROC-AUC for model selection?
- Threshold-independent measure
- Robust to class imbalance
- Captures the full trade-off between TPR and FPR

---

## 📦 Dependencies

```
pandas>=1.5.0          # Data manipulation
numpy>=1.23.0          # Numerical computing
scikit-learn>=1.2.0    # ML models and utilities
imbalanced-learn>=0.10 # SMOTE
matplotlib>=3.6.0      # Static visualizations
seaborn>=0.12.0        # Statistical plots
plotly>=5.13.0         # Interactive charts
streamlit>=1.20.0      # Dashboard framework
joblib>=1.2.0          # Model serialization
```

---

## 📄 License

MIT License — For educational and research purposes.

---

## 🙏 Acknowledgements

- Dataset: [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Original paper: *Calibrated Probability Estimation* by Dal Pozzolo et al.
