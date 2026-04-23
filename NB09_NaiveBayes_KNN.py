# Databricks notebook source
import subprocess
packages = ["imbalanced-learn", "networkx"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"✅ {pkg}" if result.returncode == 0 else f"❌ {pkg}: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE — NB09: Naive Bayes + KNN
# Cell 1: Banner + Setup
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║     FRAUDSENSE — NB09: Naive Bayes + KNN                    ║
║     Classical ML Battle Royale — Round 2                    ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

GOLD_PATH   = "/Volumes/workspace/default/fraud_data/gold"
MODELS_PATH = "/Volumes/workspace/default/fraud_data/models"

mlflow.set_experiment("/FRAUDSENSE_Master_Experiment")

plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor']   = '#161b22'
plt.rcParams['axes.edgecolor']   = '#30363d'
plt.rcParams['text.color']       = '#e6edf3'
plt.rcParams['axes.labelcolor']  = '#e6edf3'
plt.rcParams['xtick.color']      = '#e6edf3'
plt.rcParams['ytick.color']      = '#e6edf3'
plt.rcParams['grid.color']       = '#21262d'
plt.rcParams['font.family']      = 'monospace'

FRAUD_COLOR  = '#ff4444'
LEGIT_COLOR  = '#00d4aa'
ACCENT_COLOR = '#f7931a'

print("✅ All libraries loaded")
print("✅ MLflow experiment set")
print("\n⚔️  Classical ML Battle Royale — Round 2 starting...")

# COMMAND ----------

# ============================================================
# Cell 2: Load Gold Data + SMOTE Balancing
# ============================================================

print("📥 Loading CreditCard Gold dataset...")

cc = spark.read.format("delta").load(f"{GOLD_PATH}/creditcard").toPandas()

# ── Feature selection ─────────────────────────────────────────
v_cols       = [c for c in cc.columns if c.startswith('V')]
eng_feats    = ['amount_log', 'amount_zscore', 'amount_spike',
                'is_night', 'tx_velocity_10', 'high_amount_flag',
                'v1_v2_interaction', 'v3_v4_interaction', 'v14_v17_interaction',
                'v_sum_top5', 'v_abs_sum']
eng_feats    = [f for f in eng_feats if f in cc.columns]
feature_cols = v_cols + eng_feats

X = cc[feature_cols].fillna(0).values
y = cc['Class'].values

# ── Train/Test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── SMOTE ─────────────────────────────────────────────────────
print("⚖️  Applying SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ── Scale ─────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

# ── MinMax scale for KNN (distance-based, needs 0-1 range) ───
mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train_sm)
X_test_mm  = mm_scaler.transform(X_test)

print(f"✅ Features           : {len(feature_cols)}")
print(f"✅ Train size         : {X_train_sm.shape[0]:,}")
print(f"✅ Test size          : {len(X_test):,}")
print(f"✅ After SMOTE Fraud  : {y_train_sm.sum():,}")
print(f"\n✅ Data ready for training!")

# COMMAND ----------

# ============================================================
# Cell 3: Train Gaussian Naive Bayes
# ============================================================

print("🔄 Training Gaussian Naive Bayes...")

with mlflow.start_run(run_name="GaussianNB_CreditCard"):

    gnb_model = GaussianNB(var_smoothing=1e-9)
    gnb_model.fit(X_train_scaled, y_train_sm)

    gnb_probs = gnb_model.predict_proba(X_test_scaled)[:, 1]
    gnb_preds = (gnb_probs >= 0.5).astype(int)

    gnb_auc   = roc_auc_score(y_test, gnb_probs)
    gnb_ap    = average_precision_score(y_test, gnb_probs)

    mlflow.log_param("model",          "GaussianNB")
    mlflow.log_param("var_smoothing",  1e-9)
    mlflow.log_metric("auc_roc",       gnb_auc)
    mlflow.log_metric("avg_precision", gnb_ap)
    mlflow.sklearn.log_model(gnb_model, "gaussian_nb")

print(f"✅ Gaussian Naive Bayes trained!")
print(f"   AUC-ROC          : {gnb_auc:.4f}")
print(f"   Avg Precision    : {gnb_ap:.4f}")
print(f"\n{classification_report(y_test, gnb_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 4: Train K-Nearest Neighbors
# ============================================================

print("🔄 Training K-Nearest Neighbors (K=5)...")
print("   Note: KNN training is instant but prediction is slow on large datasets\n")

with mlflow.start_run(run_name="KNN_CreditCard"):

    # ── Use smaller sample for KNN — O(n) prediction time ────
    # KNN stores ALL training data and searches at prediction time
    sample_size   = 30000
    sample_idx    = np.random.choice(len(X_train_mm), sample_size, replace=False)
    X_knn_train   = X_train_mm[sample_idx]
    y_knn_train   = y_train_sm[sample_idx]

    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        metric='euclidean',
        weights='distance',
        algorithm='ball_tree',
        leaf_size=30,
        n_jobs=-1
    )
    knn_model.fit(X_knn_train, y_knn_train)

    knn_probs = knn_model.predict_proba(X_test_mm)[:, 1]
    knn_preds = (knn_probs >= 0.5).astype(int)

    knn_auc   = roc_auc_score(y_test, knn_probs)
    knn_ap    = average_precision_score(y_test, knn_probs)

    mlflow.log_param("model",       "KNN")
    mlflow.log_param("k",           5)
    mlflow.log_param("metric",      "euclidean")
    mlflow.log_param("weights",     "distance")
    mlflow.log_metric("auc_roc",    knn_auc)
    mlflow.log_metric("avg_precision", knn_ap)
    mlflow.sklearn.log_model(knn_model, "knn")

print(f"✅ KNN trained!")
print(f"   AUC-ROC          : {knn_auc:.4f}")
print(f"   Avg Precision    : {knn_ap:.4f}")
print(f"\n{classification_report(y_test, knn_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 5: Visualization + NB09 Summary
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('FRAUDSENSE — NB09: Naive Bayes vs KNN',
             fontsize=14, fontweight='bold', color='#e6edf3')

models_results = {
    'Gaussian Naive Bayes' : (gnb_probs, gnb_preds, gnb_auc, '#9b59b6'),
    'KNN (K=5)'            : (knn_probs, knn_preds, knn_auc, '#3498db'),
}

for col_idx, (name, (probs, preds, auc, color)) in enumerate(models_results.items()):

    # ── ROC Curve ────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[0][col_idx].plot(fpr, tpr, color=color, linewidth=2.5,
                          label=f'AUC = {auc:.4f}')
    axes[0][col_idx].plot([0,1], [0,1], 'white', linestyle='--', alpha=0.4)
    axes[0][col_idx].set_title(f'{name} — ROC Curve',
                                fontsize=11, fontweight='bold', color='#e6edf3')
    axes[0][col_idx].set_xlabel('False Positive Rate')
    axes[0][col_idx].set_ylabel('True Positive Rate')
    axes[0][col_idx].legend(fontsize=11)
    axes[0][col_idx].grid(alpha=0.3)

    # ── Confusion Matrix ──────────────────────────────────────
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt=',', ax=axes[1][col_idx],
                cmap=sns.light_palette(color, as_cmap=True),
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'],
                linewidths=2, linecolor='#0d1117',
                annot_kws={'size': 13, 'weight': 'bold'})
    axes[1][col_idx].set_title(f'{name} — Confusion Matrix',
                                fontsize=11, fontweight='bold', color='#e6edf3')
    axes[1][col_idx].set_ylabel('True Label')
    axes[1][col_idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('/tmp/nb09_results.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ── Save models ───────────────────────────────────────────────
os.makedirs('/tmp/fraudsense_models', exist_ok=True)

with open('/tmp/fraudsense_models/naive_bayes.pkl', 'wb') as f:
    pickle.dump({'model': gnb_model, 'scaler': scaler}, f)
with open('/tmp/fraudsense_models/knn.pkl', 'wb') as f:
    pickle.dump({'model': knn_model, 'scaler': mm_scaler}, f)

# ── Summary ───────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════╗
║              NB09 — RESULTS SUMMARY                         ║
╠══════════════════════════════════════════════════════════════╣
║  Model                  AUC-ROC    Avg Precision            ║
║  ─────────────────────────────────────────────              ║
║  Gaussian Naive Bayes : {gnb_auc:.4f}     {gnb_ap:.4f}                ║
║  KNN (K=5)            : {knn_auc:.4f}     {knn_ap:.4f}                ║
╚══════════════════════════════════════════════════════════════╝

🏆 Battle Royale Leaderboard So Far:
   Logistic Regression : 0.9632
   Linear SVM          : 0.9570
   Gaussian Naive Bayes: {gnb_auc:.4f}
   KNN (K=5)           : {knn_auc:.4f}
""")
print("✅ NB09 Complete — Round 2 done!")
print("🚀 Next → NB10: Random Forest + Extra Trees")

# COMMAND ----------

