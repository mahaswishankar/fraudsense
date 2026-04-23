# Databricks notebook source
import subprocess
packages = ["imbalanced-learn", "networkx"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"✅ {pkg}" if result.returncode == 0 else f"❌ {pkg}: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE — NB08: Classical ML — Logistic Regression + SVM
# Cell 1: Banner + Setup
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║     FRAUDSENSE — NB08: Logistic Regression + Linear SVM     ║
║     Classical ML Battle Royale — Round 1                    ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve)
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
print("\n⚔️  Classical ML Battle Royale — Round 1 starting...")

# COMMAND ----------

# ============================================================
# Cell 2: Load Gold Data + SMOTE Balancing
# ============================================================

print("📥 Loading CreditCard Gold dataset for training...")

cc = spark.read.format("delta").load(f"{GOLD_PATH}/creditcard").toPandas()

# ── Feature selection ─────────────────────────────────────────
v_cols      = [c for c in cc.columns if c.startswith('V')]
eng_feats   = ['amount_log', 'amount_zscore', 'amount_spike',
               'is_night', 'tx_velocity_10', 'high_amount_flag',
               'v1_v2_interaction', 'v3_v4_interaction', 'v14_v17_interaction',
               'v_sum_top5', 'v_abs_sum']
eng_feats   = [f for f in eng_feats if f in cc.columns]
feature_cols = v_cols + eng_feats

X = cc[feature_cols].fillna(0).values
y = cc['Class'].values

# ── Train/Test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"✅ Features           : {len(feature_cols)}")
print(f"✅ Train size         : {len(X_train):,}")
print(f"✅ Test size          : {len(X_test):,}")
print(f"✅ Train fraud rate   : {y_train.mean()*100:.3f}%")

# ── Apply SMOTE to training set only ─────────────────────────
print("\n⚖️  Applying SMOTE to balance training set...")
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"✅ Before SMOTE — Fraud: {y_train.sum():,} / Legit: {(y_train==0).sum():,}")
print(f"✅ After SMOTE  — Fraud: {y_train_sm.sum():,} / Legit: {(y_train_sm==0).sum():,}")

# ── Scale features ────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

print(f"\n✅ SMOTE + Scaling done! Ready for training.")

# COMMAND ----------

# ============================================================
# Cell 3: Train Logistic Regression
# ============================================================

print("🔄 Training Logistic Regression...")

with mlflow.start_run(run_name="LogisticRegression_CreditCard"):

    # ── Train ──────────────────────────────────────────────────
    lr_model = LogisticRegression(
        C=0.01,
        penalty='l2',
        solver='saga',
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    lr_model.fit(X_train_scaled, y_train_sm)

    # ── Predict ────────────────────────────────────────────────
    lr_probs  = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_preds  = (lr_probs >= 0.5).astype(int)

    # ── Metrics ────────────────────────────────────────────────
    lr_auc    = roc_auc_score(y_test, lr_probs)
    lr_ap     = average_precision_score(y_test, lr_probs)

    # ── Log to MLflow ──────────────────────────────────────────
    mlflow.log_param("model",         "LogisticRegression")
    mlflow.log_param("C",             0.01)
    mlflow.log_param("penalty",       "l2")
    mlflow.log_param("solver",        "saga")
    mlflow.log_metric("auc_roc",      lr_auc)
    mlflow.log_metric("avg_precision",lr_ap)
    mlflow.sklearn.log_model(lr_model, "logistic_regression")

print(f"✅ Logistic Regression trained!")
print(f"   AUC-ROC          : {lr_auc:.4f}")
print(f"   Avg Precision    : {lr_ap:.4f}")
print(f"\n{classification_report(y_test, lr_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 4: Train Linear SVM
# ============================================================

print("🔄 Training Linear SVM...")

with mlflow.start_run(run_name="LinearSVM_CreditCard"):

    # ── LinearSVC + Calibration for probability output ────────
    svm_base  = LinearSVC(
        C=0.01,
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    )
    svm_model = CalibratedClassifierCV(svm_base, cv=3, method='sigmoid')
    svm_model.fit(X_train_scaled, y_train_sm)

    # ── Predict ────────────────────────────────────────────────
    svm_probs = svm_model.predict_proba(X_test_scaled)[:, 1]
    svm_preds = (svm_probs >= 0.5).astype(int)

    # ── Metrics ────────────────────────────────────────────────
    svm_auc   = roc_auc_score(y_test, svm_probs)
    svm_ap    = average_precision_score(y_test, svm_probs)

    # ── Log to MLflow ──────────────────────────────────────────
    mlflow.log_param("model",          "LinearSVM")
    mlflow.log_param("C",              0.01)
    mlflow.log_param("calibration",    "sigmoid")
    mlflow.log_metric("auc_roc",       svm_auc)
    mlflow.log_metric("avg_precision", svm_ap)
    mlflow.sklearn.log_model(svm_model, "linear_svm")

print(f"✅ Linear SVM trained!")
print(f"   AUC-ROC          : {svm_auc:.4f}")
print(f"   Avg Precision    : {svm_ap:.4f}")
print(f"\n{classification_report(y_test, svm_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 5: ROC Curves + Precision-Recall Curves
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('FRAUDSENSE — NB08: LR vs SVM Performance Curves',
             fontsize=14, fontweight='bold', color='#e6edf3')

models = {
    'Logistic Regression' : (lr_probs,  lr_auc,  '#00d4aa'),
    'Linear SVM'          : (svm_probs, svm_auc, '#f7931a'),
}

# ── ROC Curves ───────────────────────────────────────────────
for name, (probs, auc, color) in models.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[0].plot(fpr, tpr, color=color, linewidth=2.5,
                 label=f'{name} (AUC={auc:.4f})')

axes[0].plot([0,1], [0,1], 'white', linestyle='--', alpha=0.4, label='Random')
axes[0].fill_between([0,1], [0,1], alpha=0.05, color='white')
axes[0].set_title('ROC Curve', fontsize=12, fontweight='bold', color='#e6edf3')
axes[0].set_xlabel('False Positive Rate', fontsize=11)
axes[0].set_ylabel('True Positive Rate', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# ── Precision-Recall Curves ───────────────────────────────────
for name, (probs, auc, color) in models.items():
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    axes[1].plot(rec, prec, color=color, linewidth=2.5,
                 label=f'{name} (AP={ap:.4f})')

axes[1].axhline(y=y_test.mean(), color='white', linestyle='--',
                alpha=0.4, label=f'Baseline ({y_test.mean():.3f})')
axes[1].set_title('Precision-Recall Curve', fontsize=12,
                  fontweight='bold', color='#e6edf3')
axes[1].set_xlabel('Recall', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/nb08_roc_pr_curves.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ ROC + PR curves saved!")

# COMMAND ----------

# ============================================================
# Cell 6: Confusion Matrices + Save Models
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('FRAUDSENSE — NB08: Confusion Matrices',
             fontsize=13, fontweight='bold', color='#e6edf3')

model_results = {
    'Logistic Regression' : (lr_preds,  '#00d4aa'),
    'Linear SVM'          : (svm_preds, '#f7931a'),
}

for ax, (name, (preds, color)) in zip(axes, model_results.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt=',', ax=ax,
                cmap=sns.light_palette(color, as_cmap=True),
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'],
                linewidths=2, linecolor='#0d1117',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(f'{name}', fontsize=12, fontweight='bold', color='#e6edf3')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig('/tmp/nb08_confusion_matrices.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ── Save models as pickle ─────────────────────────────────────
os.makedirs('/tmp/fraudsense_models', exist_ok=True)

with open('/tmp/fraudsense_models/logistic_regression.pkl', 'wb') as f:
    pickle.dump({'model': lr_model, 'scaler': scaler}, f)
with open('/tmp/fraudsense_models/linear_svm.pkl', 'wb') as f:
    pickle.dump({'model': svm_model, 'scaler': scaler}, f)

# ── NB08 Summary ─────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════════════════════════╗
║              NB08 — RESULTS SUMMARY                         ║
╠══════════════════════════════════════════════════════════════╣
║  Model                  AUC-ROC    Avg Precision            ║
║  ─────────────────────────────────────────────              ║
║  Logistic Regression  : {lr_auc:.4f}     {lr_ap:.4f}                ║
║  Linear SVM           : {svm_auc:.4f}     {svm_ap:.4f}                ║
╚══════════════════════════════════════════════════════════════╝
""")
print("✅ Models saved to /tmp/fraudsense_models/")
print("✅ NB08 Complete — Round 1 done!")
print("🚀 Next → NB09: Naive Bayes + KNN")

# COMMAND ----------

