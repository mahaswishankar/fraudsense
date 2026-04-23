# Databricks notebook source
import subprocess
packages = ["imbalanced-learn", "networkx"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"✅ {pkg}" if result.returncode == 0 else f"❌ {pkg}: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE — NB11: AdaBoost + GradientBoosting + HistGB
# Cell 1: Banner + Setup
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║  FRAUDSENSE — NB11: AdaBoost + GradBoost + HistGradBoost   ║
║  Classical ML Battle Royale — Round 4                       ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import (AdaBoostClassifier,
                               GradientBoostingClassifier,
                               HistGradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
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
print("\n⚔️  Classical ML Battle Royale — Round 4 starting...")
print("💥 AdaBoost vs GradientBoosting vs HistGradientBoosting!")

# COMMAND ----------

# ============================================================
# Cell 2: Load Gold Data + SMOTE
# ============================================================

print("📥 Loading CreditCard Gold dataset...")

cc = spark.read.format("delta").load(f"{GOLD_PATH}/creditcard").toPandas()

v_cols       = [c for c in cc.columns if c.startswith('V')]
eng_feats    = ['amount_log', 'amount_zscore', 'amount_spike',
                'is_night', 'tx_velocity_10', 'high_amount_flag',
                'v1_v2_interaction', 'v3_v4_interaction', 'v14_v17_interaction',
                'v_sum_top5', 'v_abs_sum']
eng_feats    = [f for f in eng_feats if f in cc.columns]
feature_cols = v_cols + eng_feats

X = cc[feature_cols].fillna(0).values
y = cc['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("⚖️  Applying SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

print(f"✅ Features           : {len(feature_cols)}")
print(f"✅ Train size         : {X_train_sm.shape[0]:,}")
print(f"✅ Test size          : {len(X_test):,}")
print(f"✅ After SMOTE Fraud  : {y_train_sm.sum():,}")
print(f"\n✅ Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: Train AdaBoost
# ============================================================

print("💥 Training AdaBoost (200 estimators)...")
print("   AdaBoost focuses on misclassified samples each round\n")

with mlflow.start_run(run_name="AdaBoost_CreditCard"):

    ada_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        algorithm='SAMME'
    )
    ada_model.fit(X_train_scaled, y_train_sm)

    ada_probs = ada_model.predict_proba(X_test_scaled)[:, 1]
    ada_preds = (ada_probs >= 0.5).astype(int)

    ada_auc   = roc_auc_score(y_test, ada_probs)
    ada_ap    = average_precision_score(y_test, ada_probs)

    mlflow.log_param("model",          "AdaBoost")
    mlflow.log_param("n_estimators",   200)
    mlflow.log_param("learning_rate",  0.1)
    mlflow.log_param("base_depth",     2)
    mlflow.log_metric("auc_roc",       ada_auc)
    mlflow.log_metric("avg_precision", ada_ap)
    mlflow.sklearn.log_model(ada_model, "adaboost")

print(f"✅ AdaBoost trained!")
print(f"   AUC-ROC          : {ada_auc:.4f}")
print(f"   Avg Precision    : {ada_ap:.4f}")
print(f"\n{classification_report(y_test, ada_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 4: Train Gradient Boosting
# ============================================================

print("📈 Training Gradient Boosting (200 estimators)...")
print("   GradBoost minimizes loss function iteratively\n")

with mlflow.start_run(run_name="GradientBoosting_CreditCard"):

    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=4,
        subsample=0.8,
        max_features='sqrt',
        random_state=42,
        verbose=0
    )
    gb_model.fit(X_train_scaled, y_train_sm)

    gb_probs = gb_model.predict_proba(X_test_scaled)[:, 1]
    gb_preds = (gb_probs >= 0.5).astype(int)

    gb_auc   = roc_auc_score(y_test, gb_probs)
    gb_ap    = average_precision_score(y_test, gb_probs)

    mlflow.log_param("model",          "GradientBoosting")
    mlflow.log_param("n_estimators",   200)
    mlflow.log_param("learning_rate",  0.05)
    mlflow.log_param("max_depth",      4)
    mlflow.log_param("subsample",      0.8)
    mlflow.log_metric("auc_roc",       gb_auc)
    mlflow.log_metric("avg_precision", gb_ap)
    mlflow.sklearn.log_model(gb_model, "gradient_boosting")

print(f"✅ Gradient Boosting trained!")
print(f"   AUC-ROC          : {gb_auc:.4f}")
print(f"   Avg Precision    : {gb_ap:.4f}")
print(f"\n{classification_report(y_test, gb_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 5: Train HistGradientBoosting
# ============================================================

print("⚡ Training HistGradientBoosting...")
print("   HistGB is sklearn's fastest native boosting — histogram-based\n")

with mlflow.start_run(run_name="HistGradientBoosting_CreditCard"):

    hgb_model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=0
    )
    hgb_model.fit(X_train_scaled, y_train_sm)

    hgb_probs = hgb_model.predict_proba(X_test_scaled)[:, 1]
    hgb_preds = (hgb_probs >= 0.5).astype(int)

    hgb_auc   = roc_auc_score(y_test, hgb_probs)
    hgb_ap    = average_precision_score(y_test, hgb_probs)

    mlflow.log_param("model",           "HistGradientBoosting")
    mlflow.log_param("max_iter",        200)
    mlflow.log_param("learning_rate",   0.05)
    mlflow.log_param("max_depth",       6)
    mlflow.log_metric("auc_roc",        hgb_auc)
    mlflow.log_metric("avg_precision",  hgb_ap)
    mlflow.sklearn.log_model(hgb_model, "hist_gradient_boosting")

print(f"✅ HistGradientBoosting trained!")
print(f"   AUC-ROC          : {hgb_auc:.4f}")
print(f"   Avg Precision    : {hgb_ap:.4f}")
print(f"\n{classification_report(y_test, hgb_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 6: ROC Curves + Full NB11 Summary
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('FRAUDSENSE — NB11: Boosting Family Performance',
             fontsize=14, fontweight='bold', color='#e6edf3')

models_results = {
    'AdaBoost'              : (ada_probs, ada_auc, '#e67e22'),
    'GradientBoosting'      : (gb_probs,  gb_auc,  '#8e44ad'),
    'HistGradientBoosting'  : (hgb_probs, hgb_auc, '#1abc9c'),
}

# ── ROC Curves ───────────────────────────────────────────────
for name, (probs, auc, color) in models_results.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[0].plot(fpr, tpr, color=color, linewidth=2.5,
                 label=f'{name} (AUC={auc:.4f})')
axes[0].plot([0,1], [0,1], 'white', linestyle='--', alpha=0.4, label='Random')
axes[0].set_title('ROC Curves — Boosting Family',
                  fontsize=12, fontweight='bold', color='#e6edf3')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# ── PR Curves ────────────────────────────────────────────────
for name, (probs, auc, color) in models_results.items():
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    axes[1].plot(rec, prec, color=color, linewidth=2.5,
                 label=f'{name} (AP={ap:.4f})')
axes[1].axhline(y=y_test.mean(), color='white', linestyle='--',
                alpha=0.4, label='Baseline')
axes[1].set_title('Precision-Recall Curves — Boosting Family',
                  fontsize=12, fontweight='bold', color='#e6edf3')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/nb11_boosting_curves.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ── Save models ───────────────────────────────────────────────
os.makedirs('/tmp/fraudsense_models', exist_ok=True)
with open('/tmp/fraudsense_models/adaboost.pkl', 'wb') as f:
    pickle.dump({'model': ada_model, 'scaler': scaler}, f)
with open('/tmp/fraudsense_models/gradient_boosting.pkl', 'wb') as f:
    pickle.dump({'model': gb_model, 'scaler': scaler}, f)
with open('/tmp/fraudsense_models/hist_gradient_boosting.pkl', 'wb') as f:
    pickle.dump({'model': hgb_model, 'scaler': scaler}, f)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║              NB11 — RESULTS SUMMARY                         ║
╠══════════════════════════════════════════════════════════════╣
║  Model                    AUC-ROC    Avg Precision          ║
║  ───────────────────────────────────────────────            ║
║  AdaBoost               : {ada_auc:.4f}     {ada_ap:.4f}              ║
║  GradientBoosting       : {gb_auc:.4f}     {gb_ap:.4f}              ║
║  HistGradientBoosting   : {hgb_auc:.4f}     {hgb_ap:.4f}              ║
╚══════════════════════════════════════════════════════════════╝

🏆 Battle Royale Leaderboard So Far:
   Extra Trees             : 0.9786  🥇
   Random Forest           : 0.9678  
   HistGradientBoosting    : {hgb_auc:.4f}  ← NEW
   GradientBoosting        : {gb_auc:.4f}  ← NEW
   AdaBoost                : {ada_auc:.4f}  ← NEW
   Logistic Regression     : 0.9632
   Gaussian Naive Bayes    : 0.9500
   Linear SVM              : 0.9570
   KNN (K=5)               : 0.8880
""")
print("✅ NB11 Complete — Round 4 done!")
print("🚀 Next → NB12: XGBoost + LightGBM + CatBoost — THE BIG 3!")

# COMMAND ----------

