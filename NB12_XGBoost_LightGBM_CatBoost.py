# Databricks notebook source
import subprocess
packages = ["xgboost", "lightgbm", "catboost"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"✅ {pkg}" if result.returncode == 0 else f"❌ {pkg}: {result.stderr}")

# COMMAND ----------

import subprocess
packages = ["imbalanced-learn", "networkx"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"✅ {pkg}" if result.returncode == 0 else f"❌ {pkg}: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE — NB12: XGBoost + LightGBM + CatBoost
# Cell 1: Banner + Setup
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║     FRAUDSENSE — NB12: XGBoost + LightGBM + CatBoost       ║
║     Classical ML Battle Royale — THE BIG 3 🔥               ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
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

print("✅ XGBoost version  :", xgb.__version__)
print("✅ LightGBM version :", lgb.__version__)
print("✅ MLflow ready")
print("\n💥 THE BIG 3 ARE HERE — LET'S GO!")

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

# Scale ratio for XGBoost class weight
scale_pos_weight = (y_train_sm == 0).sum() / (y_train_sm == 1).sum()

print(f"✅ Features              : {len(feature_cols)}")
print(f"✅ Train size            : {X_train_sm.shape[0]:,}")
print(f"✅ Test size             : {len(X_test):,}")
print(f"✅ After SMOTE Fraud     : {y_train_sm.sum():,}")
print(f"✅ Scale pos weight      : {scale_pos_weight:.2f}")
print(f"\n✅ Data ready for THE BIG 3!")

# COMMAND ----------

# ============================================================
# Cell 3: Train XGBoost
# ============================================================

print("⚡ Training XGBoost...")
print("   The Kaggle competition killer 🏆\n")

with mlflow.start_run(run_name="XGBoost_CreditCard"):

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    xgb_model.fit(
        X_train_scaled, y_train_sm,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_preds = (xgb_probs >= 0.5).astype(int)

    xgb_auc   = roc_auc_score(y_test, xgb_probs)
    xgb_ap    = average_precision_score(y_test, xgb_probs)

    mlflow.log_param("model",             "XGBoost")
    mlflow.log_param("n_estimators",      500)
    mlflow.log_param("learning_rate",     0.05)
    mlflow.log_param("max_depth",         6)
    mlflow.log_param("scale_pos_weight",  scale_pos_weight)
    mlflow.log_metric("auc_roc",          xgb_auc)
    mlflow.log_metric("avg_precision",    xgb_ap)
    mlflow.sklearn.log_model(xgb_model,   "xgboost")

print(f"✅ XGBoost trained!")
print(f"   AUC-ROC          : {xgb_auc:.4f}")
print(f"   Avg Precision    : {xgb_ap:.4f}")
print(f"\n{classification_report(y_test, xgb_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 4: Train LightGBM
# ============================================================

print("💡 Training LightGBM...")
print("   Microsoft's speed demon — leaf-wise growth 🍃\n")

with mlflow.start_run(run_name="LightGBM_CreditCard"):

    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    lgb_model.fit(
        X_train_scaled, y_train_sm,
        eval_set=[(X_test_scaled, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )

    lgb_probs = lgb_model.predict_proba(X_test_scaled)[:, 1]
    lgb_preds = (lgb_probs >= 0.5).astype(int)

    lgb_auc   = roc_auc_score(y_test, lgb_probs)
    lgb_ap    = average_precision_score(y_test, lgb_probs)

    mlflow.log_param("model",           "LightGBM")
    mlflow.log_param("n_estimators",    500)
    mlflow.log_param("num_leaves",      31)
    mlflow.log_param("learning_rate",   0.05)
    mlflow.log_metric("auc_roc",        lgb_auc)
    mlflow.log_metric("avg_precision",  lgb_ap)
    mlflow.sklearn.log_model(lgb_model, "lightgbm")

print(f"✅ LightGBM trained!")
print(f"   AUC-ROC          : {lgb_auc:.4f}")
print(f"   Avg Precision    : {lgb_ap:.4f}")
print(f"\n{classification_report(y_test, lgb_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 5: Train CatBoost
# ============================================================

print("🐱 Training CatBoost...")
print("   Yandex's categorical feature specialist 🎯\n")

with mlflow.start_run(run_name="CatBoost_CreditCard"):

    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        border_count=128,
        scale_pos_weight=scale_pos_weight,
        eval_metric='AUC',
        random_seed=42,
        verbose=0,
        allow_writing_files=False
    )

    cat_model.fit(
        X_train_scaled, y_train_sm,
        eval_set=(X_test_scaled, y_test),
        early_stopping_rounds=50,
        verbose=False
    )

    cat_probs = cat_model.predict_proba(X_test_scaled)[:, 1]
    cat_preds = (cat_probs >= 0.5).astype(int)

    cat_auc   = roc_auc_score(y_test, cat_probs)
    cat_ap    = average_precision_score(y_test, cat_probs)

    mlflow.log_param("model",           "CatBoost")
    mlflow.log_param("iterations",      500)
    mlflow.log_param("learning_rate",   0.05)
    mlflow.log_param("depth",           6)
    mlflow.log_metric("auc_roc",        cat_auc)
    mlflow.log_metric("avg_precision",  cat_ap)
    mlflow.sklearn.log_model(cat_model, "catboost")

print(f"✅ CatBoost trained!")
print(f"   AUC-ROC          : {cat_auc:.4f}")
print(f"   Avg Precision    : {cat_ap:.4f}")
print(f"\n{classification_report(y_test, cat_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 6: The Big 3 Showdown + Full Leaderboard
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('FRAUDSENSE — NB12: THE BIG 3 Showdown 🔥',
             fontsize=14, fontweight='bold', color='#e6edf3')

models_results = {
    'XGBoost'   : (xgb_probs, xgb_auc, '#f7931a'),
    'LightGBM'  : (lgb_probs, lgb_auc, '#00d4aa'),
    'CatBoost'  : (cat_probs, cat_auc, '#ff4444'),
}

# ── ROC Curves ───────────────────────────────────────────────
for name, (probs, auc, color) in models_results.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[0].plot(fpr, tpr, color=color, linewidth=3,
                 label=f'{name} (AUC={auc:.4f})')
axes[0].plot([0,1], [0,1], 'white', linestyle='--', alpha=0.4)
axes[0].set_title('ROC Curves — The Big 3',
                  fontsize=12, fontweight='bold', color='#e6edf3')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# ── PR Curves ────────────────────────────────────────────────
for name, (probs, auc, color) in models_results.items():
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    axes[1].plot(rec, prec, color=color, linewidth=3,
                 label=f'{name} (AP={ap:.4f})')
axes[1].axhline(y=y_test.mean(), color='white', linestyle='--',
                alpha=0.4, label='Baseline')
axes[1].set_title('Precision-Recall Curves — The Big 3',
                  fontsize=12, fontweight='bold', color='#e6edf3')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/nb12_big3_curves.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ── Save models ───────────────────────────────────────────────
os.makedirs('/tmp/fraudsense_models', exist_ok=True)
with open('/tmp/fraudsense_models/xgboost.pkl', 'wb') as f:
    pickle.dump({'model': xgb_model, 'scaler': scaler}, f)
with open('/tmp/fraudsense_models/lightgbm.pkl', 'wb') as f:
    pickle.dump({'model': lgb_model, 'scaler': scaler}, f)
with open('/tmp/fraudsense_models/catboost.pkl', 'wb') as f:
    pickle.dump({'model': cat_model, 'scaler': scaler}, f)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║              NB12 — THE BIG 3 RESULTS                       ║
╠══════════════════════════════════════════════════════════════╣
║  Model          AUC-ROC    Avg Precision                    ║
║  ─────────────────────────────────────────                  ║
║  XGBoost      : {xgb_auc:.4f}     {xgb_ap:.4f}                      ║
║  LightGBM     : {lgb_auc:.4f}     {lgb_ap:.4f}                      ║
║  CatBoost     : {cat_auc:.4f}     {cat_ap:.4f}                      ║
╚══════════════════════════════════════════════════════════════╝

🏆 FULL BATTLE ROYALE LEADERBOARD (12 Models):
   XGBoost                 : {xgb_auc:.4f}  ← BIG 3
   LightGBM                : {lgb_auc:.4f}  ← BIG 3
   CatBoost                : {cat_auc:.4f}  ← BIG 3
   Extra Trees             : 0.9786
   GradientBoosting        : 0.9764
   HistGradientBoosting    : 0.9711
   AdaBoost                : 0.9710
   Random Forest           : 0.9678
   Logistic Regression     : 0.9632
   Linear SVM              : 0.9570
   Gaussian Naive Bayes    : 0.9500
   KNN (K=5)               : 0.8880
""")
print("✅ NB12 Complete — THE BIG 3 have spoken!")
print("🚀 Next → NB13: TabNet + MLP Neural Networks")

# COMMAND ----------

