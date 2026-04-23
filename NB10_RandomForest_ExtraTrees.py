# Databricks notebook source
import subprocess
packages = ["imbalanced-learn", "networkx"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"✅ {pkg}" if result.returncode == 0 else f"❌ {pkg}: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE — NB10: Random Forest + Extra Trees
# Cell 1: Banner + Setup
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║     FRAUDSENSE — NB10: Random Forest + Extra Trees          ║
║     Classical ML Battle Royale — Round 3                    ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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
print("\n⚔️  Classical ML Battle Royale — Round 3 starting...")
print("🌲 Random Forest vs Extra Trees — FIGHT!")

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
# Cell 3: Train Random Forest
# ============================================================

print("🌲 Training Random Forest (200 trees)...")
print("   This will take 3-5 minutes...\n")

with mlflow.start_run(run_name="RandomForest_CreditCard"):

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(X_train_scaled, y_train_sm)

    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_preds = (rf_probs >= 0.5).astype(int)

    rf_auc   = roc_auc_score(y_test, rf_probs)
    rf_ap    = average_precision_score(y_test, rf_probs)

    mlflow.log_param("model",           "RandomForest")
    mlflow.log_param("n_estimators",    200)
    mlflow.log_param("max_depth",       15)
    mlflow.log_param("max_features",    "sqrt")
    mlflow.log_metric("auc_roc",        rf_auc)
    mlflow.log_metric("avg_precision",  rf_ap)
    mlflow.sklearn.log_model(rf_model,  "random_forest")

print(f"✅ Random Forest trained!")
print(f"   AUC-ROC          : {rf_auc:.4f}")
print(f"   Avg Precision    : {rf_ap:.4f}")
print(f"\n{classification_report(y_test, rf_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 4: Train Extra Trees
# ============================================================

print("🌳 Training Extra Trees (200 trees)...")
print("   Extra Trees is faster than RF — random splits, no bootstrap\n")

with mlflow.start_run(run_name="ExtraTrees_CreditCard"):

    et_model = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    et_model.fit(X_train_scaled, y_train_sm)

    et_probs = et_model.predict_proba(X_test_scaled)[:, 1]
    et_preds = (et_probs >= 0.5).astype(int)

    et_auc   = roc_auc_score(y_test, et_probs)
    et_ap    = average_precision_score(y_test, et_probs)

    mlflow.log_param("model",           "ExtraTrees")
    mlflow.log_param("n_estimators",    200)
    mlflow.log_param("max_depth",       15)
    mlflow.log_param("bootstrap",       False)
    mlflow.log_metric("auc_roc",        et_auc)
    mlflow.log_metric("avg_precision",  et_ap)
    mlflow.sklearn.log_model(et_model,  "extra_trees")

print(f"✅ Extra Trees trained!")
print(f"   AUC-ROC          : {et_auc:.4f}")
print(f"   Avg Precision    : {et_ap:.4f}")
print(f"\n{classification_report(y_test, et_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 5: Feature Importance — What drives fraud detection?
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('FRAUDSENSE — NB10: Feature Importance Analysis',
             fontsize=14, fontweight='bold', color='#e6edf3')

for ax, (name, model, color) in zip(axes, [
    ('Random Forest',  rf_model, '#2ecc71'),
    ('Extra Trees',    et_model, '#e74c3c')
]):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top20       = importances.nlargest(20).sort_values()

    bars = ax.barh(range(len(top20)), top20.values,
                   color=color, alpha=0.8, edgecolor='#30363d')
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20.index, fontsize=9)
    ax.set_title(f'{name} — Top 20 Feature Importances',
                 fontsize=12, fontweight='bold', color='#e6edf3')
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    for bar, val in zip(bars, top20.values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8, color='#e6edf3')

plt.tight_layout()
plt.savefig('/tmp/nb10_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()

print("🎯 Top 5 most important features (Random Forest):")
for feat, imp in importances.nlargest(5).items():
    print(f"   {feat:<25} : {imp:.4f}")

# COMMAND ----------

# ============================================================
# Cell 6: ROC Curves + Full Summary
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('FRAUDSENSE — NB10: Random Forest vs Extra Trees',
             fontsize=14, fontweight='bold', color='#e6edf3')

models_results = {
    'Random Forest' : (rf_probs, rf_auc, '#2ecc71'),
    'Extra Trees'   : (et_probs, et_auc, '#e74c3c'),
}

# ── ROC Curves ───────────────────────────────────────────────
for name, (probs, auc, color) in models_results.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[0].plot(fpr, tpr, color=color, linewidth=2.5,
                 label=f'{name} (AUC={auc:.4f})')
axes[0].plot([0,1], [0,1], 'white', linestyle='--', alpha=0.4, label='Random')
axes[0].set_title('ROC Curves', fontsize=12, fontweight='bold', color='#e6edf3')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# ── PR Curves ────────────────────────────────────────────────
for name, (probs, auc, color) in models_results.items():
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    axes[1].plot(rec, prec, color=color, linewidth=2.5,
                 label=f'{name} (AP={ap:.4f})')
axes[1].axhline(y=y_test.mean(), color='white', linestyle='--', alpha=0.4, label='Baseline')
axes[1].set_title('Precision-Recall Curves', fontsize=12, fontweight='bold', color='#e6edf3')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/nb10_roc_pr.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ── Save models ───────────────────────────────────────────────
os.makedirs('/tmp/fraudsense_models', exist_ok=True)
with open('/tmp/fraudsense_models/random_forest.pkl', 'wb') as f:
    pickle.dump({'model': rf_model, 'scaler': scaler}, f)
with open('/tmp/fraudsense_models/extra_trees.pkl', 'wb') as f:
    pickle.dump({'model': et_model, 'scaler': scaler}, f)

print(f"""
╔══════════════════════════════════════════════════════════════╗
║              NB10 — RESULTS SUMMARY                         ║
╠══════════════════════════════════════════════════════════════╣
║  Model                  AUC-ROC    Avg Precision            ║
║  ─────────────────────────────────────────────              ║
║  Random Forest        : {rf_auc:.4f}     {rf_ap:.4f}                ║
║  Extra Trees          : {et_auc:.4f}     {et_ap:.4f}                ║
╚══════════════════════════════════════════════════════════════╝

🏆 Battle Royale Leaderboard So Far:
   Random Forest       : {rf_auc:.4f}  ← NEW
   Extra Trees         : {et_auc:.4f}  ← NEW
   Logistic Regression : 0.9632
   Gaussian Naive Bayes: 0.9500
   Linear SVM          : 0.9570
   KNN (K=5)           : 0.8880
""")
print("✅ NB10 Complete — Round 3 done!")
print("🚀 Next → NB11: AdaBoost + Gradient Boosting + HistGradientBoosting")

# COMMAND ----------

