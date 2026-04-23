# Databricks notebook source
# ============================================================
# Cell 0: Install Dependencies
# ============================================================
import subprocess
packages = ["pytorch-tabnet", "torch", "scikit-learn", "imbalanced-learn"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"✅ {pkg}" if result.returncode == 0 else f"❌ {pkg}: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE — NB13: TabNet + MLP
# Cell 1: Banner + Setup
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║     FRAUDSENSE — NB13: TabNet + MLP                         ║
║     Classical ML Battle Royale — Round 5 (Neural Networks)  ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.neural_network import MLPClassifier
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
print("✅ TabNet + MLP ready")
print("\n🧠 Neural Network round starting...")

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
X_train_scaled = scaler.fit_transform(X_train_sm).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)
y_train_sm     = y_train_sm.astype(np.int64)
y_test_int     = y_test.astype(np.int64)

print(f"✅ Features           : {len(feature_cols)}")
print(f"✅ Train size         : {X_train_sm.shape[0]:,}")
print(f"✅ Test size          : {len(X_test):,}")
print(f"✅ Dtype              : float32 (required by TabNet)")
print(f"\n✅ Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: Train MLP (Multi-Layer Perceptron)
# ============================================================

print("🧠 Training MLP Neural Network...")
print("   Architecture: 256 → 128 → 64 → 1\n")

with mlflow.start_run(run_name="MLP_CreditCard"):

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=512,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    )
    mlp_model.fit(X_train_scaled, y_train_sm)

    mlp_probs = mlp_model.predict_proba(X_test_scaled)[:, 1]
    mlp_preds = (mlp_probs >= 0.5).astype(int)

    mlp_auc   = roc_auc_score(y_test, mlp_probs)
    mlp_ap    = average_precision_score(y_test, mlp_probs)

    mlflow.log_param("model",           "MLP")
    mlflow.log_param("architecture",    "256-128-64")
    mlflow.log_param("activation",      "relu")
    mlflow.log_param("solver",          "adam")
    mlflow.log_metric("auc_roc",        mlp_auc)
    mlflow.log_metric("avg_precision",  mlp_ap)
    mlflow.sklearn.log_model(mlp_model, "mlp")

print(f"✅ MLP trained!")
print(f"   AUC-ROC          : {mlp_auc:.4f}")
print(f"   Avg Precision    : {mlp_ap:.4f}")
print(f"   Iterations ran   : {mlp_model.n_iter_}")
print(f"\n{classification_report(y_test, mlp_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 4: Train TabNet
# ============================================================

print("🎯 Training TabNet...")
print("   Google's attention-based neural network for tabular data\n")

with mlflow.start_run(run_name="TabNet_CreditCard"):

    tabnet_model = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-3,
        optimizer_fn=__import__('torch').optim.Adam,
        optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
        scheduler_params=dict(step_size=10, gamma=0.9),
        scheduler_fn=__import__('torch').optim.lr_scheduler.StepLR,
        mask_type='entmax',
        verbose=0,
        seed=42
    )

    # Class weights for imbalanced data
    class_weights = {0: 1.0, 1: 10.0}

    tabnet_model.fit(
        X_train_scaled, y_train_sm,
        eval_set=[(X_test_scaled, y_test_int)],
        eval_metric=['auc'],
        max_epochs=100,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
        weights=1,
        drop_last=False
    )

    tabnet_probs = tabnet_model.predict_proba(X_test_scaled)[:, 1]
    tabnet_preds = (tabnet_probs >= 0.5).astype(int)

    tabnet_auc   = roc_auc_score(y_test, tabnet_probs)
    tabnet_ap    = average_precision_score(y_test, tabnet_probs)

    mlflow.log_param("model",       "TabNet")
    mlflow.log_param("n_d",         32)
    mlflow.log_param("n_steps",     5)
    mlflow.log_param("mask_type",   "entmax")
    mlflow.log_metric("auc_roc",    tabnet_auc)
    mlflow.log_metric("avg_precision", tabnet_ap)

print(f"✅ TabNet trained!")
print(f"   AUC-ROC          : {tabnet_auc:.4f}")
print(f"   Avg Precision    : {tabnet_ap:.4f}")
print(f"\n{classification_report(y_test, tabnet_preds, target_names=['Legit', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 5: TabNet Feature Importance + Full Visualization
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('FRAUDSENSE — NB13: TabNet + MLP Performance',
             fontsize=14, fontweight='bold', color='#e6edf3')

models_results = {
    'MLP (256-128-64)' : (mlp_probs,    mlp_auc,    '#3498db'),
    'TabNet'           : (tabnet_probs, tabnet_auc, '#e74c3c'),
}

# ── ROC Curves ───────────────────────────────────────────────
for name, (probs, auc, color) in models_results.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    axes[0][0].plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f'{name} (AUC={auc:.4f})')
axes[0][0].plot([0,1], [0,1], 'white', linestyle='--', alpha=0.4)
axes[0][0].set_title('ROC Curves', fontsize=11, fontweight='bold', color='#e6edf3')
axes[0][0].set_xlabel('False Positive Rate')
axes[0][0].set_ylabel('True Positive Rate')
axes[0][0].legend(fontsize=10)
axes[0][0].grid(alpha=0.3)

# ── PR Curves ────────────────────────────────────────────────
for name, (probs, auc, color) in models_results.items():
    prec, rec, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)
    axes[0][1].plot(rec, prec, color=color, linewidth=2.5,
                    label=f'{name} (AP={ap:.4f})')
axes[0][1].axhline(y=y_test.mean(), color='white', linestyle='--', alpha=0.4)
axes[0][1].set_title('Precision-Recall Curves', fontsize=11,
                      fontweight='bold', color='#e6edf3')
axes[0][1].set_xlabel('Recall')
axes[0][1].set_ylabel('Precision')
axes[0][1].legend(fontsize=10)
axes[0][1].grid(alpha=0.3)

# ── TabNet Feature Importance ─────────────────────────────────
tabnet_importance = pd.Series(
    tabnet_model.feature_importances_,
    index=feature_cols
).nlargest(20).sort_values()

axes[1][0].barh(range(len(tabnet_importance)), tabnet_importance.values,
                color='#e74c3c', alpha=0.8, edgecolor='#30363d')
axes[1][0].set_yticks(range(len(tabnet_importance)))
axes[1][0].set_yticklabels(tabnet_importance.index, fontsize=9)
axes[1][0].set_title('TabNet — Top 20 Feature Importances (Attention)',
                      fontsize=11, fontweight='bold', color='#e6edf3')
axes[1][0].set_xlabel('Attention Score')
axes[1][0].grid(axis='x', alpha=0.3)

# ── MLP Loss Curve ────────────────────────────────────────────
axes[1][1].plot(mlp_model.loss_curve_, color='#3498db', linewidth=2.5,
                label='Training Loss')
if hasattr(mlp_model, 'validation_scores_'):
    axes[1][1].plot(mlp_model.validation_scores_, color=ACCENT_COLOR,
                    linewidth=2, linestyle='--', label='Validation Score')
axes[1][1].set_title('MLP — Training Loss Curve',
                      fontsize=11, fontweight='bold', color='#e6edf3')
axes[1][1].set_xlabel('Iteration')
axes[1][1].set_ylabel('Loss')
axes[1][1].legend(fontsize=10)
axes[1][1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/nb13_neural_networks.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()

# ── Save models ───────────────────────────────────────────────
os.makedirs('/tmp/fraudsense_models', exist_ok=True)
with open('/tmp/fraudsense_models/mlp.pkl', 'wb') as f:
    pickle.dump({'model': mlp_model, 'scaler': scaler}, f)
tabnet_model.save_model('/tmp/fraudsense_models/tabnet')

print(f"""
╔══════════════════════════════════════════════════════════════╗
║              NB13 — RESULTS SUMMARY                         ║
╠══════════════════════════════════════════════════════════════╣
║  Model              AUC-ROC    Avg Precision                ║
║  ─────────────────────────────────────────────              ║
║  MLP (256-128-64) : {mlp_auc:.4f}     {mlp_ap:.4f}                  ║
║  TabNet           : {tabnet_auc:.4f}     {tabnet_ap:.4f}                  ║
╚══════════════════════════════════════════════════════════════╝

🏆 FULL BATTLE ROYALE LEADERBOARD (14 Models):
   Extra Trees             : 0.9786  🥇
   XGBoost                 : 0.9765
   GradientBoosting        : 0.9764
   LightGBM                : 0.9724
   HistGradientBoosting    : 0.9711
   AdaBoost                : 0.9710
   CatBoost                : 0.9681
   Random Forest           : 0.9678
   MLP                     : {mlp_auc:.4f}  ← NEW
   TabNet                  : {tabnet_auc:.4f}  ← NEW
   Logistic Regression     : 0.9632
   Gaussian Naive Bayes    : 0.9500
   Linear SVM              : 0.9570
   KNN (K=5)               : 0.8880
""")
print("✅ NB13 Complete — Neural Network round done!")
print("🚀 Next → NB14: MLflow Master Leaderboard")

# COMMAND ----------

