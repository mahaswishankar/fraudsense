# Databricks notebook source
# ============================================================
# Cell 0: Install Dependencies
# ============================================================
import subprocess
packages = ["scikit-learn", "xgboost", "lightgbm", "imbalanced-learn", "mlflow"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"{pkg} installed" if result.returncode == 0 else f"{pkg} failed: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE - NB19: Layer 1 Stacking
# Cell 1: Banner + Setup
# ============================================================
print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB19: Layer 1 Stacking             ║
║     Phase 5 - Ensemble Round 1                      ║
║     XGB + LGB + CatBoost -> Logistic Meta           ║
╚══════════════════════════════════════════════════════╝
""")

from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics         import roc_auc_score, average_precision_score, classification_report
from imblearn.over_sampling  import SMOTE
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier

plt.rcParams['figure.facecolor']  = '#0d1117'
plt.rcParams['axes.facecolor']    = '#161b22'
plt.rcParams['axes.edgecolor']    = '#30363d'
plt.rcParams['text.color']        = '#e6edf3'
plt.rcParams['axes.labelcolor']   = '#e6edf3'
plt.rcParams['xtick.color']       = '#e6edf3'
plt.rcParams['ytick.color']       = '#e6edf3'
plt.rcParams['grid.color']        = '#21262d'
plt.rcParams['font.family']       = 'monospace'

FRAUD_COLOR   = '#ff4444'
LEGIT_COLOR   = '#00d4aa'
ACCENT_COLOR  = '#f7931a'
STACK_COLOR   = '#34d399'

print("All libraries loaded")
print("Layer 1 Stacking starting...")

# COMMAND ----------

# ============================================================
# Cell 2: Load Gold Data + SMOTE
# ============================================================
GOLD_PATH = "/Volumes/workspace/default/fraud_data/gold/creditcard"
cc = spark.read.format("delta").load(GOLD_PATH).toPandas()

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

smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled  = scaler.transform(X_test)

print(f"Train size : {len(X_train_scaled):,}")
print(f"Test size  : {len(X_test_scaled):,}")
print(f"Features   : {X_train_scaled.shape[1]}")
print("Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: Define Base Learners (Layer 1)
# ============================================================
print("Defining Layer 1 base learners...")
print("Base learners: XGBoost + LightGBM + Extra Trees")

base_learners = {
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        eval_metric='auc',
        random_state=42,
        verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        random_state=42,
        verbose=-1
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

print(f"Base learners defined: {list(base_learners.keys())}")
print("Meta learner: Logistic Regression")

# COMMAND ----------

# ============================================================
# Cell 4: Generate Out-of-Fold Predictions (OOF)
# ============================================================
print("Generating out-of-fold predictions...")
print("Using 5-fold Stratified CV to avoid data leakage\n")

# OOF predictions become the training data for the meta-learner
# This is critical — if we train base learners on full train set and then
# use those predictions to train meta-learner, it's data leakage

N_FOLDS = 5
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Containers for OOF train predictions and test predictions
oof_train = np.zeros((len(X_train_scaled), len(base_learners)))
oof_test  = np.zeros((len(X_test_scaled),  len(base_learners)))

base_auc_scores = {}

for col_idx, (name, clf) in enumerate(base_learners.items()):
    print(f"\n--- {name} ---")
    fold_test_preds = np.zeros((len(X_test_scaled), N_FOLDS))
    fold_aucs       = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train_sm)):
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_train_sm[train_idx]
        X_fold_val   = X_train_scaled[val_idx]
        y_fold_val   = y_train_sm[val_idx]

        clf.fit(X_fold_train, y_fold_train)

        val_probs  = clf.predict_proba(X_fold_val)[:, 1]
        test_probs = clf.predict_proba(X_test_scaled)[:, 1]

        oof_train[val_idx, col_idx] = val_probs
        fold_test_preds[:, fold]    = test_probs

        fold_auc = roc_auc_score(y_fold_val, val_probs)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.4f}")

    # Average test predictions across folds
    oof_test[:, col_idx] = fold_test_preds.mean(axis=1)

    mean_auc = np.mean(fold_aucs)
    base_auc_scores[name] = mean_auc
    print(f"  Mean CV AUC: {mean_auc:.4f}")

print("\n--- Base Learner CV Summary ---")
for name, auc in base_auc_scores.items():
    print(f"  {name:<15} : {auc:.4f}")

print("\nOOF predictions generated!")
print(f"OOF train shape : {oof_train.shape}")
print(f"OOF test shape  : {oof_test.shape}")

# COMMAND ----------

# ============================================================
# Cell 5: Train Meta Learner (Logistic Regression)
# ============================================================
print("Training meta-learner on OOF predictions...")
print("Meta learner: Logistic Regression\n")

# Meta-learner sees only the OOF predictions as features
# It learns the optimal weight to give each base model
meta_learner = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

meta_scaler = StandardScaler()
oof_train_scaled = meta_scaler.fit_transform(oof_train)
oof_test_scaled  = meta_scaler.transform(oof_test)

with mlflow.start_run(run_name="Layer1_Stacking"):

    meta_learner.fit(oof_train_scaled, y_train_sm)

    # Final predictions
    stack_probs = meta_learner.predict_proba(oof_test_scaled)[:, 1]
    stack_preds = (stack_probs >= 0.5).astype(int)
    stack_auc   = roc_auc_score(y_test, stack_probs)
    stack_ap    = average_precision_score(y_test, stack_probs)

    # Log base learner scores
    for name, auc in base_auc_scores.items():
        mlflow.log_metric(f"base_{name}_auc", auc)

    mlflow.log_param("base_learners",  "XGB+LGB+ExtraTrees")
    mlflow.log_param("meta_learner",   "LogisticRegression")
    mlflow.log_param("n_folds",        N_FOLDS)
    mlflow.log_param("meta_C",         1.0)
    mlflow.log_metric("stack_auc_roc", stack_auc)
    mlflow.log_metric("stack_avg_precision", stack_ap)
    mlflow.set_tag("project",          "FRAUDSENSE")
    mlflow.set_tag("notebook",         "NB19")
    mlflow.set_tag("phase",            "Phase5_Ensemble")

    mlflow.sklearn.log_model(meta_learner, "layer1_meta_learner")

print(f"Layer 1 Stack AUC-ROC   : {stack_auc:.4f}")
print(f"Layer 1 Stack Avg Prec  : {stack_ap:.4f}")

# Show meta-learner coefficients (how much weight each base model gets)
print("\nMeta-learner coefficients (feature importance):")
for name, coef in zip(base_learners.keys(), meta_learner.coef_[0]):
    print(f"  {name:<15} : {coef:.4f}")

# COMMAND ----------

# ============================================================
# Cell 6: OOF Prediction Correlation Heatmap
# ============================================================
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Correlation between base learner OOF predictions
corr_matrix = np.corrcoef(oof_test.T)
im = axes[0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
axes[0].set_xticks(range(len(base_learners)))
axes[0].set_yticks(range(len(base_learners)))
axes[0].set_xticklabels(list(base_learners.keys()), rotation=45, ha='right')
axes[0].set_yticklabels(list(base_learners.keys()))
axes[0].set_title('Base Learner Prediction Correlation', fontsize=12, fontweight='bold')
for i in range(len(base_learners)):
    for j in range(len(base_learners)):
        axes[0].text(j, i, f'{corr_matrix[i,j]:.2f}',
                     ha='center', va='center', fontsize=11, color='white')
plt.colorbar(im, ax=axes[0])

# Base learner AUC comparison vs stack
names  = list(base_auc_scores.keys()) + ['Layer1 Stack']
scores = list(base_auc_scores.values()) + [stack_auc]
colors = [ACCENT_COLOR] * len(base_auc_scores) + [STACK_COLOR]
bars   = axes[1].bar(names, scores, color=colors, edgecolor='#30363d', linewidth=0.5)
axes[1].set_ylim(min(scores) - 0.01, 1.0)
axes[1].set_title('Base Learners vs Layer 1 Stack', fontsize=12, fontweight='bold')
axes[1].set_ylabel('AUC-ROC')
for bar, score in zip(bars, scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=9)
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', rotation=15)

plt.suptitle('FRAUDSENSE NB19 - Layer 1 Stacking Analysis',
             fontsize=13, fontweight='bold', color='#e6edf3', y=1.02)
plt.tight_layout()

chart_path = "/Volumes/workspace/default/fraud_data/models/nb19_layer1_stacking.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart_path}")

# COMMAND ----------

# ============================================================
# Cell 7: Save OOF Predictions for NB20
# ============================================================
import os
os.makedirs("/tmp/fraudsense_stacking", exist_ok=True)

np.save("/tmp/fraudsense_stacking/oof_train_l1.npy", oof_train)
np.save("/tmp/fraudsense_stacking/oof_test_l1.npy",  oof_test)
np.save("/tmp/fraudsense_stacking/y_train_sm.npy",   y_train_sm)
np.save("/tmp/fraudsense_stacking/y_test.npy",       y_test)

print("OOF predictions saved for NB20:")
print("  /tmp/fraudsense_stacking/oof_train_l1.npy")
print("  /tmp/fraudsense_stacking/oof_test_l1.npy")
print("  /tmp/fraudsense_stacking/y_train_sm.npy")
print("  /tmp/fraudsense_stacking/y_test.npy")

# COMMAND ----------

# ============================================================
# Cell 8: Results Summary
# ============================================================
print(classification_report(y_test, stack_preds, target_names=['Legit', 'Fraud']))

print("╔══════════════════════════════════════════════════════╗")
print("║         NB19 - LAYER 1 STACKING RESULTS             ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  {'Model':<24} {'AUC-ROC':<10} {'Avg Precision'}   ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  {'XGBoost (base)':<24} {base_auc_scores['XGBoost']:.4f}     {'(CV mean)':12}   ║")
print(f"║  {'LightGBM (base)':<24} {base_auc_scores['LightGBM']:.4f}     {'(CV mean)':12}   ║")
print(f"║  {'ExtraTrees (base)':<24} {base_auc_scores['ExtraTrees']:.4f}     {'(CV mean)':12}   ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  {'Layer 1 Stack':<24} {stack_auc:.4f}     {stack_ap:.4f}  <- NEW  ║")
print("╚══════════════════════════════════════════════════════╝")

print("NB19 complete!")
print("Next: NB20 - Layer 2 Stacking")

# COMMAND ----------

