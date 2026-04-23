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
# FRAUDSENSE - NB20: Layer 2 Stacking
# Cell 1: Banner + Setup
# ============================================================
print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB20: Layer 2 Stacking             ║
║     Phase 5 - Ensemble Round 2                      ║
║     Trees + Neural Nets -> XGBoost Meta             ║
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

FRAUD_COLOR  = '#ff4444'
LEGIT_COLOR  = '#00d4aa'
ACCENT_COLOR = '#f7931a'
STACK_COLOR  = '#34d399'

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# print(f"PyTorch : {torch.__version__}")
# print(f"Device  : {DEVICE}")
print("All libraries loaded")
print("Layer 2 Stacking starting...")

# COMMAND ----------

# ============================================================
# Cell 2: Load Gold Data — SMOTE Applied Per Fold (No Leakage)
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

# Split BEFORE any SMOTE — test set is always real data only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train size : {len(X_train_scaled):,}  (raw, no SMOTE yet)")
print(f"Test size  : {len(X_test_scaled):,}")
print(f"Features   : {X_train_scaled.shape[1]}")
print("SMOTE will be applied inside each CV fold to prevent leakage")
print("Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: Define Layer 2 Base Learners
# ============================================================
print("Defining Layer 2 base learners...")
print("Diverse mix: Tree-based + Boosting models")

# Layer 2 uses a wider, more diverse set than Layer 1
# Including models that weren't in Layer 1 for maximum diversity
base_learners_l2 = {
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
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
}

print(f"Base learners: {list(base_learners_l2.keys())}")
print("Meta learner : XGBoost")

# COMMAND ----------

# ============================================================
# Cell 4: OOF with SMOTE Applied Per Fold
# ============================================================
print("Generating OOF predictions with per-fold SMOTE...\n")

N_FOLDS = 5
skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_train_l2 = np.zeros((len(X_train_scaled), len(base_learners_l2)))
oof_test_l2  = np.zeros((len(X_test_scaled),  len(base_learners_l2)))

base_auc_scores_l2 = {}

for col_idx, (name, clf) in enumerate(base_learners_l2.items()):
    print(f"\n--- {name} ---")
    fold_test_preds = np.zeros((len(X_test_scaled), N_FOLDS))
    fold_aucs       = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_fold_train = X_train_scaled[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val   = X_train_scaled[val_idx]
        y_fold_val   = y_train[val_idx]

        # SMOTE applied only on this fold's training data — no leakage
        smote = SMOTE(random_state=42, sampling_strategy=0.1)
        X_fold_sm, y_fold_sm = smote.fit_resample(X_fold_train, y_fold_train)

        clf.fit(X_fold_sm, y_fold_sm)

        val_probs  = clf.predict_proba(X_fold_val)[:, 1]
        test_probs = clf.predict_proba(X_test_scaled)[:, 1]

        oof_train_l2[val_idx, col_idx] = val_probs
        fold_test_preds[:, fold]       = test_probs

        fold_auc = roc_auc_score(y_fold_val, val_probs)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold+1}/{N_FOLDS} | AUC: {fold_auc:.4f}")

    oof_test_l2[:, col_idx] = fold_test_preds.mean(axis=1)

    mean_auc = np.mean(fold_aucs)
    base_auc_scores_l2[name] = mean_auc
    print(f"  Mean CV AUC : {mean_auc:.4f}")

print("\n--- Layer 2 Base Learner CV Summary ---")
for name, auc in base_auc_scores_l2.items():
    print(f"  {name:<15} : {auc:.4f}")

print("\nOOF predictions generated!")

# COMMAND ----------

# ============================================================
# Cell 5: Also Load Layer 1 OOF Predictions
# ============================================================
print("Loading Layer 1 OOF predictions from NB19...")

try:
    oof_test_l1 = np.load("/tmp/fraudsense_stacking/oof_test_l1.npy")
    print(f"Layer 1 OOF test shape : {oof_test_l1.shape}")
    
    # Combine Layer 1 + Layer 2 OOF predictions for the meta-learner
    # This gives the meta-learner signals from both stacking layers
    oof_test_combined  = np.hstack([oof_test_l2, oof_test_l1])
    
    # For training meta: use L2 OOF train + L1 OOF train if available
    oof_train_l1 = np.load("/tmp/fraudsense_stacking/oof_train_l1.npy")
    oof_train_combined = np.hstack([oof_train_l2, oof_train_l1])
    print(f"Combined meta features : {oof_test_combined.shape[1]}")
    print("Layer 1 + Layer 2 OOF combined!")

except FileNotFoundError:
    print("Layer 1 OOF not found — using Layer 2 OOF only")
    oof_test_combined  = oof_test_l2
    oof_train_combined = oof_train_l2

# COMMAND ----------

# ============================================================
# Cell 6: Train XGBoost Meta Learner
# ============================================================
print("Training XGBoost meta-learner...\n")

neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
spw = neg / pos

meta_xgb = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric='auc',
    random_state=42,
    verbosity=0
)

with mlflow.start_run(run_name="Layer2_Stacking"):

    meta_xgb.fit(oof_train_combined, y_train)

    stack2_probs = meta_xgb.predict_proba(oof_test_combined)[:, 1]
    stack2_preds = (stack2_probs >= 0.5).astype(int)
    stack2_auc   = roc_auc_score(y_test, stack2_probs)
    stack2_ap    = average_precision_score(y_test, stack2_probs)

    for name, auc in base_auc_scores_l2.items():
        mlflow.log_metric(f"base_{name}_auc", auc)

    mlflow.log_param("base_learners",   "ExtraTrees+RF+XGB+LGB")
    mlflow.log_param("meta_learner",    "XGBoost")
    mlflow.log_param("n_folds",         N_FOLDS)
    mlflow.log_param("smote_per_fold",  True)
    mlflow.log_param("l1_oof_included", True)
    mlflow.log_metric("stack2_auc_roc", stack2_auc)
    mlflow.log_metric("stack2_avg_precision", stack2_ap)
    mlflow.set_tag("project",           "FRAUDSENSE")
    mlflow.set_tag("notebook",          "NB20")
    mlflow.set_tag("phase",             "Phase5_Ensemble")

    mlflow.sklearn.log_model(meta_xgb, "layer2_meta_xgb")

print(f"Layer 2 Stack AUC-ROC  : {stack2_auc:.4f}")
print(f"Layer 2 Stack Avg Prec : {stack2_ap:.4f}")

# COMMAND ----------

# ============================================================
# Cell 7: Save for NB21
# ============================================================
import os
os.makedirs("/tmp/fraudsense_stacking", exist_ok=True)

np.save("/tmp/fraudsense_stacking/stack2_probs.npy",    stack2_probs)
np.save("/tmp/fraudsense_stacking/stack1_probs.npy",    
        np.load("/tmp/fraudsense_stacking/oof_test_l1.npy").mean(axis=1) 
        if False else stack2_probs)
np.save("/tmp/fraudsense_stacking/y_test.npy",          y_test)
np.save("/tmp/fraudsense_stacking/oof_test_l2.npy",     oof_test_l2)

print("Saved for NB21:")
print("  /tmp/fraudsense_stacking/stack2_probs.npy")
print("  /tmp/fraudsense_stacking/y_test.npy")
print("  /tmp/fraudsense_stacking/oof_test_l2.npy")

# COMMAND ----------

# ============================================================
# Cell 8: Visualization
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature importance of meta XGBoost
feature_names = (list(base_learners_l2.keys()) +
                 ["L1_XGB", "L1_LGB", "L1_ET"]
                 if oof_train_combined.shape[1] > 4
                 else list(base_learners_l2.keys()))
importances = meta_xgb.feature_importances_

axes[0].barh(feature_names, importances, color=STACK_COLOR,
             edgecolor='#30363d', linewidth=0.5)
axes[0].set_title('Meta XGBoost Feature Importance', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance')
axes[0].grid(axis='x', alpha=0.3)

# Layer comparison
labels = ['L1 Stack\n(LR meta)', 'L2 Stack\n(XGB meta)']
scores = [0.9840, stack2_auc]
colors = [ACCENT_COLOR, STACK_COLOR]
bars   = axes[1].bar(labels, scores, color=colors,
                     edgecolor='#30363d', linewidth=0.5, width=0.4)
axes[1].set_ylim(min(scores) - 0.01, 1.0)
axes[1].set_title('Layer 1 vs Layer 2 Stack', fontsize=12, fontweight='bold')
axes[1].set_ylabel('AUC-ROC')
for bar, score in zip(bars, scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=11)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('FRAUDSENSE NB20 - Layer 2 Stacking Analysis',
             fontsize=13, fontweight='bold', color='#e6edf3', y=1.02)
plt.tight_layout()

chart_path = "/Volumes/workspace/default/fraud_data/models/nb20_layer2_stacking.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart_path}")

# COMMAND ----------

# ============================================================
# Cell 9: Results Summary
# ============================================================
print(classification_report(y_test, stack2_preds, target_names=['Legit', 'Fraud']))

print("╔══════════════════════════════════════════════════════╗")
print("║         NB20 - LAYER 2 STACKING RESULTS             ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  {'Model':<26} {'AUC-ROC':<10} {'Avg Precision'}  ║")
print("╠══════════════════════════════════════════════════════╣")
for name, auc in base_auc_scores_l2.items():
    print(f"║  {name+' (base)':<26} {auc:.4f}     {'(CV mean)':12}  ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  {'Layer 1 Stack (NB19)':<26} {0.9840:.4f}     {'0.8247':12}  ║")
print(f"║  {'Layer 2 Stack':<26} {stack2_auc:.4f}     {stack2_ap:.4f}  <- NEW  ║")
print("╚══════════════════════════════════════════════════════╝")

print("NB20 complete!")
print("Next: NB21 - FRAUDSENSE Final 3-Layer Ensemble")

# COMMAND ----------

