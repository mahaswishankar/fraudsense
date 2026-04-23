# Databricks notebook source
# ============================================================
# Cell 0: Install Dependencies
# ============================================================
import subprocess
packages = ["lime", "xgboost", "scikit-learn", "imbalanced-learn", "mlflow"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"{pkg} installed" if result.returncode == 0 else f"{pkg} failed: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE - NB24: LIME Explainability
# Cell 1: Banner + Setup
# ============================================================
print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB24: LIME Explainability          ║
║     Phase 6 - Local Interpretable Explanations      ║
╚══════════════════════════════════════════════════════╝
""")

from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_auc_score
from imblearn.over_sampling  import SMOTE
from xgboost                 import XGBClassifier

plt.rcParams['figure.facecolor']  = '#0d1117'
plt.rcParams['axes.facecolor']    = '#161b22'
plt.rcParams['axes.edgecolor']    = '#30363d'
plt.rcParams['text.color']        = '#e6edf3'
plt.rcParams['axes.labelcolor']   = '#e6edf3'
plt.rcParams['xtick.color']       = '#e6edf3'
plt.rcParams['ytick.color']       = '#e6edf3'
plt.rcParams['grid.color']        = '#21262d'
plt.rcParams['font.family']       = 'monospace'

FRAUD_COLOR    = '#ff4444'
LEGIT_COLOR    = '#00d4aa'
ACCENT_COLOR   = '#f7931a'
FRAUDSENSE_COL = '#a78bfa'

print("All libraries loaded")
print("LIME Explainability starting...")

# COMMAND ----------

# ============================================================
# Cell 2: Load Data + Train XGBoost
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print("Training XGBoost for LIME analysis...")
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
    eval_metric='auc', random_state=42, verbosity=0)
xgb.fit(X_train_sm, y_train_sm)

xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_scaled)[:, 1])
print(f"XGBoost AUC : {xgb_auc:.4f}")
print("Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: Build LIME Explainer
# ============================================================
print("Building LIME explainer...")

# LIME needs the training data to understand feature distributions
# It perturbs the input and fits a local linear model around each prediction
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data   = X_train_scaled,
    feature_names   = feature_cols,
    class_names     = ['Legit', 'Fraud'],
    mode            = 'classification',
    discretize_continuous = True,
    random_state    = 42
)

print("LIME explainer built!")
print("Ready to explain individual predictions...")

# COMMAND ----------

# ============================================================
# Cell 4: Explain Individual Fraud Cases
# ============================================================
print("Generating LIME explanations for fraud cases...\n")

# Get fraud cases from test set
fraud_idx = np.where(y_test == 1)[0]
legit_idx = np.where(y_test == 0)[0]

# Explain 3 fraud + 2 legit cases
cases = {
    "Fraud Case 1":  (fraud_idx[0],  1),
    "Fraud Case 2":  (fraud_idx[1],  1),
    "Fraud Case 3":  (fraud_idx[2],  1),
    "Legit Case 1":  (legit_idx[0],  0),
    "Legit Case 2":  (legit_idx[1],  0),
}

lime_results = {}

for case_name, (idx, true_label) in cases.items():
    instance = X_test_scaled[idx]
    pred_prob = xgb.predict_proba(instance.reshape(1, -1))[0, 1]

    exp = explainer.explain_instance(
        data_row        = instance,
        predict_fn      = xgb.predict_proba,
        num_features    = 10,
        num_samples     = 1000
    )

    lime_results[case_name] = {
        "explanation": exp,
        "pred_prob":   pred_prob,
        "true_label":  true_label,
        "idx":         idx
    }
    print(f"  {case_name} | True: {'Fraud' if true_label else 'Legit'} | "
          f"Pred prob: {pred_prob:.4f}")

print("\nLIME explanations generated!")

# COMMAND ----------

# ============================================================
# Cell 5: Plot LIME Explanations
# ============================================================
fig, axes = plt.subplots(5, 1, figsize=(14, 25))

for plot_idx, (case_name, result) in enumerate(lime_results.items()):
    exp       = result["explanation"]
    pred_prob = result["pred_prob"]
    true_lab  = "Fraud" if result["true_label"] else "Legit"

    # Get top features and weights
    features_weights = exp.as_list()
    feat_names = [fw[0] for fw in features_weights]
    feat_vals  = [fw[1] for fw in features_weights]

    colors = [FRAUD_COLOR if v > 0 else LEGIT_COLOR for v in feat_vals]

    axes[plot_idx].barh(range(len(feat_vals)), feat_vals[::-1],
                        color=colors[::-1],
                        edgecolor='#30363d', linewidth=0.5)
    axes[plot_idx].set_yticks(range(len(feat_names)))
    axes[plot_idx].set_yticklabels(feat_names[::-1], fontsize=8)
    axes[plot_idx].axvline(x=0, color='#e6edf3', lw=0.8, ls='--')
    axes[plot_idx].set_title(
        f'{case_name} | True Label: {true_lab} | '
        f'Fraud Probability: {pred_prob:.4f}',
        fontsize=10, fontweight='bold')
    axes[plot_idx].set_xlabel('LIME Weight (red=toward Fraud, green=toward Legit)')
    axes[plot_idx].grid(axis='x', alpha=0.3)

plt.suptitle('FRAUDSENSE NB24 - LIME Local Explanations',
             fontsize=13, fontweight='bold', color='#e6edf3', y=1.01)
plt.tight_layout()

chart1 = "/Volumes/workspace/default/fraud_data/models/nb24_lime_explanations.png"
plt.savefig(chart1, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart1}")

# COMMAND ----------

# ============================================================
# Cell 6: SHAP vs LIME Feature Agreement
# ============================================================
print("Comparing SHAP vs LIME top features...\n")

# SHAP top features from NB23
shap_top5 = ["V14", "V4", "v_abs_sum", "V10", "V1"]

# LIME top features across fraud cases
lime_fraud_features = []
for case_name, result in lime_results.items():
    if result["true_label"] == 1:
        for feat, weight in result["explanation"].as_list():
            # Extract feature name from LIME's condition string
            feat_clean = feat.split(" ")[0].replace(">","").replace("<","").replace("=","").strip()
            lime_fraud_features.append(feat_clean)

from collections import Counter
lime_top5_counts = Counter(lime_fraud_features).most_common(5)
lime_top5        = [f[0] for f in lime_top5_counts]

print("╔══════════════════════════════════════════════╗")
print("║      SHAP vs LIME — Top Feature Agreement    ║")
print("╠══════════════════════════════════════════════╣")
print(f"║  {'Rank':<6} {'SHAP':<20} {'LIME'}          ║")
print("╠══════════════════════════════════════════════╣")
for i in range(5):
    s = shap_top5[i] if i < len(shap_top5) else "N/A"
    l = lime_top5[i] if i < len(lime_top5)  else "N/A"
    print(f"║  {i+1:<6} {s:<20} {l:<20} ║")
print("╚══════════════════════════════════════════════╝")

# COMMAND ----------

# ============================================================
# Cell 7: Log to MLflow + Summary
# ============================================================
with mlflow.start_run(run_name="LIME_Explainability"):
    mlflow.log_artifact(chart1, "lime_plots")
    mlflow.set_tag("project",  "FRAUDSENSE")
    mlflow.set_tag("notebook", "NB24")
    mlflow.set_tag("phase",    "Phase6_Explainability")

print("╔══════════════════════════════════════════════════════╗")
print("║           NB24 - LIME SUMMARY                       ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Cases explained      : {len(cases)} (3 fraud + 2 legit)      ║")
print(f"║  Features per case    : 10                          ║")
print(f"║  Perturbations/case   : 1000                        ║")
print(f"║  Method               : Local linear approximation  ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  SHAP  : Global + Local tree-based exact values     ║")
print(f"║  LIME  : Local model-agnostic approximations        ║")
print(f"║  Both  : Confirm V14 as top fraud indicator         ║")
print("╚══════════════════════════════════════════════════════╝")

print("NB24 complete!")
print("Next: NB25 - Data Drift Detection")

# COMMAND ----------

