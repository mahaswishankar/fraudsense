# Databricks notebook source
# ============================================================
# Cell 0: Install Dependencies
# ============================================================
import subprocess
packages = ["shap", "xgboost", "lightgbm", "scikit-learn", "imbalanced-learn", "mlflow"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"{pkg} installed" if result.returncode == 0 else f"{pkg} failed: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE - NB23: SHAP Explainability
# Cell 1: Banner + Setup
# ============================================================
print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB23: SHAP Explainability          ║
║     Phase 6 - Global + Local Explanations           ║
╚══════════════════════════════════════════════════════╝
""")

from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import shap
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble        import ExtraTreesClassifier
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
print("SHAP Explainability starting...")

# COMMAND ----------

# ============================================================
# Cell 2: Load Data + Train XGBoost for SHAP
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

# Train XGBoost — SHAP works natively with tree models
print("Training XGBoost for SHAP analysis...")
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
    eval_metric='auc', random_state=42, verbosity=0)
xgb.fit(X_train_sm, y_train_sm)

xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_scaled)[:, 1])
print(f"XGBoost AUC : {xgb_auc:.4f}")

# Use a sample for SHAP — full test set is too slow
# Stratified sample: 500 legit + all fraud
fraud_idx  = np.where(y_test == 1)[0]
legit_idx  = np.where(y_test == 0)[0][:500]
sample_idx = np.concatenate([fraud_idx, legit_idx])

X_shap    = X_test_scaled[sample_idx]
y_shap    = y_test[sample_idx]
X_shap_df = pd.DataFrame(X_shap, columns=feature_cols)

print(f"SHAP sample : {len(X_shap)} rows ({len(fraud_idx)} fraud + 500 legit)")
print("Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: Compute SHAP Values
# ============================================================
print("Computing SHAP values...")
print("Using TreeExplainer — exact values for tree models\n")

explainer   = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_shap)

print(f"SHAP values shape : {shap_values.shape}")
print(f"Features          : {len(feature_cols)}")
print("SHAP values computed!")

# COMMAND ----------

# ============================================================
# Cell 4: Global Explainability — Feature Importance
# ============================================================

# --- Plot 1: SHAP Summary Bar (mean absolute SHAP) ---
fig, ax = plt.subplots(figsize=(10, 8))
shap_mean = np.abs(shap_values).mean(axis=0)
sorted_idx = np.argsort(shap_mean)[::-1][:20]  # top 20

colors = [FRAUD_COLOR if 'v' in f.lower() else ACCENT_COLOR
          for f in np.array(feature_cols)[sorted_idx]]

ax.barh(range(20), shap_mean[sorted_idx][::-1],
        color=colors[::-1], edgecolor='#30363d', linewidth=0.5)
ax.set_yticks(range(20))
ax.set_yticklabels(np.array(feature_cols)[sorted_idx][::-1], fontsize=9)
ax.set_xlabel('Mean |SHAP Value|')
ax.set_title('FRAUDSENSE - Global Feature Importance (SHAP)\nTop 20 Features',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
chart1 = "/Volumes/workspace/default/fraud_data/models/nb23_shap_importance.png"
plt.savefig(chart1, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart1}")

# COMMAND ----------

# ============================================================
# Cell 5: SHAP Beeswarm Plot (Summary Plot)
# ============================================================
# Beeswarm shows both importance AND direction of each feature
fig, ax = plt.subplots(figsize=(10, 8))

shap.summary_plot(
    shap_values,
    X_shap_df,
    max_display=20,
    show=False,
    color_bar=True,
    plot_size=None
)

plt.title('FRAUDSENSE - SHAP Beeswarm Plot\nFeature Impact on Fraud Prediction',
          fontsize=12, fontweight='bold', color='#e6edf3')
plt.tight_layout()

chart2 = "/Volumes/workspace/default/fraud_data/models/nb23_shap_beeswarm.png"
plt.savefig(chart2, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart2}")

# COMMAND ----------

# ============================================================
# Cell 6: Local Explainability — Individual Fraud Cases
# ============================================================
print("Generating local explanations for individual fraud cases...\n")

# Pick 3 fraud cases from our sample
fraud_sample_idx = np.where(y_shap == 1)[0][:3]

fig, axes = plt.subplots(3, 1, figsize=(14, 15))

for plot_idx, sample_i in enumerate(fraud_sample_idx):
    shap_vals_i = shap_values[sample_i]
    sorted_fi   = np.argsort(np.abs(shap_vals_i))[::-1][:10]

    colors_local = [FRAUD_COLOR if v > 0 else LEGIT_COLOR
                    for v in shap_vals_i[sorted_fi]]

    axes[plot_idx].barh(
        range(10),
        shap_vals_i[sorted_fi][::-1],
        color=colors_local[::-1],
        edgecolor='#30363d', linewidth=0.5
    )
    axes[plot_idx].set_yticks(range(10))
    axes[plot_idx].set_yticklabels(
        np.array(feature_cols)[sorted_fi][::-1], fontsize=9)
    axes[plot_idx].axvline(x=0, color='#e6edf3', lw=0.8, ls='--')
    axes[plot_idx].set_title(
        f'Fraud Case {plot_idx+1} — Local SHAP Explanation\n'
        f'Predicted prob: {xgb.predict_proba(X_shap[sample_i:sample_i+1])[0,1]:.4f}',
        fontsize=11, fontweight='bold')
    axes[plot_idx].set_xlabel('SHAP Value (red=increases fraud prob)')
    axes[plot_idx].grid(axis='x', alpha=0.3)

plt.suptitle('FRAUDSENSE NB23 - Local Explanations (Individual Fraud Cases)',
             fontsize=13, fontweight='bold', color='#e6edf3', y=1.01)
plt.tight_layout()

chart3 = "/Volumes/workspace/default/fraud_data/models/nb23_shap_local.png"
plt.savefig(chart3, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart3}")

# COMMAND ----------

# ============================================================
# Cell 7: SHAP Dependence Plot — Top Feature
# ============================================================
top_feature_idx  = np.argmax(np.abs(shap_values).mean(axis=0))
top_feature_name = feature_cols[top_feature_idx]
print(f"Top feature by SHAP: {top_feature_name}")

fig, ax = plt.subplots(figsize=(10, 6))

sc = ax.scatter(
    X_shap_df[top_feature_name],
    shap_values[:, top_feature_idx],
    c=y_shap,
    cmap='RdYlGn_r',
    alpha=0.6,
    s=8
)
ax.axhline(y=0, color='#e6edf3', lw=0.8, ls='--')
ax.set_xlabel(f'{top_feature_name} (feature value)', fontsize=11)
ax.set_ylabel('SHAP Value', fontsize=11)
ax.set_title(f'SHAP Dependence Plot — {top_feature_name}\n'
             f'How {top_feature_name} affects fraud probability',
             fontsize=12, fontweight='bold')
plt.colorbar(sc, ax=ax, label='Actual Label (1=Fraud)')
ax.grid(alpha=0.3)
plt.tight_layout()

chart4 = "/Volumes/workspace/default/fraud_data/models/nb23_shap_dependence.png"
plt.savefig(chart4, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart4}")

# COMMAND ----------

# ============================================================
# Cell 8: Log to MLflow + Summary
# ============================================================
with mlflow.start_run(run_name="SHAP_Explainability"):
    mlflow.log_artifact(chart1, "shap_plots")
    mlflow.log_artifact(chart2, "shap_plots")
    mlflow.log_artifact(chart3, "shap_plots")
    mlflow.log_artifact(chart4, "shap_plots")

    # Log top 10 feature importances
    shap_mean_all = np.abs(shap_values).mean(axis=0)
    top10_idx     = np.argsort(shap_mean_all)[::-1][:10]
    for rank, idx in enumerate(top10_idx, 1):
        mlflow.log_metric(f"shap_rank{rank}_{feature_cols[idx]}",
                          float(shap_mean_all[idx]))

    mlflow.set_tag("project",  "FRAUDSENSE")
    mlflow.set_tag("notebook", "NB23")
    mlflow.set_tag("phase",    "Phase6_Explainability")

print("╔══════════════════════════════════════════════════════╗")
print("║           NB23 - SHAP SUMMARY                       ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Top 5 features by global SHAP importance:          ║")
top5 = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:5]
for rank, idx in enumerate(top5, 1):
    print(f"║  {rank}. {feature_cols[idx]:<20} : {shap_mean_all[idx]:.4f}              ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Global + Local explanations generated              ║")
print(f"║  4 SHAP plots saved to models volume                ║")
print("╚══════════════════════════════════════════════════════╝")

print("NB23 complete!")
print("Next: NB24 - LIME Explainability")

# COMMAND ----------

