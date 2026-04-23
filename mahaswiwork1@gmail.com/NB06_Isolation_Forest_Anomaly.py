# Databricks notebook source
# ============================================================
# FRAUDSENSE — NB06: Isolation Forest + PCA Anomaly Detection
# Cell 1: Banner + Setup
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║     FRAUDSENSE — NB06: Isolation Forest + PCA Anomaly       ║
║     Unsupervised Anomaly Detection on Financial Data        ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, precision_recall_curve,
                              average_precision_score)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

GOLD_PATH    = "/Volumes/workspace/default/fraud_data/gold"
MODELS_PATH  = "/Volumes/workspace/default/fraud_data/models"

# ── Plot styling ─────────────────────────────────────────────
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

print("✅ Libraries loaded")
print("✅ Isolation Forest + PCA ready")
print("\n🌲 Starting unsupervised anomaly detection...")

# COMMAND ----------

# ============================================================
# Cell 2: Prepare CreditCard Data for Isolation Forest
# ============================================================

print("📥 Loading CreditCard Gold dataset...")

cc = spark.read.format("delta").load(f"{GOLD_PATH}/creditcard").toPandas()

# ── Select V features + engineered features ──────────────────
v_cols      = [c for c in cc.columns if c.startswith('V')]
extra_feats = ['amount_log', 'amount_zscore', 'is_night',
               'v1_v2_interaction', 'v3_v4_interaction', 'v14_v17_interaction']
extra_feats = [f for f in extra_feats if f in cc.columns]
feature_cols = v_cols + extra_feats

X = cc[feature_cols].fillna(0).values
y = cc['Class'].values

# ── Scale features ───────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✅ Feature matrix shape : {X_scaled.shape}")
print(f"✅ Features used        : {len(feature_cols)} ({len(v_cols)} V-features + {len(extra_feats)} engineered)")
print(f"✅ Fraud cases          : {y.sum():,} ({y.mean()*100:.3f}%)")
print(f"\n🌲 Data ready for Isolation Forest training...")

# COMMAND ----------

# ============================================================
# Cell 3: Train Isolation Forest
# ============================================================

print("🌲 Training Isolation Forest...")
print("   contamination = 0.004 (matches ~0.4% fraud rate in dataset)\n")

# ── Train Isolation Forest ───────────────────────────────────
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.004,
    max_samples='auto',
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

iso_forest.fit(X_scaled)

# ── Get predictions and anomaly scores ───────────────────────
# IsolationForest returns: -1 = anomaly, 1 = normal
iso_preds  = iso_forest.predict(X_scaled)
iso_scores = iso_forest.decision_function(X_scaled)  # Higher = more normal
iso_binary = (iso_preds == -1).astype(int)           # Convert to 0/1

# ── Save anomaly scores back to dataframe ────────────────────
cc['iso_anomaly_score'] = iso_scores
cc['iso_prediction']    = iso_binary

# ── Evaluate against true labels ─────────────────────────────
auc_roc = roc_auc_score(y, -iso_scores)  # Negate: lower score = more anomalous
avg_prec = average_precision_score(y, -iso_scores)

print("✅ Isolation Forest trained!")
print(f"\n📊 Evaluation vs True Labels (unsupervised — no labels used in training):")
print(f"   AUC-ROC            : {auc_roc:.4f}")
print(f"   Avg Precision      : {avg_prec:.4f}")
print(f"   Anomalies detected : {iso_binary.sum():,}")
print(f"   True fraud cases   : {y.sum():,}")
print(f"\n{classification_report(y, iso_binary, target_names=['Legitimate', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 4: PCA 2D Visualization of Anomalies
# ============================================================

print("🔍 Running PCA for 2D anomaly visualization...")

# ── PCA to 2 components ──────────────────────────────────────
pca = PCA(n_components=2, random_state=42)

# Sample for visualization (full dataset too large to plot)
sample_size = 8000
sample_idx  = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sample    = X_scaled[sample_idx]
y_sample    = y[sample_idx]
iso_sample  = iso_binary[sample_idx]
scores_sample = iso_scores[sample_idx]

X_pca = pca.fit_transform(X_sample)

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# ── Plot 1: True Labels in PCA space ─────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
scatter_legit = ax1.scatter(X_pca[y_sample==0, 0], X_pca[y_sample==0, 1],
                             c=LEGIT_COLOR, alpha=0.3, s=8, label='Legitimate')
scatter_fraud = ax1.scatter(X_pca[y_sample==1, 0], X_pca[y_sample==1, 1],
                             c=FRAUD_COLOR, alpha=0.8, s=25, label='Fraud', zorder=5)
ax1.set_title('True Labels in PCA Space', fontsize=12, fontweight='bold', color='#e6edf3')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=10)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=10)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# ── Plot 2: Isolation Forest Predictions in PCA space ────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(X_pca[iso_sample==0, 0], X_pca[iso_sample==0, 1],
            c=LEGIT_COLOR, alpha=0.3, s=8, label='Normal')
ax2.scatter(X_pca[iso_sample==1, 0], X_pca[iso_sample==1, 1],
            c=FRAUD_COLOR, alpha=0.8, s=25, label='Anomaly (IF)', zorder=5)
ax2.set_title('Isolation Forest Predictions in PCA Space', fontsize=12,
              fontweight='bold', color='#e6edf3')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=10)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# ── Plot 3: Anomaly Score Heatmap ────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
sc = ax3.scatter(X_pca[:, 0], X_pca[:, 1],
                 c=-scores_sample, cmap='RdYlGn_r',
                 alpha=0.5, s=8)
plt.colorbar(sc, ax=ax3, label='Anomaly Score (Higher = More Anomalous)')
ax3.set_title('Anomaly Score Heatmap in PCA Space', fontsize=12,
              fontweight='bold', color='#e6edf3')
ax3.set_xlabel(f'PC1', fontsize=10)
ax3.set_ylabel(f'PC2', fontsize=10)
ax3.grid(alpha=0.3)

# ── Plot 4: Anomaly Score Distribution ───────────────────────
ax4 = fig.add_subplot(gs[1, 1])
fraud_scores = -iso_scores[y == 1]
legit_scores = -iso_scores[y == 0]
ax4.hist(legit_scores, bins=80, color=LEGIT_COLOR, alpha=0.6,
         label='Legitimate', density=True)
ax4.hist(fraud_scores, bins=80, color=FRAUD_COLOR, alpha=0.7,
         label='Fraud', density=True)
ax4.axvline(x=0, color=ACCENT_COLOR, linestyle='--', linewidth=2, label='Decision Boundary')
ax4.set_title('Anomaly Score Distribution by Class', fontsize=12,
              fontweight='bold', color='#e6edf3')
ax4.set_xlabel('Anomaly Score', fontsize=10)
ax4.set_ylabel('Density', fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

fig.suptitle('FRAUDSENSE — Isolation Forest Anomaly Detection Analysis',
             fontsize=15, fontweight='bold', color='#e6edf3', y=1.01)

plt.savefig('/tmp/isolation_forest_pca.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()

print(f"✅ PCA explained variance: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"✅ Total variance captured: {sum(pca.explained_variance_ratio_)*100:.1f}%")

# COMMAND ----------

# ============================================================
# Cell 5: Isolation Forest on PaySim Dataset
# ============================================================

print("🌲 Running Isolation Forest on PaySim dataset...")

paysim = spark.read.format("delta").load(f"{GOLD_PATH}/paysim").toPandas()

# ── Feature selection for PaySim ─────────────────────────────
paysim_features = [
    'Amount', 'amount_log', 'amount_zscore',
    'balance_diff_orig', 'balance_diff_dest', 'balance_ratio_orig',
    'orig_balance_zero', 'dest_balance_zero',
    'tx_count_3steps', 'tx_count_10steps',
    'amount_mean_10steps', 'amount_std_10steps',
    'is_transfer', 'is_cash_out', 'high_risk_type',
    'sudden_large_tx', 'account_drain', 'amount_spike'
]
paysim_features = [f for f in paysim_features if f in paysim.columns]

X_ps = paysim[paysim_features].fillna(0).values
y_ps = paysim['Class'].values

scaler_ps = StandardScaler()
X_ps_scaled = scaler_ps.fit_transform(X_ps)

# ── Train ─────────────────────────────────────────────────────
iso_paysim = IsolationForest(
    n_estimators=200,
    contamination=0.0013,  # ~0.13% fraud rate in PaySim
    random_state=42,
    n_jobs=-1
)
iso_paysim.fit(X_ps_scaled)

paysim_preds  = iso_paysim.predict(X_ps_scaled)
paysim_scores = iso_paysim.decision_function(X_ps_scaled)
paysim_binary = (paysim_preds == -1).astype(int)

auc_ps   = roc_auc_score(y_ps, -paysim_scores)
avgp_ps  = average_precision_score(y_ps, -paysim_scores)

print(f"✅ PaySim Isolation Forest done!")
print(f"   AUC-ROC        : {auc_ps:.4f}")
print(f"   Avg Precision  : {avgp_ps:.4f}")
print(f"   Features used  : {len(paysim_features)}")
print(f"\n{classification_report(y_ps, paysim_binary, target_names=['Legitimate', 'Fraud'])}")

# COMMAND ----------

# ============================================================
# Cell 6: Save IF Scores → FRAUDSENSE Ensemble Input
# ============================================================

print("💾 Saving Isolation Forest scores for FRAUDSENSE ensemble...")

# ── Save CreditCard IF scores to Gold ────────────────────────
cc_if_scores = spark.createDataFrame(
    pd.DataFrame({
        'iso_anomaly_score' : iso_scores,
        'iso_prediction'    : iso_binary,
        'true_label'        : y
    })
)

cc_if_scores.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{GOLD_PATH}/iso_forest_scores_cc")

# ── Save PaySim IF scores ─────────────────────────────────────
ps_if_scores = spark.createDataFrame(
    pd.DataFrame({
        'iso_anomaly_score' : paysim_scores,
        'iso_prediction'    : paysim_binary,
        'true_label'        : y_ps
    })
)

ps_if_scores.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{GOLD_PATH}/iso_forest_scores_ps")

print("✅ CreditCard IF scores saved!")
print("✅ PaySim IF scores saved!")
print(f"\n📊 FRAUDSENSE Ensemble Component — Isolation Forest:")
print(f"   CreditCard AUC-ROC : {auc_roc:.4f}")
print(f"   PaySim AUC-ROC     : {auc_ps:.4f}")
print(f"   Weight in FRAUDSENSE formula : 0.20 (20%)")
print(f"\n✅ NB06 Complete — Unsupervised anomaly detection done!")
print(f"🚀 Next → NB07: Graph Fraud Detection (PageRank + Connected Components)")

# COMMAND ----------

