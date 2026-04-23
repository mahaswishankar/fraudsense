# Databricks notebook source
# ============================================================
# Cell 0: Install Dependencies
# ============================================================
import subprocess
packages = ["scikit-learn", "scipy", "mlflow", "evidently"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"{pkg} installed" if result.returncode == 0 else f"{pkg} failed: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE - NB25: Data Drift Detection
# Cell 1: Banner + Setup
# ============================================================
print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB25: Data Drift Detection         ║
║     Phase 6 - KS Test + PSI Monitoring              ║
╚══════════════════════════════════════════════════════╝
""")

from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy.stats              import ks_2samp
from sklearn.preprocessing    import StandardScaler
from sklearn.model_selection  import train_test_split

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
DRIFT_COLOR    = '#f472b6'

print("All libraries loaded")
print("Drift Detection starting...")

# COMMAND ----------

# ============================================================
# Cell 2: Load Data + Create Reference/Production Split
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

# Reference = first 70% of data (training period)
# Production = last 30% of data (deployment period)
# This simulates real-world drift monitoring
split_idx  = int(len(X) * 0.7)
X_ref      = X[:split_idx]
X_prod     = X[split_idx:]
y_ref      = y[:split_idx]
y_prod     = y[split_idx:]

print(f"Reference set  : {len(X_ref):,} rows (training period)")
print(f"Production set : {len(X_prod):,} rows (deployment period)")
print(f"Ref fraud rate : {y_ref.mean()*100:.4f}%")
print(f"Prod fraud rate: {y_prod.mean()*100:.4f}%")
print("Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: KS Test — Feature-level Drift Detection
# ============================================================
print("Running Kolmogorov-Smirnov test on all features...\n")

# KS test checks if two samples come from the same distribution
# p-value < 0.05 = significant drift detected
ks_results = []

for feat in feature_cols:
    feat_idx   = feature_cols.index(feat)
    ref_vals   = X_ref[:, feat_idx]
    prod_vals  = X_prod[:, feat_idx]

    ks_stat, p_value = ks_2samp(ref_vals, prod_vals)
    drift_detected   = p_value < 0.05

    ks_results.append({
        "Feature":        feat,
        "KS_Statistic":   ks_stat,
        "P_Value":        p_value,
        "Drift_Detected": drift_detected
    })

df_ks = pd.DataFrame(ks_results).sort_values("KS_Statistic", ascending=False)

n_drifted = df_ks["Drift_Detected"].sum()
print(f"Features with drift (p<0.05) : {n_drifted}/{len(feature_cols)}")
print(f"\nTop 10 most drifted features:")
print(f"{'Feature':<25} {'KS Stat':<12} {'P-Value':<12} {'Drift'}")
print("-" * 55)
for _, row in df_ks.head(10).iterrows():
    flag = "DRIFT" if row["Drift_Detected"] else "ok"
    print(f"{row['Feature']:<25} {row['KS_Statistic']:.4f}       "
          f"{row['P_Value']:.6f}   {flag}")

# COMMAND ----------

# ============================================================
# Cell 4: PSI — Population Stability Index
# ============================================================
print("Computing Population Stability Index (PSI)...\n")

def compute_psi(reference, production, bins=10):
    """
    PSI < 0.1  : No significant drift
    PSI 0.1-0.2: Moderate drift — monitor
    PSI > 0.2  : Significant drift — retrain
    """
    ref_min  = min(reference.min(), production.min())
    ref_max  = max(reference.max(), production.max())
    bin_edges = np.linspace(ref_min, ref_max, bins + 1)

    ref_counts  = np.histogram(reference,  bins=bin_edges)[0]
    prod_counts = np.histogram(production, bins=bin_edges)[0]

    # Avoid division by zero
    ref_pct  = (ref_counts  + 1e-6) / (len(reference)  + 1e-6 * bins)
    prod_pct = (prod_counts + 1e-6) / (len(production) + 1e-6 * bins)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return psi

psi_results = []
for feat in feature_cols:
    feat_idx = feature_cols.index(feat)
    psi_val  = compute_psi(X_ref[:, feat_idx], X_prod[:, feat_idx])

    if psi_val < 0.1:
        status = "Stable"
    elif psi_val < 0.2:
        status = "Monitor"
    else:
        status = "RETRAIN"

    psi_results.append({
        "Feature": feat,
        "PSI":     psi_val,
        "Status":  status
    })

df_psi = pd.DataFrame(psi_results).sort_values("PSI", ascending=False)

print(f"{'Feature':<25} {'PSI':<10} {'Status'}")
print("-" * 45)
for _, row in df_psi.head(15).iterrows():
    print(f"{row['Feature']:<25} {row['PSI']:.4f}     {row['Status']}")

print(f"\nStable  (PSI<0.1)  : {(df_psi['Status']=='Stable').sum()}")
print(f"Monitor (PSI<0.2)  : {(df_psi['Status']=='Monitor').sum()}")
print(f"Retrain (PSI>0.2)  : {(df_psi['Status']=='RETRAIN').sum()}")

# COMMAND ----------

# ============================================================
# Cell 5: Fraud Rate Drift Over Time
# ============================================================
print("Analyzing fraud rate drift over time...\n")

# Split into time windows and track fraud rate
N_WINDOWS  = 10
window_size = len(X) // N_WINDOWS

window_fraud_rates = []
window_labels      = []

for i in range(N_WINDOWS):
    start = i * window_size
    end   = start + window_size
    y_win = y[start:end]
    window_fraud_rates.append(y_win.mean() * 100)
    window_labels.append(f"W{i+1}")

ref_fraud_rate = np.mean(window_fraud_rates[:7])

print(f"Reference fraud rate (W1-W7) : {ref_fraud_rate:.4f}%")
print(f"Production fraud rate (W8-W10): {np.mean(window_fraud_rates[7:]):.4f}%")

# COMMAND ----------

# ============================================================
# Cell 6: Visualization
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: KS Statistics for top 20 features
top20_ks = df_ks.head(20)
colors_ks = [FRAUD_COLOR if d else LEGIT_COLOR
             for d in top20_ks["Drift_Detected"]]
axes[0,0].barh(range(20), top20_ks["KS_Statistic"].values[::-1],
               color=colors_ks[::-1], edgecolor='#30363d', linewidth=0.5)
axes[0,0].set_yticks(range(20))
axes[0,0].set_yticklabels(top20_ks["Feature"].values[::-1], fontsize=8)
axes[0,0].axvline(x=0.05, color=ACCENT_COLOR, lw=1.5, ls='--',
                  label='Drift threshold')
axes[0,0].set_title('KS Test — Top 20 Features', fontsize=11, fontweight='bold')
axes[0,0].set_xlabel('KS Statistic')
axes[0,0].legend(fontsize=8, facecolor='#161b22',
                 edgecolor='#30363d', labelcolor='#e6edf3')
axes[0,0].grid(axis='x', alpha=0.3)

# Plot 2: PSI for top 20 features
top20_psi  = df_psi.head(20)
colors_psi = []
for s in top20_psi["Status"].values:
    if s == "RETRAIN":  colors_psi.append(FRAUD_COLOR)
    elif s == "Monitor": colors_psi.append(ACCENT_COLOR)
    else:                colors_psi.append(LEGIT_COLOR)

axes[0,1].barh(range(20), top20_psi["PSI"].values[::-1],
               color=colors_psi[::-1], edgecolor='#30363d', linewidth=0.5)
axes[0,1].set_yticks(range(20))
axes[0,1].set_yticklabels(top20_psi["Feature"].values[::-1], fontsize=8)
axes[0,1].axvline(x=0.1, color=ACCENT_COLOR, lw=1.5, ls='--', label='Monitor (0.1)')
axes[0,1].axvline(x=0.2, color=FRAUD_COLOR,  lw=1.5, ls='--', label='Retrain (0.2)')
axes[0,1].set_title('PSI — Population Stability Index', fontsize=11, fontweight='bold')
axes[0,1].set_xlabel('PSI Value')
axes[0,1].legend(fontsize=8, facecolor='#161b22',
                 edgecolor='#30363d', labelcolor='#e6edf3')
axes[0,1].grid(axis='x', alpha=0.3)

# Plot 3: Fraud rate over time windows
colors_win = [LEGIT_COLOR if i < 7 else FRAUD_COLOR
              for i in range(N_WINDOWS)]
axes[1,0].bar(window_labels, window_fraud_rates,
              color=colors_win, edgecolor='#30363d', linewidth=0.5)
axes[1,0].axhline(y=ref_fraud_rate, color=ACCENT_COLOR, lw=1.5,
                  ls='--', label=f'Ref rate: {ref_fraud_rate:.4f}%')
axes[1,0].set_title('Fraud Rate Drift Over Time Windows',
                    fontsize=11, fontweight='bold')
axes[1,0].set_xlabel('Time Window')
axes[1,0].set_ylabel('Fraud Rate (%)')
axes[1,0].legend(fontsize=8, facecolor='#161b22',
                 edgecolor='#30363d', labelcolor='#e6edf3')
axes[1,0].grid(axis='y', alpha=0.3)

# Plot 4: PSI status distribution pie
status_counts = df_psi["Status"].value_counts()
colors_pie    = [LEGIT_COLOR, ACCENT_COLOR, FRAUD_COLOR]
axes[1,1].pie(status_counts.values,
              labels=status_counts.index,
              colors=colors_pie[:len(status_counts)],
              autopct='%1.1f%%',
              textprops={'color': '#e6edf3'})
axes[1,1].set_title('PSI Status Distribution\nAll Features',
                    fontsize=11, fontweight='bold')

plt.suptitle('FRAUDSENSE NB25 - Data Drift Detection Report',
             fontsize=13, fontweight='bold', color='#e6edf3', y=1.01)
plt.tight_layout()

chart1 = "/Volumes/workspace/default/fraud_data/models/nb25_drift_detection.png"
plt.savefig(chart1, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart1}")

# COMMAND ----------

# ============================================================
# Cell 7: Log to MLflow + Summary
# ============================================================
n_ks_drift = df_ks["Drift_Detected"].sum()
n_psi_retrain = (df_psi["Status"] == "RETRAIN").sum()
n_psi_monitor = (df_psi["Status"] == "Monitor").sum()

with mlflow.start_run(run_name="DataDrift_Detection"):
    mlflow.log_metric("n_features_ks_drift",    int(n_ks_drift))
    mlflow.log_metric("n_features_psi_retrain", int(n_psi_retrain))
    mlflow.log_metric("n_features_psi_monitor", int(n_psi_monitor))
    mlflow.log_metric("ref_fraud_rate",         ref_fraud_rate)
    mlflow.log_metric("prod_fraud_rate",        np.mean(window_fraud_rates[7:]))
    mlflow.log_artifact(chart1, "drift_plots")
    mlflow.set_tag("project",  "FRAUDSENSE")
    mlflow.set_tag("notebook", "NB25")
    mlflow.set_tag("phase",    "Phase6_Monitoring")

print("╔══════════════════════════════════════════════════════╗")
print("║           NB25 - DRIFT DETECTION SUMMARY            ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  KS Test — features with drift    : {n_ks_drift:<3}/{len(feature_cols)}           ║")
print(f"║  PSI — features stable            : {(df_psi['Status']=='Stable').sum():<3}/{len(feature_cols)}           ║")
print(f"║  PSI — features to monitor        : {n_psi_monitor:<3}/{len(feature_cols)}           ║")
print(f"║  PSI — features need retrain      : {n_psi_retrain:<3}/{len(feature_cols)}           ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Ref fraud rate   : {ref_fraud_rate:.4f}%                      ║")
print(f"║  Prod fraud rate  : {np.mean(window_fraud_rates[7:]):.4f}%                      ║")
print("╚══════════════════════════════════════════════════════╝")

print("NB25 complete!")
print("Next: NB26 - Auto PDF Report Generator")

# COMMAND ----------

