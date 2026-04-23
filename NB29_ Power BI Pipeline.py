# Databricks notebook source
# ============================================================
# CELL 0 — INSTALL DEPENDENCIES
# ============================================================
import subprocess
packages = ["matplotlib", "seaborn"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"],
                            capture_output=True, text=True)
    print(f"{pkg} installed" if result.returncode == 0
          else f"{pkg} failed: {result.stderr}")

# COMMAND ----------

# ============================================================
# CELL 1 — BANNER + CONFIG
# ============================================================
import warnings
warnings.filterwarnings('ignore')

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from pyspark.sql import SparkSession

print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB29: Power BI / Tableau Pipeline  ║
║     Phase 7 - Business Intelligence Export          ║
╚══════════════════════════════════════════════════════╝
""")

BASE_PATH   = "/Volumes/workspace/default/fraud_data"
DASH_PATH   = f"{BASE_PATH}/dashboard"
MODELS_PATH = f"{BASE_PATH}/models"
EXPORT_PATH = f"{BASE_PATH}/exports"

FRAUD_COLOR  = '#ff4444'
LEGIT_COLOR  = '#00d4aa'
ACCENT_COLOR = '#f7931a'

plt.rcParams.update({
    'figure.facecolor': '#0d1117', 'axes.facecolor':  '#161b22',
    'axes.edgecolor':   '#30363d', 'axes.labelcolor': '#c9d1d9',
    'xtick.color':      '#8b949e', 'ytick.color':     '#8b949e',
    'text.color':       '#c9d1d9', 'grid.color':      '#21262d',
    'grid.alpha':       0.5,       'legend.facecolor':'#161b22',
    'legend.edgecolor': '#30363d', 'font.size':       10,
})

spark = SparkSession.builder.getOrCreate()
os.makedirs(EXPORT_PATH, exist_ok=True)

print(f"Export path: {EXPORT_PATH}")

# COMMAND ----------

# ============================================================
# CELL 2 — LOAD ALL DASHBOARD TABLES
# ============================================================
def load_delta(name):
    path = f"{DASH_PATH}/{name}"
    pdf  = spark.read.format("delta").load(path).toPandas()
    print(f"Loaded {name:30s} : {pdf.shape}")
    return pdf

fraud_by_time    = load_delta("fraud_by_time")
model_comparison = load_delta("model_comparison")
risk_summary     = load_delta("risk_summary")
shap_importance  = load_delta("shap_importance")
scored_txns      = load_delta("scored_transactions")

# ── Fix dtypes after string round-trip ────────────────────
model_comparison['auc_roc']      = pd.to_numeric(model_comparison['auc_roc'],   errors='coerce')
shap_importance['mean_shap']     = pd.to_numeric(shap_importance['mean_shap'],  errors='coerce')
shap_importance['rank']          = pd.to_numeric(shap_importance['rank'],       errors='coerce')
scored_txns['fraudsense_score']  = pd.to_numeric(scored_txns['fraudsense_score'], errors='coerce')
scored_txns['xgb_score']         = pd.to_numeric(scored_txns['xgb_score'],      errors='coerce')
fraud_by_time['fraud_rate_pct']  = pd.to_numeric(fraud_by_time['fraud_rate_pct'], errors='coerce')
fraud_by_time['total_txns']      = pd.to_numeric(fraud_by_time['total_txns'],   errors='coerce')

print("\nAll tables loaded and dtypes fixed")

# COMMAND ----------

# ============================================================
# CELL 3 — EXPORT CSV + PARQUET
# ============================================================
import os

def export_table(pdf, name):
    # ── Nuclear clean — rebuild from numpy to kill Spark metadata ──
    clean = pd.DataFrame(
        pdf.values,
        columns=pdf.columns
    ).infer_objects()
    
    csv_path = f"{EXPORT_PATH}/{name}.csv"
    pq_path  = f"{EXPORT_PATH}/{name}.parquet"
    
    clean.to_csv(csv_path,    index=False)
    clean.to_parquet(pq_path, index=False, engine='pyarrow')
    print(f"Exported {name:35s} → CSV + Parquet ({len(clean):,} rows)")

export_table(fraud_by_time,    "fraud_by_time")
export_table(model_comparison, "model_comparison")
export_table(risk_summary,     "risk_summary")
export_table(shap_importance,  "shap_importance")
export_table(
    scored_txns.sample(min(10000, len(scored_txns)), random_state=42),
    "scored_transactions_sample"
)

print(f"\nAll exports saved to: {EXPORT_PATH}")

# COMMAND ----------

# ============================================================
# CELL 4 — SUMMARY STATISTICS JSON
# ============================================================
top_shap = shap_importance.sort_values(
    'mean_shap', ascending=False
)['feature'].head(5).tolist()

summary_stats = {
    "report_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "project":    "FRAUDSENSE — Real-Time Financial Risk & Fraud Intelligence Platform",
    "university": "Bennett University — B.Tech CSE (Data Science)",
    "pipeline": {
        "total_notebooks": 29,
        "total_records":   7381119,
        "gold_features":   "40+",
        "datasets":        ["Kaggle CC Fraud (284K)", "PaySim Mobile (6.36M)"]
    },
    "best_model": {
        "name":    "Layer 1 Stacking (ET+RF+XGB+LGB → LR)",
        "auc_roc": 0.9840
    },
    "fraudsense": {
        "auc_roc":    0.9777,
        "recall":     0.8421,
        "precision":  0.8856,
        "f1_score":   0.8634,
        "vs_xgboost": "+0.23% AUC, +6.67% Recall",
        "formula":    "0.35×Stack + 0.25×CNN1D + 0.20×XGB + 0.20×IsoForest"
    },
    "top_shap_features":   top_shap,
    "risk_tiers": {
        "HIGH":     "fraud_score ≥ 0.8",
        "MEDIUM":   "fraud_score 0.5–0.8",
        "LOW":      "fraud_score 0.3–0.5",
        "VERY LOW": "fraud_score < 0.3"
    },
    "model_count": len(model_comparison),
    "streaming": {
        "architecture": "Micro-batch simulation → XGBoost scorer → Parquet output",
        "batches":      "10 x 500 transactions",
        "recall":       1.0,
        "precision":    0.9091
    }
}

json_path = f"{EXPORT_PATH}/fraudsense_summary.json"
with open(json_path, 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"Summary JSON saved: {json_path}")
print(json.dumps(summary_stats, indent=2))

# COMMAND ----------

# ============================================================
# CELL 5 — STATIC MATPLOTLIB DASHBOARD
# ============================================================
def build_dashboard(save_path):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Title bar ─────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor('#161b22')
    ax_title.text(0.5, 0.65,
                  'FRAUDSENSE — Intelligence Dashboard',
                  ha='center', va='center', fontsize=17,
                  color=ACCENT_COLOR, fontweight='bold',
                  transform=ax_title.transAxes)
    ax_title.text(0.5, 0.2,
                  f'22 Models  |  7.38M Records  |  '
                  f'FRAUDSENSE AUC: 0.9777  |  '
                  f'Generated {datetime.now().strftime("%Y-%m-%d")}',
                  ha='center', va='center', fontsize=9.5,
                  color='#8b949e', transform=ax_title.transAxes)
    ax_title.axis('off')

    # ── Plot 1: Model leaderboard top 10 ──────────────────
    ax1   = fig.add_subplot(gs[1, 0])
    top10 = model_comparison.sort_values(
        'auc_roc', ascending=True
    ).tail(10)
    bar_colors = [
        ACCENT_COLOR if 'FRAUDSENSE' in str(m)
        else FRAUD_COLOR if float(a) >= 0.97
        else LEGIT_COLOR
        for m, a in zip(top10['model'], top10['auc_roc'])
    ]
    ax1.barh(top10['model'], top10['auc_roc'].astype(float),
             color=bar_colors, height=0.6, edgecolor='none')
    ax1.set_xlim(0.87, 1.0)
    ax1.set_title('Model Leaderboard (Top 10)',
                  color='#c9d1d9', fontsize=9.5)
    ax1.set_xlabel('AUC-ROC', fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.grid(axis='x', alpha=0.3)

    # ── Plot 2: SHAP top features ─────────────────────────
    ax2      = fig.add_subplot(gs[1, 1])
    top_shap = shap_importance.sort_values(
        'mean_shap', ascending=False
    ).head(10).sort_values('mean_shap', ascending=True)
    ax2.barh(top_shap['feature'],
             top_shap['mean_shap'].astype(float),
             color=LEGIT_COLOR, height=0.6, edgecolor='none')
    ax2.set_title('Top 10 SHAP Features',
                  color='#c9d1d9', fontsize=9.5)
    ax2.set_xlabel('Mean |SHAP|', fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(axis='x', alpha=0.3)

    # ── Plot 3: Risk tier distribution ────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    tier_colors = {
        'HIGH':     FRAUD_COLOR,
        'MEDIUM':   ACCENT_COLOR,
        'LOW':      '#ffd700',
        'VERY LOW': LEGIT_COLOR
    }
    if len(risk_summary) > 0:
        tiers  = risk_summary['risk_tier'].astype(str).tolist()
        counts = pd.to_numeric(
            risk_summary['count'], errors='coerce'
        ).tolist()
        cols   = [tier_colors.get(t, LEGIT_COLOR) for t in tiers]
        bars   = ax3.bar(tiers, counts, color=cols,
                         edgecolor='none', width=0.5)
        for bar, val in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 10,
                     str(int(val)), ha='center', va='bottom',
                     color='#c9d1d9', fontsize=8)
    ax3.set_title('Risk Tier Distribution',
                  color='#c9d1d9', fontsize=9.5)
    ax3.set_ylabel('Transactions', fontsize=8)
    ax3.tick_params(labelsize=8)

    # ── Plot 4: Fraud score histogram ─────────────────────
    ax4    = fig.add_subplot(gs[2, 0])
    scores = pd.to_numeric(
        scored_txns['fraudsense_score'], errors='coerce'
    ).dropna()
    ax4.hist(scores, bins=50,
             color=LEGIT_COLOR, edgecolor='none', alpha=0.85)
    ax4.axvline(0.5, color=FRAUD_COLOR,  linewidth=1.5,
                linestyle='--', label='0.5')
    ax4.axvline(0.8, color=ACCENT_COLOR, linewidth=1.2,
                linestyle=':',  label='0.8')
    ax4.set_title('FRAUDSENSE Score Distribution',
                  color='#c9d1d9', fontsize=9.5)
    ax4.set_xlabel('Fraud Probability', fontsize=8)
    ax4.legend(fontsize=7)
    ax4.tick_params(labelsize=8)

    # ── Plot 5: Fraud rate over time ──────────────────────
    ax5 = fig.add_subplot(gs[2, 1:])
    if 'fraud_rate_pct' in fraud_by_time.columns:
        y_vals = pd.to_numeric(
            fraud_by_time['fraud_rate_pct'], errors='coerce'
        ).fillna(0).head(80)
        x_vals = range(len(y_vals))
        ax5.fill_between(x_vals, y_vals,
                         alpha=0.35, color=FRAUD_COLOR)
        ax5.plot(x_vals, y_vals,
                 color=FRAUD_COLOR, linewidth=1.5)
        ax5.axhline(y_vals.mean(), color=ACCENT_COLOR,
                    linewidth=1, linestyle='--',
                    label=f'Avg: {y_vals.mean():.3f}%')
    ax5.set_title('Fraud Rate Over Time (Synthetic Windows)',
                  color='#c9d1d9', fontsize=9.5)
    ax5.set_xlabel('Time Window', fontsize=8)
    ax5.set_ylabel('Fraud Rate (%)', fontsize=8)
    ax5.legend(fontsize=7)
    ax5.tick_params(labelsize=8)
    ax5.grid(axis='y', alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Dashboard saved: {save_path}")

DASH_IMG = f"{MODELS_PATH}/fraudsense_dashboard.png"
build_dashboard(DASH_IMG)

# COMMAND ----------

# ============================================================
# CELL 6 — BI CONNECTION GUIDE
# ============================================================
guide = """
╔══════════════════════════════════════════════════════════════╗
║        POWER BI / TABLEAU CONNECTION GUIDE                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  OPTION A — Connect via CSV/Parquet exports:                 ║
║    Power BI  : Get Data → Text/CSV or Parquet                ║
║    Tableau   : Connect → Text File or Parquet                ║
║    Location  : /Volumes/.../fraud_data/exports/              ║
║                                                              ║
║  OPTION B — Connect directly to Delta tables:                ║
║    Power BI  : Get Data → Databricks                         ║
║    Enter workspace URL + personal access token               ║
║    Select tables from dashboard/ catalog                     ║
║                                                              ║
║  KEY TABLES:                                                 ║
║    fraud_by_time         → Time series fraud rate chart      ║
║    model_comparison      → Model leaderboard visual          ║
║    shap_importance       → Feature importance bar chart      ║
║    risk_summary          → Risk tier breakdown               ║
║    scored_transactions   → Transaction drill-through         ║
║                                                              ║
║  SUMMARY JSON  : exports/fraudsense_summary.json            ║
║  DASHBOARD PNG : models/fraudsense_dashboard.png            ║
╚══════════════════════════════════════════════════════════════╝
"""
print(guide)

# COMMAND ----------

# ============================================================
# CELL 7 — RESULTS SUMMARY
# ============================================================
print("╔══════════════════════════════════════════════════════╗")
print("║          NB29 - RESULTS SUMMARY                     ║")
print("╠══════════════════════════════════════════════════════╣")
print("║  CSV exports           : 5 tables                   ║")
print("║  Parquet exports       : 5 tables                   ║")
print("║  Summary JSON          : fraudsense_summary.json    ║")
print("║  Static dashboard      : fraudsense_dashboard.png   ║")
print("║  BI connection options : CSV / Parquet / Delta      ║")
print("╠══════════════════════════════════════════════════════╣")
print("║                                                      ║")
print("║  FRAUDSENSE PIPELINE — COMPLETE                     ║")
print("║  NB01 → NB29 : All 29 notebooks done                ║")
print("║  Phase 1–7   : Data → ML → DL → Ensemble →         ║")
print("║                XAI → Streaming → BI                 ║")
print("║                                                      ║")
print("║  Best AUC    : 0.9840 (Layer 1 Stack)               ║")
print("║  FRAUDSENSE  : 0.9777 (+0.23% vs XGB baseline)      ║")
print("║  Records     : 7,381,119 processed                  ║")
print("║  Models      : 22 trained and logged to MLflow      ║")
print("╚══════════════════════════════════════════════════════╝")

# COMMAND ----------

