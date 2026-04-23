# Databricks notebook source
# ============================================================
# CELL 0 — INSTALL DEPENDENCIES
# ============================================================
import subprocess
packages = ["xgboost", "scikit-learn", "imbalanced-learn", "shap"]
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

import os, pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB28: Gold Export + Dashboard Prep ║
║     Phase 7 - BI Pipeline Preparation               ║
╚══════════════════════════════════════════════════════╝
""")

BASE_PATH   = "/Volumes/workspace/default/fraud_data"
GOLD_CC     = f"{BASE_PATH}/gold/creditcard"
MODELS_PATH = f"{BASE_PATH}/models"
DASH_PATH   = f"{BASE_PATH}/dashboard"

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
os.makedirs(DASH_PATH, exist_ok=True)

print(f"Dashboard output : {DASH_PATH}")

# COMMAND ----------

# ============================================================
# CELL 2 — LOAD GOLD DATA + TRAIN MODELS
# ============================================================
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, IsolationForest
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

cc = spark.read.format("delta").load(GOLD_CC).toPandas()

v_cols    = [c for c in cc.columns if c.startswith('V')]
eng_feats = ['amount_log', 'amount_zscore', 'amount_spike',
             'is_night', 'tx_velocity_10', 'high_amount_flag',
             'v1_v2_interaction', 'v3_v4_interaction',
             'v14_v17_interaction', 'v_sum_top5', 'v_abs_sum']
eng_feats    = [f for f in eng_feats if f in cc.columns]
feature_cols = v_cols + eng_feats

# ── Pure numpy rebuild to avoid Spark metadata ────────────
X_raw = cc[feature_cols].fillna(0).values.astype(np.float64)
y_raw = cc['Class'].values.astype(int)
cc_clean = pd.DataFrame(X_raw, columns=feature_cols)
cc_clean['Class'] = y_raw

X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

scaler     = StandardScaler()
X_tr_sc    = scaler.fit_transform(X_tr)
X_te_sc    = scaler.transform(X_te)
X_full_sc  = scaler.transform(X_raw)

smote             = SMOTE(random_state=42, sampling_strategy=0.1)
X_sm, y_sm        = smote.fit_resample(X_tr_sc, y_tr)

# ── XGBoost ───────────────────────────────────────────────
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
    eval_metric='auc', random_state=42, verbosity=0
)
xgb.fit(X_sm, y_sm)
xgb_scores = xgb.predict_proba(X_full_sc)[:, 1]
xgb_auc    = roc_auc_score(y_te, xgb.predict_proba(X_te_sc)[:, 1])

# ── Extra Trees ───────────────────────────────────────────
et = ExtraTreesClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
et.fit(X_sm, y_sm)
et_scores = et.predict_proba(X_full_sc)[:, 1]
et_auc    = roc_auc_score(y_te, et.predict_proba(X_te_sc)[:, 1])

# ── Isolation Forest ──────────────────────────────────────
iso = IsolationForest(
    n_estimators=100, contamination=0.01,
    random_state=42, n_jobs=-1
)
iso.fit(X_full_sc)
iso_raw   = iso.score_samples(X_full_sc)
iso_norm  = (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min())
iso_scores = 1 - iso_norm

# ── FRAUDSENSE ensemble score ─────────────────────────────
fraudsense_scores = (
    0.35 * xgb_scores +
    0.25 * xgb_scores +
    0.20 * xgb_scores +
    0.20 * iso_scores
)

print(f"XGBoost AUC      : {xgb_auc:.4f}")
print(f"Extra Trees AUC  : {et_auc:.4f}")
print(f"Features         : {len(feature_cols)}")
print("All models trained and scored")

# COMMAND ----------

# ============================================================
# CELL 3 — SHAP FEATURE IMPORTANCE
# ============================================================
print("Computing SHAP importance...")

explainer   = shap.TreeExplainer(xgb)
shap_sample = cc_clean[feature_cols].sample(
    min(3000, len(cc_clean)), random_state=42
).values.astype(np.float32)

shap_values = explainer.shap_values(shap_sample)

shap_importance = pd.DataFrame({
    'feature':   feature_cols,
    'mean_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_shap', ascending=False).reset_index(drop=True)
shap_importance['rank'] = shap_importance.index + 1

print("Top 10 SHAP features:")
print(shap_importance.head(10).to_string(index=False))

# COMMAND ----------

# ============================================================
# CELL 4 — BUILD DASHBOARD TABLES
# ============================================================

# ── Scored transactions table ─────────────────────────────
scored_df = cc_clean.copy()
scored_df['xgb_score']        = xgb_scores.astype(np.float32)
scored_df['et_score']         = et_scores.astype(np.float32)
scored_df['fraudsense_score'] = fraudsense_scores.astype(np.float32)
scored_df['xgb_flag']         = (scored_df['xgb_score'] >= 0.5).astype(int)
scored_df['fraudsense_flag']  = (scored_df['fraudsense_score'] >= 0.5).astype(int)
scored_df['risk_tier']        = pd.cut(
    scored_df['fraudsense_score'],
    bins=[0.0, 0.3, 0.5, 0.8, 1.0],
    labels=['VERY LOW', 'LOW', 'MEDIUM', 'HIGH'],
    include_lowest=True
).astype(str)

# ── Synthetic time windows ────────────────────────────────
n = len(scored_df)
scored_df['time_window'] = pd.date_range(
    start='2024-01-01', periods=n, freq='1min'
).strftime('%Y-%m-%d %H:00')

# ── Table 1: Fraud rate by time window ───────────────────
fraud_by_time = (
    scored_df.groupby('time_window')
             .agg(
                 total_txns    =('fraudsense_flag', 'count'),
                 fraud_txns    =('fraudsense_flag', 'sum'),
                 avg_score     =('fraudsense_score', 'mean')
             )
             .reset_index()
)
fraud_by_time['fraud_rate_pct'] = (
    fraud_by_time['fraud_txns'] / fraud_by_time['total_txns'] * 100
).round(4)

# ── Table 2: Model comparison ─────────────────────────────
model_comparison = pd.DataFrame({
    'model':   ['Layer1 Stack','CNN 1D','Extra Trees','TabTransformer',
                'FRAUDSENSE','XGBoost','GradientBoosting','LightGBM',
                'TabNet','HistGB','AdaBoost','CatBoost',
                'Random Forest','Logistic Regression','Linear SVM',
                'Isolation Forest CC','Gaussian NB','BiGRU',
                'Isolation Forest PS','BiLSTM','MLP','KNN'],
    'auc_roc': [0.9840,0.9799,0.9786,0.9778,0.9777,0.9765,0.9764,
                0.9724,0.9692,0.9711,0.9710,0.9681,0.9678,0.9632,
                0.9570,0.9515,0.9500,0.9479,0.9282,0.9259,0.9337,0.8880],
    'phase':   ['Ensemble','DL','ML','DL','Ensemble','ML','ML','ML',
                'DL','ML','ML','ML','ML','ML','ML',
                'Unsup','ML','DL','Unsup','DL','DL','ML'],
    'is_fraudsense_component': [0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
})

# ── Table 3: Risk tier summary ────────────────────────────
risk_summary = (
    scored_df.groupby('risk_tier', observed=True)
             .agg(
                 count        =('fraudsense_flag', 'count'),
                 actual_fraud =('Class', 'sum'),
                 avg_score    =('fraudsense_score', 'mean')
             )
             .reset_index()
)

print(f"fraud_by_time    : {fraud_by_time.shape}")
print(f"model_comparison : {model_comparison.shape}")
print(f"risk_summary     :\n{risk_summary.to_string(index=False)}")
print(f"shap_importance  : {shap_importance.shape}")
print(f"scored_df        : {scored_df.shape}")

# COMMAND ----------

# ============================================================
# CELL 5 — SAVE ALL TABLES AS DELTA
# ============================================================
def save_as_delta(pdf, name):
    path = f"{DASH_PATH}/{name}"
    sdf  = spark.createDataFrame(pdf.astype(str))
    sdf.write.format("delta").mode("overwrite").save(path)
    print(f"Saved {name:30s} ({len(pdf):,} rows) → {path}")

save_as_delta(fraud_by_time,    "fraud_by_time")
save_as_delta(model_comparison, "model_comparison")
save_as_delta(risk_summary,     "risk_summary")
save_as_delta(shap_importance,  "shap_importance")
save_as_delta(
    scored_df[['Class', 'xgb_score', 'et_score',
               'fraudsense_score', 'fraudsense_flag',
               'risk_tier', 'time_window']],
    "scored_transactions"
)

print("\nAll dashboard Delta tables saved!")

# COMMAND ----------

# ============================================================
# CELL 6 — RESULTS SUMMARY
# ============================================================
print("╔══════════════════════════════════════════════════════╗")
print("║          NB28 - RESULTS SUMMARY                     ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Records scored        : {len(scored_df):,}                  ║")
print(f"║  FRAUDSENSE flags      : {int(scored_df['fraudsense_flag'].sum()):,}                    ║")
print(f"║  XGBoost AUC           : {xgb_auc:.4f}                    ║")
print(f"║  Extra Trees AUC       : {et_auc:.4f}                    ║")
print(f"║  SHAP top feature      : {shap_importance.iloc[0]['feature']}                   ║")
print(f"║  Dashboard tables      : 5 Delta tables             ║")
print(f"║    fraud_by_time                                    ║")
print(f"║    model_comparison                                 ║")
print(f"║    risk_summary                                     ║")
print(f"║    shap_importance                                  ║")
print(f"║    scored_transactions                              ║")
print(f"║  Output path           : dashboard/                 ║")
print("╚══════════════════════════════════════════════════════╝")

# COMMAND ----------

