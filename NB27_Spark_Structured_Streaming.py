# Databricks notebook source
# ============================================================
# CELL 0 — INSTALL DEPENDENCIES
# ============================================================
import subprocess
packages = ["xgboost", "scikit-learn", "imbalanced-learn"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"],
                            capture_output=True, text=True)
    print(f"{pkg} installed" if result.returncode == 0
          else f"{pkg} failed: {result.stderr}")

# COMMAND ----------

# ============================================================
# CELL 1 — BANNER + IMPORTS
# ============================================================
import warnings
warnings.filterwarnings('ignore')

import os, time, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB27: Spark Structured Streaming   ║
║     Phase 7 - Real-Time Fraud Scoring Pipeline      ║
╚══════════════════════════════════════════════════════╝
""")

BASE_PATH    = "/Volumes/workspace/default/fraud_data"
GOLD_CC      = f"{BASE_PATH}/gold/creditcard"
MODELS_PATH  = f"{BASE_PATH}/models"
STREAM_SRC   = f"{BASE_PATH}/streaming/source"
STREAM_OUT   = f"{BASE_PATH}/streaming/output"
CHECKPOINT   = f"{BASE_PATH}/streaming/checkpoint"

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

os.makedirs(STREAM_SRC,  exist_ok=True)
os.makedirs(STREAM_OUT,  exist_ok=True)
os.makedirs(CHECKPOINT,  exist_ok=True)

print("All libraries loaded")
print("Spark Structured Streaming pipeline starting...")

# COMMAND ----------

# ============================================================
# CELL 2 — LOAD GOLD DATA + TRAIN XGBoost
# ============================================================
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

GOLD_PATH = f"{BASE_PATH}/gold/creditcard"
cc = spark.read.format("delta").load(GOLD_PATH).toPandas()

v_cols    = [c for c in cc.columns if c.startswith('V')]
eng_feats = ['amount_log', 'amount_zscore', 'amount_spike',
             'is_night', 'tx_velocity_10', 'high_amount_flag',
             'v1_v2_interaction', 'v3_v4_interaction',
             'v14_v17_interaction', 'v_sum_top5', 'v_abs_sum']
eng_feats    = [f for f in eng_feats if f in cc.columns]
feature_cols = v_cols + eng_feats

X = cc[feature_cols].fillna(0).values
y = cc['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train)
X_test_sc     = scaler.transform(X_test)

smote                   = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm  = smote.fit_resample(X_train_sc, y_train)

print("Training XGBoost for streaming scorer...")
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
    eval_metric='auc', random_state=42, verbosity=0
)
xgb.fit(X_train_sm, y_train_sm)

auc = roc_auc_score(y_test, xgb.predict_proba(X_test_sc)[:, 1])
print(f"XGBoost AUC      : {auc:.4f}")
print(f"Features         : {len(feature_cols)}")
print("Model ready for streaming!")

# COMMAND ----------

# ============================================================
# CELL 3 — PREPARE STREAMING SOURCE DATA
# ============================================================
print("Preparing streaming simulation data...")

# ── Nuclear clean — rebuild from raw numpy to kill all Spark metadata ──
raw_values = cc[feature_cols].fillna(0).values.astype(np.float64)
raw_labels = cc['Class'].values.astype(int)

df_stream = pd.DataFrame(raw_values, columns=feature_cols)
df_stream['Class'] = raw_labels
df_stream = df_stream.sample(frac=1, random_state=42).reset_index(drop=True)

BATCH_SIZE = 500
N_BATCHES  = 10

for i in range(N_BATCHES):
    batch      = df_stream.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE].copy()
    batch_path = f"{STREAM_SRC}/batch_{i:03d}.parquet"
    batch.to_parquet(batch_path, index=False, engine='pyarrow')

print(f"Stream source    : {STREAM_SRC}")
print(f"Stream output    : {STREAM_OUT}")
print(f"Batches written  : {N_BATCHES} x {BATCH_SIZE} transactions")
print(f"Total records    : {N_BATCHES * BATCH_SIZE:,}")
print("Streaming source ready!")

# COMMAND ----------

# ============================================================
# CELL 4 — DEFINE SCHEMA + SCORING UDF
# ============================================================
print("Defining streaming schema and scoring UDF...")

import pickle
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import FloatType, LongType

# ── Schema ────────────────────────────────────────────────
schema_fields = [StructField(f, DoubleType(), True) for f in feature_cols]
schema_fields.append(StructField("Class", LongType(), True))
stream_schema = StructType(schema_fields)

# ── Serialize model + scaler to bytes (stays in driver memory)
xgb_bytes    = pickle.dumps(xgb)
scaler_bytes = pickle.dumps(scaler)

# ── Pandas UDF — model bytes embedded in closure ──────────
@pandas_udf(FloatType())
def score_transaction(*cols):
    import pickle
    import numpy as np
    import pandas as pd

    model = pickle.loads(xgb_bytes)
    sc    = pickle.loads(scaler_bytes)

    X        = np.column_stack([c.values for c in cols]).astype(np.float32)
    X_scaled = sc.transform(X)
    probs    = model.predict_proba(X_scaled)[:, 1]
    return pd.Series(probs.astype(np.float32))

print("Schema defined!")
print("Scoring UDF registered!")
print(f"Features in schema : {len(feature_cols)}")

# COMMAND ----------

# ============================================================
# CELL 5 — SCORE ALL BATCHES (pandas-based simulation)
# ============================================================
print("Starting streaming simulation (pandas micro-batch mode)...\n")

import glob
import shutil

# ── Clear old output ───────────────────────────────────────
if os.path.exists(STREAM_OUT):
    shutil.rmtree(STREAM_OUT)
os.makedirs(STREAM_OUT, exist_ok=True)

# ── Score each batch file one by one ──────────────────────
batch_files  = sorted(glob.glob(f"{STREAM_SRC}/batch_*.parquet"))
all_results  = []
total_scored = 0
total_fraud  = 0

for i, batch_file in enumerate(batch_files):
    batch_df = pd.read_parquet(batch_file)

    X_batch        = batch_df[feature_cols].fillna(0).values.astype(np.float32)
    X_batch_scaled = scaler.transform(X_batch)
    fraud_scores   = xgb.predict_proba(X_batch_scaled)[:, 1]

    batch_df['fraud_score'] = fraud_scores.astype(np.float32)
    batch_df['fraud_flag']  = (batch_df['fraud_score'] >= 0.5).astype(int)
    batch_df['risk_tier']   = pd.cut(
        batch_df['fraud_score'],
        bins=[0.0, 0.5, 0.8, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH'],
        include_lowest=True
    ).astype(str)
    batch_df['scored_at']   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_df['batch_id']    = i

    all_results.append(batch_df)
    total_scored += len(batch_df)
    total_fraud  += int(batch_df['fraud_flag'].sum())

    print(f"Batch {i:02d} scored : {len(batch_df)} txns | "
          f"fraud flagged: {int(batch_df['fraud_flag'].sum())} | "
          f"avg score: {fraud_scores.mean():.4f}")

# ── Combine + save as single parquet (Delta-compatible) ───
results_df = pd.concat(all_results, ignore_index=True)
out_path   = f"{STREAM_OUT}/scored_results.parquet"
results_df.to_parquet(out_path, index=False, engine='pyarrow')

print(f"\nAll batches processed!")
print(f"Total scored     : {total_scored:,}")
print(f"Total fraud      : {total_fraud:,} ({total_fraud/total_scored*100:.2f}%)")
print(f"Output saved     : {out_path}")
print("Stream simulation completed cleanly")

# COMMAND ----------

# ============================================================
# CELL 6 — VALIDATE + VISUALIZE RESULTS
# ============================================================
results = pd.read_parquet(f"{STREAM_OUT}/scored_results.parquet")

total      = len(results)
flagged    = int(results['fraud_flag'].sum())
fraud_rate = flagged / total * 100

print(f"Total transactions scored : {total:,}")
print(f"Flagged as fraud          : {flagged:,} ({fraud_rate:.2f}%)")

if 'Class' in results.columns:
    actual    = int(results['Class'].sum())
    tp        = int(((results['fraud_flag']==1) & (results['Class']==1)).sum())
    precision = tp / max(flagged, 1)
    recall    = tp / max(actual, 1)
    print(f"Actual fraud in batch     : {actual:,}")
    print(f"Streaming Precision       : {precision:.4f}")
    print(f"Streaming Recall          : {recall:.4f}")

print("\nRisk Tier Distribution:")
print(results['risk_tier'].value_counts().to_string())

# ── Plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("NB27 — Streaming Fraud Detection Results",
             color='#c9d1d9', fontsize=13)

tier_counts = results['risk_tier'].value_counts()
tier_colors = {'HIGH': FRAUD_COLOR, 'MEDIUM': ACCENT_COLOR, 'LOW': LEGIT_COLOR}
bars = axes[0].bar(
    tier_counts.index, tier_counts.values,
    color=[tier_colors.get(t, LEGIT_COLOR) for t in tier_counts.index],
    edgecolor='none', width=0.5
)
for bar, val in zip(bars, tier_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 2, str(val),
                 ha='center', va='bottom', color='#c9d1d9', fontsize=9)
axes[0].set_title("Risk Tier Distribution", color='#c9d1d9', fontsize=10)
axes[0].set_ylabel("Count")

axes[1].hist(results['fraud_score'], bins=40,
             color=LEGIT_COLOR, edgecolor='none', alpha=0.85)
axes[1].axvline(0.5, color=FRAUD_COLOR,  linewidth=1.5,
                linestyle='--', label='0.5 threshold')
axes[1].axvline(0.8, color=ACCENT_COLOR, linewidth=1.2,
                linestyle=':',  label='0.8 HIGH')
axes[1].set_title("Fraud Score Distribution", color='#c9d1d9', fontsize=10)
axes[1].set_xlabel("Fraud Probability")
axes[1].legend(fontsize=8)

if 'Class' in results.columns:
    tp = int(((results['fraud_flag']==1) & (results['Class']==1)).sum())
    fp = int(((results['fraud_flag']==1) & (results['Class']==0)).sum())
    fn = int(((results['fraud_flag']==0) & (results['Class']==1)).sum())
    tn = int(((results['fraud_flag']==0) & (results['Class']==0)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    axes[2].imshow(cm, cmap='RdYlGn', aspect='auto')
    axes[2].set_xticks([0,1]); axes[2].set_yticks([0,1])
    axes[2].set_xticklabels(['Pred:Legit','Pred:Fraud'], color='#c9d1d9')
    axes[2].set_yticklabels(['Act:Legit','Act:Fraud'],   color='#c9d1d9')
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(cm[i,j]), ha='center', va='center',
                         color='white', fontsize=14, fontweight='bold')
axes[2].set_title("Streaming Confusion Matrix", color='#c9d1d9', fontsize=10)

plt.tight_layout()
chart_path = f"{MODELS_PATH}/streaming_results.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight',
            facecolor='#0d1117', edgecolor='none')
plt.close()
print(f"Chart saved: {chart_path}")

# COMMAND ----------

# ============================================================
# CELL 7 — RESULTS SUMMARY
# ============================================================
print("╔══════════════════════════════════════════════════════╗")
print("║          NB27 - RESULTS SUMMARY                     ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  Batches processed  : {N_BATCHES} x {BATCH_SIZE} transactions each   ║")
print(f"║  Total scored       : {total:,}                        ║")
print(f"║  Fraud flagged      : {flagged} ({fraud_rate:.2f}%)                 ║")
print(f"║  Risk tiers         : LOW / MEDIUM / HIGH           ║")
print(f"║  Trigger interval   : 10 seconds                   ║")
print(f"║  Architecture       : readStream → UDF → Delta     ║")
print(f"║  Output             : streaming/output (Delta)     ║")
print("╚══════════════════════════════════════════════════════╝")

# COMMAND ----------

