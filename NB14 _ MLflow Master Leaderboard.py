# Databricks notebook source
# ============================================================
# Cell 0: Install Dependencies
# ============================================================
import subprocess
packages = ["mlflow", "matplotlib", "pandas"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"✅ {pkg}" if result.returncode == 0 else f"❌ {pkg}: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE — NB14: MLflow Master Leaderboard
# Cell 1: Banner + Setup
# ============================================================
print("""
╔══════════════════════════════════════════════════════╗
║       FRAUDSENSE — NB14: MLflow Master Leaderboard  ║
║       Phase 3 Complete — All 16 Models Ranked        ║
╚══════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Dark theme (same as all previous NBs)
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

print("✅ All libraries loaded")
print("✅ MLflow Master Leaderboard ready")
print("\n🏆 Phase 3 complete — building master rankings...")

# COMMAND ----------

# ============================================================
# Cell 2: Define Master Leaderboard (All 16 Models)
# ============================================================

# All confirmed AUC-ROC scores from NB06 → NB13
ALL_MODELS = [
    # (Model Name,              AUC-ROC, Notebook, Category)
    ("Extra Trees",             0.9786,  "NB10",  "Ensemble"),
    ("XGBoost",                 0.9765,  "NB12",  "Boosting"),
    ("GradientBoosting",        0.9764,  "NB11",  "Boosting"),
    ("LightGBM",                0.9724,  "NB12",  "Boosting"),
    ("TabNet",                  0.9692,  "NB13",  "Neural Net"),
    ("HistGradientBoosting",    0.9711,  "NB11",  "Boosting"),
    ("AdaBoost",                0.9710,  "NB11",  "Boosting"),
    ("CatBoost",                0.9681,  "NB12",  "Boosting"),
    ("Random Forest",           0.9678,  "NB10",  "Ensemble"),
    ("Logistic Regression",     0.9632,  "NB08",  "Linear"),
    ("Linear SVM",              0.9570,  "NB08",  "Linear"),
    ("Gaussian Naive Bayes",    0.9500,  "NB09",  "Probabilistic"),
    ("Isolation Forest (CC)",   0.9515,  "NB06",  "Anomaly"),
    ("Isolation Forest (PS)",   0.9282,  "NB06",  "Anomaly"),
    ("KNN",                     0.8880,  "NB09",  "Instance"),
    ("MLP",                     0.9337,  "NB13",  "Neural Net"),
]

# Sort by AUC-ROC descending
ALL_MODELS_SORTED = sorted(ALL_MODELS, key=lambda x: x[1], reverse=True)

print("╔══════════════════════════════════════════════════════════════╗")
print("║           FRAUDSENSE — FULL MODEL LEADERBOARD               ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  {'Rank':<5} {'Model':<28} {'AUC-ROC':<10} {'NB':<6} {'Category'}   ║")
print("╠══════════════════════════════════════════════════════════════╣")

medals = {1: "🥇", 2: "🥈", 3: "🥉"}
for i, (name, auc, nb, cat) in enumerate(ALL_MODELS_SORTED, 1):
    medal = medals.get(i, "  ")
    tag   = "← NEW" if nb == "NB13" else ""
    print(f"║  {i:<5} {medal} {name:<26} {auc:.4f}     {nb:<6} {cat:<12} {tag:<6} ║")

print("╚══════════════════════════════════════════════════════════════╝")
print(f"\n✅ Total models evaluated : {len(ALL_MODELS_SORTED)}")
print(f"✅ Best model             : {ALL_MODELS_SORTED[0][0]} @ {ALL_MODELS_SORTED[0][1]:.4f}")
print(f"✅ Models above 0.97      : {sum(1 for _,a,_,_ in ALL_MODELS_SORTED if a >= 0.97)}")
print(f"✅ Models above 0.95      : {sum(1 for _,a,_,_ in ALL_MODELS_SORTED if a >= 0.95)}")

# COMMAND ----------

# ============================================================
# Cell 3: Log All Models to MLflow
# ============================================================
EXPERIMENT_NAME = "/FRAUDSENSE_Master_Experiment"

try:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✅ MLflow experiment set: {EXPERIMENT_NAME}")
except Exception as e:
    print(f"⚠️  MLflow setup issue: {e}")

print("\n📝 Logging all models to MLflow...")
print("-" * 50)

logged = 0
for i, (name, auc, nb, cat) in enumerate(ALL_MODELS_SORTED, 1):
    run_name = f"{name.replace(' ', '_')}_{nb}"
    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_metric("auc_roc",       auc)
            mlflow.log_metric("rank",          i)
            mlflow.log_param("model_name",     name)
            mlflow.log_param("notebook",       nb)
            mlflow.log_param("category",       cat)
            mlflow.set_tag("project",          "FRAUDSENSE")
            mlflow.set_tag("phase",            "Phase3_Complete")
        print(f"  ✅ {name:<28} | AUC: {auc:.4f} | {nb}")
        logged += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")

print(f"\n✅ Logged {logged}/{len(ALL_MODELS_SORTED)} models to MLflow successfully")

# COMMAND ----------

# ============================================================
# Cell 4: Horizontal Bar Chart — Full Leaderboard
# ============================================================

CATEGORY_COLORS = {
    "Boosting":    ACCENT_COLOR,   # orange
    "Ensemble":    '#00d4aa',      # teal
    "Neural Net":  '#a78bfa',      # purple
    "Linear":      '#38bdf8',      # blue
    "Probabilistic":'#94a3b8',     # slate
    "Instance":    '#64748b',      # gray
    "Anomaly":     FRAUD_COLOR,    # red
}

names  = [m[0] for m in ALL_MODELS_SORTED]
scores = [m[1] for m in ALL_MODELS_SORTED]
cats   = [m[3] for m in ALL_MODELS_SORTED]
colors = [CATEGORY_COLORS[c] for c in cats]

# Reverse for horizontal bar (best at top)
names_r  = names[::-1]
scores_r = scores[::-1]
colors_r = colors[::-1]
cats_r   = cats[::-1]

fig, ax = plt.subplots(figsize=(13, 9))

bars = ax.barh(names_r, scores_r, color=colors_r, height=0.65,
               edgecolor='#30363d', linewidth=0.5)

# Value labels on bars
for bar, val in zip(bars, scores_r):
    ax.text(bar.get_width() - 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='right',
            fontsize=8.5, color='#0d1117', fontweight='bold')

# Reference lines
ax.axvline(x=scores[0], color='#ff4444', lw=1.5, ls='--', alpha=0.8,
           label=f'Best: {scores[0]:.4f} ({names[0]})')
ax.axvline(x=0.95,      color='#30363d', lw=1.0, ls=':',  alpha=0.7,
           label='0.95 threshold')
ax.axvline(x=0.97,      color='#21262d', lw=1.0, ls=':',  alpha=0.7,
           label='0.97 threshold')

ax.set_xlabel("AUC-ROC Score", fontsize=11)
ax.set_title("FRAUDSENSE — Master Model Leaderboard (16 Models)\nPhase 3: Battle Royale Complete",
             fontsize=13, fontweight='bold', color='#e6edf3', pad=14)
ax.set_xlim(0.87, 1.005)
ax.grid(axis='x', alpha=0.3, linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Category legend
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in CATEGORY_COLORS.items()]
ax.legend(handles=legend_patches, loc='lower right', fontsize=8,
          facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')

plt.tight_layout()

chart_path = "/Volumes/workspace/default/fraud_data/models/nb14_master_leaderboard.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"\n✅ Chart saved → {chart_path}")

# COMMAND ----------

# ============================================================
# Cell 5: Category Summary Stats
# ============================================================

print("\n📊 PERFORMANCE BREAKDOWN BY CATEGORY")
print("=" * 52)

df = pd.DataFrame(ALL_MODELS_SORTED, columns=["Model","AUC","NB","Category"])

# Exclude anomaly detection (different scoring paradigm)
df_ml = df[df["Category"] != "Anomaly"]

for cat, grp in df_ml.groupby("Category"):
    print(f"\n  [{cat}]")
    print(f"    Models  : {len(grp)}")
    print(f"    Best    : {grp['AUC'].max():.4f}  ({grp.loc[grp['AUC'].idxmax(),'Model']})")
    print(f"    Average : {grp['AUC'].mean():.4f}")
    print(f"    Worst   : {grp['AUC'].min():.4f}  ({grp.loc[grp['AUC'].idxmin(),'Model']})")

print("\n" + "=" * 52)
print("\n🎯 KEY TAKEAWAYS:")
print(f"  → Boosting dominates   : 6/10 top spots are boosting models")
print(f"  → TabNet surprises     : 0.9692 — beats CatBoost + RF + LR")
print(f"  → MLP baseline solid   : 0.9337 — expected for vanilla MLP")
print(f"  → Next up (NB15-18)    : LSTM, CNN1D, GRU, TabTransformer")
print(f"  → Target for ensemble  : beat Extra Trees (0.9786) with FRAUDSENSE")

# COMMAND ----------

# ============================================================
# Cell 6: Save Leaderboard to Gold Delta Table
# ============================================================

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.sql import Row

schema = StructType([
    StructField("Rank",     IntegerType(), False),
    StructField("Model",    StringType(),  False),
    StructField("AUC_ROC",  DoubleType(),  False),
    StructField("Category", StringType(),  False),
    StructField("Notebook", StringType(),  False),
])

rows = [Row(Rank=i, Model=name, AUC_ROC=float(auc), Category=cat, Notebook=nb)
        for i, (name, auc, nb, cat) in enumerate(ALL_MODELS_SORTED, 1)]

df_spark = spark.createDataFrame(rows, schema=schema)

SAVE_PATH = "/Volumes/workspace/default/fraud_data/gold/model_leaderboard"
df_spark.write.mode("overwrite").format("delta").save(SAVE_PATH)

print(f"✅ Leaderboard saved to Gold layer → {SAVE_PATH}")
print()
df_spark.show(truncate=False)

# COMMAND ----------

# ============================================================
# Cell 7: Phase Summary + What's Next
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║              PHASE 3 COMPLETE — SUMMARY                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  NB08  LR + SVM          ✅   LR: 0.9632  |  SVM: 0.9570   ║
║  NB09  NB + KNN          ✅   NB: 0.9500  |  KNN: 0.8880   ║
║  NB10  RF + ExTrees      ✅   RF: 0.9678  |   ET: 0.9786   ║
║  NB11  Boost Trio        ✅  Ada: 0.9710  |   GB: 0.9764   ║
║  NB12  XGB+LGB+Cat       ✅  XGB: 0.9765  |  LGB: 0.9724   ║
║  NB13  TabNet + MLP      ✅  Tab: 0.9692  |  MLP: 0.9337   ║
║  NB14  MLflow Leaderboard✅  All 16 models logged + saved   ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  CHAMPION (so far) : Extra Trees @ 0.9786                    ║
╠══════════════════════════════════════════════════════════════╣
║  PHASE 4 — DEEP LEARNING (NB15-18)                          ║
║                                                              ║
║  NB15  LSTM          Sequential fraud patterns               ║
║  NB16  CNN 1D        Local pattern detection                 ║
║  NB17  GRU / RNN     Lightweight sequential                  ║
║  NB18  TabTransformer Attention on tabular data              ║
║                                                              ║
║  PHASE 5 — FRAUDSENSE ENSEMBLE (NB19-22)                    ║
║  Target: beat 0.9786 with 3-layer stacking                   ║
╚══════════════════════════════════════════════════════════════╝
""")

print("✅ NB14 complete — MLflow Master Leaderboard done!")
print("🚀 Next: NB15 — LSTM Sequential Fraud Patterns")

# COMMAND ----------

