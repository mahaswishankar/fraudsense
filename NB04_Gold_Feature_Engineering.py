# Databricks notebook source
# ============================================================
# FRAUDSENSE — NB04: Silver → Gold (Feature Engineering)
# Cell 1: Banner + Path Configuration
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║       FRAUDSENSE — NB04: Gold Layer Feature Engineering     ║
║       Silver (Clean) → Gold (ML-Ready, Feature Rich)        ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *
import datetime

spark = SparkSession.builder.getOrCreate()

SILVER_PATH = "/Volumes/workspace/default/fraud_data/silver"
GOLD_PATH   = "/Volumes/workspace/default/fraud_data/gold"

print(f"✅ Spark ready")
print(f"✅ Silver source : {SILVER_PATH}")
print(f"✅ Gold target   : {GOLD_PATH}")
print(f"✅ Started at    : {datetime.datetime.now()}")
print("\n🥇 Beginning Gold Feature Engineering...")

# COMMAND ----------

# ============================================================
# Cell 2: CreditCard → Gold Feature Engineering
# ============================================================

print("⚙️  Engineering features for CreditCard dataset...")

cc = spark.read.format("delta").load(f"{SILVER_PATH}/creditcard")

# ── Window specs for velocity features ──────────────────────
# CreditCard has no account ID, so we use time-based windows only
time_window = Window.orderBy("Time").rowsBetween(-10, 0)

# ── Amount-based features ────────────────────────────────────
cc_gold = cc \
    .withColumn("amount_log",           F.log1p(F.col("Amount"))) \
    .withColumn("amount_squared",       F.pow(F.col("Amount"), 2)) \
    .withColumn("amount_rolling_mean",  F.avg("Amount").over(time_window)) \
    .withColumn("amount_rolling_std",   F.stddev("Amount").over(time_window)) \
    .withColumn("amount_zscore",        
                (F.col("Amount") - F.avg("Amount").over(time_window)) / 
                (F.stddev("Amount").over(time_window) + F.lit(1e-9))) \
    .withColumn("amount_rolling_max",   F.max("Amount").over(time_window)) \
    .withColumn("amount_spike",         
                (F.col("Amount") > F.avg("Amount").over(time_window) * 3).cast(IntegerType()))

# ── Time-based features ──────────────────────────────────────
cc_gold = cc_gold \
    .withColumn("hour_of_day",    (F.col("Time") % 86400 / 3600).cast(IntegerType())) \
    .withColumn("is_night",       ((F.col("hour_of_day") < 6) | (F.col("hour_of_day") > 22)).cast(IntegerType())) \
    .withColumn("is_weekend",     F.lit(0)) \
    .withColumn("tx_velocity_10", F.count("Amount").over(time_window))

# ── V-feature interactions (top correlated with fraud) ───────
cc_gold = cc_gold \
    .withColumn("v1_v2_interaction",  F.col("V1") * F.col("V2")) \
    .withColumn("v3_v4_interaction",  F.col("V3") * F.col("V4")) \
    .withColumn("v14_v17_interaction",F.col("V14") * F.col("V17")) \
    .withColumn("v_sum_top5",         F.col("V1") + F.col("V2") + F.col("V3") + F.col("V4") + F.col("V5")) \
    .withColumn("v_abs_sum",          sum([F.abs(F.col(f"V{i}")) for i in range(1, 29)]))

# ── Risk indicators ──────────────────────────────────────────
cc_gold = cc_gold \
    .withColumn("high_amount_flag",   (F.col("Amount") > 500).cast(IntegerType())) \
    .withColumn("very_high_amount",   (F.col("Amount") > 2000).cast(IntegerType())) \
    .withColumn("_gold_processed_at", F.current_timestamp()) \
    .withColumn("dataset_source",     F.lit("creditcard"))

# Write to Gold
cc_gold.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{GOLD_PATH}/creditcard")

print(f"✅ CreditCard Gold done!")
print(f"   Rows            : {cc_gold.count():,}")
print(f"   Features added  : amount_log, amount_zscore, amount_spike, hour_of_day,")
print(f"                     is_night, tx_velocity_10, V-interactions, risk flags")
print(f"   Total columns   : {len(cc_gold.columns)}")

# COMMAND ----------

# ============================================================
# Cell 3: PaySim → Gold Feature Engineering
# ============================================================

print("⚙️  Engineering features for PaySim dataset...")

paysim = spark.read.format("delta").load(f"{SILVER_PATH}/paysim")

# ── Window specs — PaySim has named accounts! ────────────────
account_window    = Window.partitionBy("nameOrig").orderBy("step")
account_window_3  = Window.partitionBy("nameOrig").orderBy("step").rowsBetween(-3, 0)
account_window_10 = Window.partitionBy("nameOrig").orderBy("step").rowsBetween(-10, 0)

# ── Balance-based features ───────────────────────────────────
paysim_gold = paysim \
    .withColumn("balance_diff_orig",    F.col("newbalanceOrig") - F.col("oldbalanceOrg")) \
    .withColumn("balance_diff_dest",    F.col("newbalanceDest") - F.col("oldbalanceDest")) \
    .withColumn("balance_ratio_orig",   
                F.col("amount") / (F.col("oldbalanceOrg") + F.lit(1e-9))) \
    .withColumn("orig_balance_zero",    (F.col("oldbalanceOrg") == 0).cast(IntegerType())) \
    .withColumn("dest_balance_zero",    (F.col("oldbalanceDest") == 0).cast(IntegerType())) \
    .withColumn("amount_log",           F.log1p(F.col("amount")))

# ── Velocity features per account ────────────────────────────
paysim_gold = paysim_gold \
    .withColumn("tx_count_3steps",      F.count("amount").over(account_window_3)) \
    .withColumn("tx_count_10steps",     F.count("amount").over(account_window_10)) \
    .withColumn("amount_mean_10steps",  F.avg("amount").over(account_window_10)) \
    .withColumn("amount_std_10steps",   F.stddev("amount").over(account_window_10)) \
    .withColumn("amount_max_10steps",   F.max("amount").over(account_window_10)) \
    .withColumn("amount_zscore",        
                (F.col("amount") - F.avg("amount").over(account_window_10)) /
                (F.stddev("amount").over(account_window_10) + F.lit(1e-9)))

# ── Transaction type encoding ────────────────────────────────
paysim_gold = paysim_gold \
    .withColumn("is_transfer",          (F.col("type") == "TRANSFER").cast(IntegerType())) \
    .withColumn("is_cash_out",          (F.col("type") == "CASH_OUT").cast(IntegerType())) \
    .withColumn("is_payment",           (F.col("type") == "PAYMENT").cast(IntegerType())) \
    .withColumn("is_cash_in",           (F.col("type") == "CASH_IN").cast(IntegerType())) \
    .withColumn("is_debit",             (F.col("type") == "DEBIT").cast(IntegerType()))

# ── Risk indicators ──────────────────────────────────────────
paysim_gold = paysim_gold \
    .withColumn("high_risk_type",       
                ((F.col("type") == "TRANSFER") | (F.col("type") == "CASH_OUT")).cast(IntegerType())) \
    .withColumn("sudden_large_tx",      
                (F.col("amount") > F.col("amount_mean_10steps") * 3).cast(IntegerType())) \
    .withColumn("account_drain",        
                ((F.col("balance_diff_orig") < -1000) & (F.col("orig_balance_zero") == 0)).cast(IntegerType())) \
    .withColumn("_gold_processed_at",   F.current_timestamp()) \
    .withColumn("dataset_source",       F.lit("paysim"))

# Rename amount → Amount for consistency
paysim_gold = paysim_gold.withColumnRenamed("amount", "Amount")

paysim_gold.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{GOLD_PATH}/paysim")

print(f"✅ PaySim Gold done!")
print(f"   Rows            : {paysim_gold.count():,}")
print(f"   Features added  : balance diffs, velocity (3/10 steps), type encoding,")
print(f"                     account_drain, sudden_large_tx, high_risk_type")
print(f"   Total columns   : {len(paysim_gold.columns)}")

# COMMAND ----------

# ============================================================
# Cell 4: IEEE-CIS Transaction + Identity → Gold (FIXED)
# ============================================================

print("⚙️  Engineering features for IEEE-CIS dataset...")

ieee_trans  = spark.read.format("delta").load(f"{SILVER_PATH}/ieee_transaction")
ieee_ident  = spark.read.format("delta").load(f"{SILVER_PATH}/ieee_identity")

# ── Drop metadata cols from identity before join to avoid duplicates ──
meta_cols = ["_source", "_silver_processed_at", "_ingestion_date", "_ingested_at", "_gold_processed_at"]
ieee_ident_clean = ieee_ident.drop(*[c for c in meta_cols if c in ieee_ident.columns])
ieee_trans_clean = ieee_trans.drop(*[c for c in meta_cols if c in ieee_trans.columns])

# ── Join Transaction + Identity on TransactionID ─────────────
ieee = ieee_trans_clean.join(ieee_ident_clean, on="TransactionID", how="left")
print(f"   Joined IEEE dataset: {ieee.count():,} rows, {len(ieee.columns)} columns")

# ── Window spec ──────────────────────────────────────────────
card_window    = Window.partitionBy("card1").orderBy("TransactionDT")
card_window_5  = Window.partitionBy("card1").orderBy("TransactionDT").rowsBetween(-5, 0)
card_window_20 = Window.partitionBy("card1").orderBy("TransactionDT").rowsBetween(-20, 0)

# ── Time-based features ──────────────────────────────────────
ieee_gold = ieee \
    .withColumn("tx_hour",          (F.col("TransactionDT") % 86400 / 3600).cast(IntegerType())) \
    .withColumn("tx_day",           (F.col("TransactionDT") / 86400).cast(IntegerType())) \
    .withColumn("is_night",         ((F.col("tx_hour") < 6) | (F.col("tx_hour") > 22)).cast(IntegerType())) \
    .withColumn("tx_day_of_week",   (F.col("tx_day") % 7).cast(IntegerType())) \
    .withColumn("is_weekend",       ((F.col("tx_day_of_week") >= 5)).cast(IntegerType()))

# ── Amount features ──────────────────────────────────────────
ieee_gold = ieee_gold \
    .withColumn("amount_log",           F.log1p(F.col("TransactionAmt"))) \
    .withColumn("amount_rolling_mean",  F.avg("TransactionAmt").over(card_window_5)) \
    .withColumn("amount_rolling_std",   F.stddev("TransactionAmt").over(card_window_5)) \
    .withColumn("amount_zscore",
                (F.col("TransactionAmt") - F.avg("TransactionAmt").over(card_window_20)) /
                (F.stddev("TransactionAmt").over(card_window_20) + F.lit(1e-9))) \
    .withColumn("amount_spike",
                (F.col("TransactionAmt") > F.avg("TransactionAmt").over(card_window_20) * 3).cast(IntegerType()))

# ── Card velocity features ───────────────────────────────────
ieee_gold = ieee_gold \
    .withColumn("card_tx_count_5",      F.count("TransactionAmt").over(card_window_5)) \
    .withColumn("card_tx_count_20",     F.count("TransactionAmt").over(card_window_20)) \
    .withColumn("card_amount_sum_5",    F.sum("TransactionAmt").over(card_window_5)) \
    .withColumn("card_amount_sum_20",   F.sum("TransactionAmt").over(card_window_20))

# ── Email domain features ────────────────────────────────────
ieee_gold = ieee_gold \
    .withColumn("p_email_domain_risk",
                F.when(F.col("P_emaildomain").isin(["gmail.com","yahoo.com","hotmail.com"]), 0)
                 .when(F.col("P_emaildomain").isNull(), 1)
                 .otherwise(2)) \
    .withColumn("r_email_domain_risk",
                F.when(F.col("R_emaildomain").isin(["gmail.com","yahoo.com","hotmail.com"]), 0)
                 .when(F.col("R_emaildomain").isNull(), 1)
                 .otherwise(2))

# ── Device & identity risk ───────────────────────────────────
ieee_gold = ieee_gold \
    .withColumn("has_identity",         (F.col("id_01").isNotNull()).cast(IntegerType())) \
    .withColumn("device_type_mobile",   (F.col("DeviceType") == "mobile").cast(IntegerType())) \
    .withColumn("high_amount_flag",     (F.col("TransactionAmt") > 500).cast(IntegerType())) \
    .withColumn("Amount",               F.col("TransactionAmt")) \
    .withColumn("_gold_processed_at",   F.current_timestamp()) \
    .withColumn("dataset_source",       F.lit("ieee_cis"))

# ── Add velocity placeholder for unified schema ───────────────
ieee_gold = ieee_gold.withColumn("tx_velocity_10", F.col("card_tx_count_20"))

ieee_gold.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{GOLD_PATH}/ieee")

fraud_count = ieee_gold.filter(F.col("Class") == 1).count()
print(f"✅ IEEE-CIS Gold done!")
print(f"   Rows            : {ieee_gold.count():,}")
print(f"   Fraud cases     : {fraud_count:,}")
print(f"   Features added  : time features, amount velocity, card velocity,")
print(f"                     email domain risk, device risk, identity flag")
print(f"   Total columns   : {len(ieee_gold.columns)}")

# COMMAND ----------

# ============================================================
# Cell 5: Create Unified Gold Master Table
# ============================================================

print("🔗 Creating Unified Gold Master table...")

# ── Load all 3 gold datasets ─────────────────────────────────
cc_gold     = spark.read.format("delta").load(f"{GOLD_PATH}/creditcard")
paysim_gold = spark.read.format("delta").load(f"{GOLD_PATH}/paysim")
ieee_gold   = spark.read.format("delta").load(f"{GOLD_PATH}/ieee")

# ── Select unified common feature set ────────────────────────
COMMON_FEATURES = [
    "Amount", "amount_log", "amount_zscore", "amount_spike",
    "is_night", "is_weekend", "high_amount_flag",
    "tx_velocity_10", "dataset_source", "Class"
]

def safe_select(df, features):
    """Select only columns that exist in the dataframe, fill missing with 0"""
    selected = []
    for f in features:
        if f in df.columns:
            selected.append(F.col(f))
        else:
            selected.append(F.lit(0).cast(DoubleType()).alias(f))
    return df.select(selected)

cc_unified     = safe_select(cc_gold, COMMON_FEATURES)
paysim_unified = safe_select(paysim_gold, COMMON_FEATURES)
ieee_unified   = safe_select(ieee_gold, COMMON_FEATURES)

# ── Union all three ──────────────────────────────────────────
master_gold = cc_unified.union(paysim_unified).union(ieee_unified)

# ── Add unified risk score placeholder ───────────────────────
master_gold = master_gold \
    .withColumn("_gold_processed_at", F.current_timestamp()) \
    .withColumn("row_id", F.monotonically_increasing_id())

master_gold.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{GOLD_PATH}/master")

total = master_gold.count()
fraud = master_gold.filter(F.col("Class") == 1).count()

print(f"✅ Unified Gold Master table created!")
print(f"   Total rows    : {total:,}")
print(f"   Fraud cases   : {fraud:,}")
print(f"   Fraud rate    : {fraud/total*100:.3f}%")
print(f"   Features      : {len(master_gold.columns)}")
print(f"   Location      : {GOLD_PATH}/master")

# COMMAND ----------

# ============================================================
# Cell 6: Gold Layer Health Check & Final Summary
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║              🥇 GOLD LAYER — HEALTH CHECK                   ║
╚══════════════════════════════════════════════════════════════╝
""")

gold_tables = {
    "CreditCard Gold"  : f"{GOLD_PATH}/creditcard",
    "PaySim Gold"      : f"{GOLD_PATH}/paysim",
    "IEEE-CIS Gold"    : f"{GOLD_PATH}/ieee",
    "Master Unified"   : f"{GOLD_PATH}/master",
}

print(f"{'Table':<22} {'Rows':>12} {'Fraud':>10} {'Fraud%':>8}  Columns  Status")
print("─" * 75)

for name, path in gold_tables.items():
    try:
        df = spark.read.format("delta").load(path)
        rows = df.count()
        cols = len(df.columns)
        if "Class" in df.columns:
            fraud = df.filter(F.col("Class") == 1).count()
            fraud_pct = fraud/rows*100
            print(f"✅ {name:<20} {rows:>12,} {fraud:>10,} {fraud_pct:>7.3f}%  {cols:>6}   HEALTHY")
        else:
            print(f"✅ {name:<20} {rows:>12,} {'N/A':>10} {'N/A':>8}  {cols:>6}   HEALTHY")
    except Exception as e:
        print(f"❌ {name:<20} ERROR: {str(e)[:40]}")

print("─" * 75)
print(f"""
🎯 FRAUDSENSE Medallion Architecture Complete!

  🥉 Bronze  → Raw data preserved (7,382,200 records)
  🥈 Silver  → Cleaned & typed  (7,381,119 records)  
  🥇 Gold    → Feature engineered & ML-ready

✅ NB04 Complete — Gold Layer is live!
🚀 Next → NB05: Deep EDA + Statistical Analysis
""")

# COMMAND ----------

