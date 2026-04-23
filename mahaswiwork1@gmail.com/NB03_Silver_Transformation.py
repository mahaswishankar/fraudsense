# Databricks notebook source
# ============================================================
# FRAUDSENSE — NB03: Bronze → Silver Transformation
# Cell 1: Banner + Path Configuration
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║       FRAUDSENSE — NB03: Silver Layer Transformation        ║
║       Bronze (Raw) → Silver (Cleaned, Typed, Deduped)       ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import datetime

spark = SparkSession.builder.getOrCreate()

BRONZE_PATH = "/Volumes/workspace/default/fraud_data/bronze"
SILVER_PATH = "/Volumes/workspace/default/fraud_data/silver"

print(f"✅ Spark ready")
print(f"✅ Bronze source : {BRONZE_PATH}")
print(f"✅ Silver target : {SILVER_PATH}")
print(f"✅ Started at    : {datetime.datetime.now()}")
print("\n🥈 Beginning Silver transformation...")

# COMMAND ----------

# ============================================================
# Cell 2: CreditCard Bronze → Silver
# ============================================================

print("🔄 Processing CreditCard dataset...")

cc_bronze = spark.read.format("delta").load(f"{BRONZE_PATH}/creditcard")

# Drop duplicates
before = cc_bronze.count()
cc_silver = cc_bronze.dropDuplicates()
after = cc_silver.count()
dupes_removed = before - after

# Drop rows where critical columns are null
cc_silver = cc_silver.dropna(subset=["Time", "Amount", "Class"])

# Ensure correct types
cc_silver = cc_silver \
    .withColumn("Time",   F.col("Time").cast(DoubleType())) \
    .withColumn("Amount", F.col("Amount").cast(DoubleType())) \
    .withColumn("Class",  F.col("Class").cast(IntegerType()))

# Add silver metadata
cc_silver = cc_silver \
    .withColumn("_silver_processed_at", F.current_timestamp()) \
    .withColumn("_is_fraud", F.col("Class") == 1)

# Write to silver
cc_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{SILVER_PATH}/creditcard")

fraud_count = cc_silver.filter(F.col("Class") == 1).count()
print(f"✅ CreditCard Silver done!")
print(f"   Rows before dedup : {before:,}")
print(f"   Duplicates removed: {dupes_removed:,}")
print(f"   Final rows        : {after:,}")
print(f"   Fraud cases       : {fraud_count:,} ({fraud_count/after*100:.3f}%)")

# COMMAND ----------

# ============================================================
# Cell 3: PaySim Bronze → Silver
# ============================================================

print("🔄 Processing PaySim dataset...")

paysim_bronze = spark.read.format("delta").load(f"{BRONZE_PATH}/paysim")

before = paysim_bronze.count()
paysim_silver = paysim_bronze.dropDuplicates()
after = paysim_silver.count()
dupes_removed = before - after

# Drop nulls in critical columns
paysim_silver = paysim_silver.dropna(subset=["type", "amount", "isFraud"])

# Cast correct types
paysim_silver = paysim_silver \
    .withColumn("step",             F.col("step").cast(IntegerType())) \
    .withColumn("amount",           F.col("amount").cast(DoubleType())) \
    .withColumn("oldbalanceOrg",    F.col("oldbalanceOrg").cast(DoubleType())) \
    .withColumn("newbalanceOrig",   F.col("newbalanceOrig").cast(DoubleType())) \
    .withColumn("oldbalanceDest",   F.col("oldbalanceDest").cast(DoubleType())) \
    .withColumn("newbalanceDest",   F.col("newbalanceDest").cast(DoubleType())) \
    .withColumn("isFraud",          F.col("isFraud").cast(IntegerType())) \
    .withColumn("isFlaggedFraud",   F.col("isFlaggedFraud").cast(IntegerType()))

# Standardize fraud column name
paysim_silver = paysim_silver \
    .withColumnRenamed("isFraud", "Class") \
    .withColumn("_silver_processed_at", F.current_timestamp()) \
    .withColumn("_is_fraud", F.col("Class") == 1)

paysim_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{SILVER_PATH}/paysim")

fraud_count = paysim_silver.filter(F.col("Class") == 1).count()
print(f"✅ PaySim Silver done!")
print(f"   Rows before dedup : {before:,}")
print(f"   Duplicates removed: {dupes_removed:,}")
print(f"   Final rows        : {after:,}")
print(f"   Fraud cases       : {fraud_count:,} ({fraud_count/after*100:.3f}%)")

# COMMAND ----------

# ============================================================
# Cell 4: IEEE-CIS Transaction Bronze → Silver
# ============================================================

print("🔄 Processing IEEE-CIS Transaction dataset...")

ieee_trans_bronze = spark.read.format("delta").load(f"{BRONZE_PATH}/ieee_transaction")

before = ieee_trans_bronze.count()
ieee_trans_silver = ieee_trans_bronze.dropDuplicates()
after = ieee_trans_silver.count()
dupes_removed = before - after

# Drop nulls in critical columns only
ieee_trans_silver = ieee_trans_silver.dropna(subset=["TransactionID", "TransactionAmt", "isFraud"])

# Cast critical columns
ieee_trans_silver = ieee_trans_silver \
    .withColumn("TransactionID",  F.col("TransactionID").cast(LongType())) \
    .withColumn("TransactionDT",  F.col("TransactionDT").cast(LongType())) \
    .withColumn("TransactionAmt", F.col("TransactionAmt").cast(DoubleType())) \
    .withColumn("isFraud",        F.col("isFraud").cast(IntegerType()))

# Standardize fraud column
ieee_trans_silver = ieee_trans_silver \
    .withColumnRenamed("isFraud", "Class") \
    .withColumn("_silver_processed_at", F.current_timestamp()) \
    .withColumn("_is_fraud", F.col("Class") == 1)

# Fill nulls in V-columns (PCA features) with 0
v_cols = [c for c in ieee_trans_silver.columns if c.startswith("V")]
for vc in v_cols:
    ieee_trans_silver = ieee_trans_silver.withColumn(vc, F.coalesce(F.col(vc), F.lit(0.0)))

ieee_trans_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{SILVER_PATH}/ieee_transaction")

fraud_count = ieee_trans_silver.filter(F.col("Class") == 1).count()
print(f"✅ IEEE Transaction Silver done!")
print(f"   Rows before dedup : {before:,}")
print(f"   Duplicates removed: {dupes_removed:,}")
print(f"   Final rows        : {after:,}")
print(f"   Fraud cases       : {fraud_count:,} ({fraud_count/after*100:.3f}%)")

# COMMAND ----------

# ============================================================
# Cell 5: IEEE-CIS Identity Bronze → Silver
# ============================================================

print("🔄 Processing IEEE-CIS Identity dataset...")

ieee_ident_bronze = spark.read.format("delta").load(f"{BRONZE_PATH}/ieee_identity")

before = ieee_ident_bronze.count()
ieee_ident_silver = ieee_ident_bronze.dropDuplicates()
after = ieee_ident_silver.count()
dupes_removed = before - after

# Drop nulls only on TransactionID (join key)
ieee_ident_silver = ieee_ident_silver.dropna(subset=["TransactionID"])

# Cast TransactionID
ieee_ident_silver = ieee_ident_silver \
    .withColumn("TransactionID", F.col("TransactionID").cast(LongType())) \
    .withColumn("_silver_processed_at", F.current_timestamp())

# Fill remaining nulls with "UNKNOWN" for string cols, 0 for numeric
string_cols = [f.name for f in ieee_ident_silver.schema.fields if str(f.dataType) == "StringType()"]
numeric_cols = [f.name for f in ieee_ident_silver.schema.fields 
                if str(f.dataType) in ["DoubleType()", "IntegerType()", "LongType()"]
                and f.name != "TransactionID"]

for col in string_cols:
    ieee_ident_silver = ieee_ident_silver.withColumn(col, F.coalesce(F.col(col), F.lit("UNKNOWN")))
for col in numeric_cols:
    ieee_ident_silver = ieee_ident_silver.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))

ieee_ident_silver.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{SILVER_PATH}/ieee_identity")

print(f"✅ IEEE Identity Silver done!")
print(f"   Rows before dedup : {before:,}")
print(f"   Duplicates removed: {dupes_removed:,}")
print(f"   Final rows        : {after:,}")
print(f"   Columns           : {len(ieee_ident_silver.columns)}")

# COMMAND ----------

# ============================================================
# Cell 6: Silver Layer Health Check & Summary
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║              🥈 SILVER LAYER — HEALTH CHECK                 ║
╚══════════════════════════════════════════════════════════════╝
""")

silver_tables = {
    "CreditCard"       : f"{SILVER_PATH}/creditcard",
    "PaySim"           : f"{SILVER_PATH}/paysim",
    "IEEE Transaction" : f"{SILVER_PATH}/ieee_transaction",
    "IEEE Identity"    : f"{SILVER_PATH}/ieee_identity",
}

total_rows = 0
total_fraud = 0

print(f"{'Table':<22} {'Rows':>12} {'Fraud':>10} {'Fraud%':>8}  Status")
print("─" * 70)

for name, path in silver_tables.items():
    try:
        df = spark.read.format("delta").load(path)
        rows = df.count()
        total_rows += rows
        if "Class" in df.columns:
            fraud = df.filter(F.col("Class") == 1).count()
            total_fraud += fraud
            fraud_pct = fraud / rows * 100
            print(f"✅ {name:<20} {rows:>12,} {fraud:>10,} {fraud_pct:>7.3f}%  HEALTHY")
        else:
            print(f"✅ {name:<20} {rows:>12,} {'N/A':>10} {'N/A':>8}  HEALTHY")
    except Exception as e:
        print(f"❌ {name:<20} ERROR: {str(e)[:30]}")

print("─" * 70)
print(f"{'TOTAL':<22} {total_rows:>12,} {total_fraud:>10,}")
print(f"\n🎯 Silver Layer complete with {total_rows:,} clean records!")
print(f"⚠️  Total fraud cases across datasets: {total_fraud:,}")
print(f"✅ NB03 Complete — Data is clean, typed, and deduped!")
print(f"🚀 Next → NB04: Silver → Gold (Feature Engineering)")

# COMMAND ----------

