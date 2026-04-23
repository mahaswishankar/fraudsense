# Databricks notebook source
# ============================================================
# FRAUDSENSE — NB02: Data Ingestion → Bronze Layer
# Cell 1: Banner + Path Configuration
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║         FRAUDSENSE — NB02: Bronze Layer Ingestion           ║
║         Raw Data → Delta Tables (No Transformations)        ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable
import datetime

spark = SparkSession.builder.getOrCreate()

# ── Paths ────────────────────────────────────────────────────
VOLUME_PATH      = "/Volumes/workspace/default/fraud_data"
BRONZE_PATH      = "/Volumes/workspace/default/fraud_data/bronze"

CREDITCARD_RAW   = f"{VOLUME_PATH}/creditcard.csv"
PAYSIM_RAW       = f"{VOLUME_PATH}/PS_20174392719_1491204439457_log.csv"
IEEE_TRANS_RAW   = f"{VOLUME_PATH}/train_transaction.csv"
IEEE_IDENT_RAW   = f"{VOLUME_PATH}/train_identity.csv"

print("✅ Spark Session ready")
print(f"✅ Bronze target path: {BRONZE_PATH}")
print(f"✅ Ingestion started at: {datetime.datetime.now()}")
print("\n🥉 Beginning Bronze Layer ingestion...")

# COMMAND ----------

# ============================================================
# Cell 2: Ingest Kaggle Credit Card Fraud → Bronze Delta
# ============================================================

print("📥 Ingesting Kaggle Credit Card Fraud dataset...")

# Read raw CSV
cc_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(CREDITCARD_RAW)

# Add ingestion metadata columns
cc_df = cc_df \
    .withColumn("_source", F.lit("kaggle_creditcard")) \
    .withColumn("_ingested_at", F.current_timestamp()) \
    .withColumn("_ingestion_date", F.current_date())

# Write to Bronze Delta table
cc_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{BRONZE_PATH}/creditcard")

# Verify
count = spark.read.format("delta").load(f"{BRONZE_PATH}/creditcard").count()
print(f"✅ CreditCard Bronze table written!")
print(f"   Rows      : {count:,}")
print(f"   Columns   : {len(cc_df.columns)}")
print(f"   Location  : {BRONZE_PATH}/creditcard")

# COMMAND ----------

# ============================================================
# Cell 3: Ingest PaySim Mobile Money → Bronze Delta
# ============================================================

print("📥 Ingesting PaySim Mobile Money dataset...")

paysim_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(PAYSIM_RAW)

paysim_df = paysim_df \
    .withColumn("_source", F.lit("paysim_mobile")) \
    .withColumn("_ingested_at", F.current_timestamp()) \
    .withColumn("_ingestion_date", F.current_date())

paysim_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{BRONZE_PATH}/paysim")

count = spark.read.format("delta").load(f"{BRONZE_PATH}/paysim").count()
print(f"✅ PaySim Bronze table written!")
print(f"   Rows      : {count:,}")
print(f"   Columns   : {len(paysim_df.columns)}")
print(f"   Location  : {BRONZE_PATH}/paysim")

# COMMAND ----------

# ============================================================
# Cell 4: Ingest IEEE-CIS Transaction → Bronze Delta
# ============================================================

print("📥 Ingesting IEEE-CIS Transaction dataset...")

ieee_trans_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(IEEE_TRANS_RAW)

ieee_trans_df = ieee_trans_df \
    .withColumn("_source", F.lit("ieee_cis_transaction")) \
    .withColumn("_ingested_at", F.current_timestamp()) \
    .withColumn("_ingestion_date", F.current_date())

ieee_trans_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{BRONZE_PATH}/ieee_transaction")

count = spark.read.format("delta").load(f"{BRONZE_PATH}/ieee_transaction").count()
print(f"✅ IEEE-CIS Transaction Bronze table written!")
print(f"   Rows      : {count:,}")
print(f"   Columns   : {len(ieee_trans_df.columns)}")
print(f"   Location  : {BRONZE_PATH}/ieee_transaction")

# COMMAND ----------

# ============================================================
# Cell 5: Ingest IEEE-CIS Identity → Bronze Delta
# ============================================================

print("📥 Ingesting IEEE-CIS Identity dataset...")

ieee_ident_df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(IEEE_IDENT_RAW)

ieee_ident_df = ieee_ident_df \
    .withColumn("_source", F.lit("ieee_cis_identity")) \
    .withColumn("_ingested_at", F.current_timestamp()) \
    .withColumn("_ingestion_date", F.current_date())

ieee_ident_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{BRONZE_PATH}/ieee_identity")

count = spark.read.format("delta").load(f"{BRONZE_PATH}/ieee_identity").count()
print(f"✅ IEEE-CIS Identity Bronze table written!")
print(f"   Rows      : {count:,}")
print(f"   Columns   : {len(ieee_ident_df.columns)}")
print(f"   Location  : {BRONZE_PATH}/ieee_identity")

# COMMAND ----------

# ============================================================
# Cell 6: Bronze Layer Health Check & Summary
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║              🥉 BRONZE LAYER — HEALTH CHECK                 ║
╚══════════════════════════════════════════════════════════════╝
""")

bronze_tables = {
    "CreditCard"        : f"{BRONZE_PATH}/creditcard",
    "PaySim"            : f"{BRONZE_PATH}/paysim",
    "IEEE Transaction"  : f"{BRONZE_PATH}/ieee_transaction",
    "IEEE Identity"     : f"{BRONZE_PATH}/ieee_identity",
}

total_rows = 0
print(f"{'Table':<25} {'Rows':>12} {'Columns':>10} {'Status'}")
print("─" * 60)

for name, path in bronze_tables.items():
    try:
        df = spark.read.format("delta").load(path)
        rows = df.count()
        cols = len(df.columns)
        total_rows += rows
        print(f"✅ {name:<23} {rows:>12,} {cols:>10}   HEALTHY")
    except Exception as e:
        print(f"❌ {name:<23} {'ERROR':>12}            {str(e)[:20]}")

print("─" * 60)
print(f"{'TOTAL':.<25} {total_rows:>12,}")
print(f"\n🎯 Bronze Layer fully loaded with {total_rows:,} raw records!")
print(f"✅ NB02 Complete — Ready for Silver transformation!")
print(f"🚀 Next → NB03: Bronze → Silver (Cleaning & Typing)")

# COMMAND ----------

