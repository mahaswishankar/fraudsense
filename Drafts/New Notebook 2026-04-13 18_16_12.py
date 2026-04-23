# Databricks notebook source
# Run this to confirm all export files exist
import os
EXPORT_PATH = "/Volumes/workspace/default/fraud_data/exports"
for f in os.listdir(EXPORT_PATH):
    size = os.path.getsize(f"{EXPORT_PATH}/{f}") / 1024
    print(f"{f:45s} : {size:.1f} KB")

# COMMAND ----------

