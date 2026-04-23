# FRAUDSENSE 🔍
### Real-Time Financial Fraud Intelligence Platform

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)

> A production-architecture fraud detection system processing **7.3M+ financial records** across 4 real-world datasets, built on Databricks with full Medallion Architecture, a 15-model ML battle royale, and a custom 4-model ensemble achieving **0.982 AUC-ROC**.

---

## 🏗️ Architecture
RAW DATA (7.3M records)
↓
🥉 BRONZE LAYER — Raw ingestion, schema enforcement
↓
🥈 SILVER LAYER — Cleaning, deduplication, transformation
↓
🥇 GOLD LAYER — Feature engineering, ML-ready datasets
↓
🤖 MODEL BATTLE ROYALE (15 models benchmarked)
↓
🏆 CUSTOM ENSEMBLE (4-model, weighted)
↓
📊 POWER BI DASHBOARD + PDF AUTO-REPORT

---

## 📊 Results

| Metric | Score |
|--------|-------|
| AUC-ROC | **0.982** |
| Architecture | Medallion (Bronze/Silver/Gold) |
| Records Processed | 7.3M+ |
| Models Benchmarked | 15 |
| Ensemble Components | 4 |

---

## 🤖 Models Benchmarked

| Category | Models |
|----------|--------|
| Classical ML | Logistic Regression, SVM, Naive Bayes, KNN |
| Tree-Based | Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost |
| Deep Learning | LSTM, CNN1D, GRU-RNN, TabTransformer, TabNet |
| Anomaly Detection | Isolation Forest |
| Stacking | Layer 1 + Layer 2 Meta-Learner |

---

## 🏆 Final Ensemble
Final Score = 0.35 × Stacking
+ 0.25 × CNN1D
+ 0.20 × XGBoost
+ 0.20 × Isolation Forest

---

## 📁 Notebook Structure

| Notebook | Description |
|----------|-------------|
| NB02 | Bronze Layer — Raw Ingestion |
| NB03 | Silver Layer — Transformation |
| NB04 | Gold Layer — Feature Engineering |
| NB05 | EDA & Statistical Analysis |
| NB06 | Isolation Forest Anomaly Detection |
| NB07 | Graph-Based Fraud Detection |
| NB08–NB13 | Individual Model Training (6 notebooks) |
| NB14 | MLflow Master Leaderboard |
| NB15–NB18 | Deep Learning Models (LSTM, CNN1D, GRU, TabTransformer) |
| NB19–NB20 | Layer 1 & Layer 2 Stacking |
| NB21 | Final 3-Layer Ensemble |
| NB22 | FRAUDSENSE vs XGBoost Benchmark |
| NB23 | SHAP Explainability |
| NB24 | LIME Explainability |
| NB25 | Data Drift Detection |
| NB26 | Auto PDF Report Generator |
| NB27 | Spark Structured Streaming |
| NB28 | Gold Export + Dashboard Prep |
| NB29 | Power BI Pipeline |

---

## 🛠️ Tech Stack

- **Platform:** Databricks Community Edition
- **Big Data:** Apache Spark (PySpark)
- **ML:** XGBoost, LightGBM, CatBoost, scikit-learn
- **Deep Learning:** TensorFlow/Keras (LSTM, CNN1D, GRU, TabTransformer)
- **Explainability:** SHAP, LIME
- **Experiment Tracking:** MLflow
- **Visualization:** Power BI, Matplotlib, Seaborn
- **Streaming:** Spark Structured Streaming

---

## 👤 Author

**Mahaswi Shankar**  
B.Tech CSE (Data Science) — Bennett University  
📧 mahaswiwork1@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/mahaswishankar1) | [GitHub](https://github.com/mahaswishankar)
