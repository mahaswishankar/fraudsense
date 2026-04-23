# Databricks notebook source
# ============================================================
# Cell 0: Install Dependencies
# ============================================================
import subprocess
packages = ["torch", "scikit-learn", "imbalanced-learn", "mlflow"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"✅ {pkg}" if result.returncode == 0 else f"❌ {pkg}: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE — NB15: LSTM Sequential Patterns
# Cell 1: Banner + Setup
# ============================================================
print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE — NB15: LSTM Sequential Patterns     ║
║     Phase 4 — Deep Learning Round 1 (PyTorch)       ║
╚══════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from imblearn.over_sampling import SMOTE

# Dark theme
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
LSTM_COLOR   = '#a78bfa'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"✅ PyTorch version : {torch.__version__}")
print(f"✅ Device          : {DEVICE}")
print("✅ All libraries loaded")
print("\n🧠 LSTM round starting...")

# COMMAND ----------

# ============================================================
# Cell 2: Load Gold Data + SMOTE
# ============================================================
print("📂 Loading CreditCard Gold dataset...")

GOLD_PATH = "/Volumes/workspace/default/fraud_data/gold/creditcard"
cc = spark.read.format("delta").load(GOLD_PATH).toPandas()

print(f"✅ Loaded  : {len(cc):,} rows")
print(f"✅ Columns : {cc.shape[1]}")
print(f"✅ Fraud % : {cc['Class'].mean()*100:.4f}%")

# Same features as NB13
v_cols    = [c for c in cc.columns if c.startswith('V')]
eng_feats = ['amount_log', 'amount_zscore', 'amount_spike',
             'is_night', 'tx_velocity_10', 'high_amount_flag',
             'v1_v2_interaction', 'v3_v4_interaction', 'v14_v17_interaction',
             'v_sum_top5', 'v_abs_sum']
eng_feats    = [f for f in eng_feats if f in cc.columns]
feature_cols = v_cols + eng_feats

X = cc[feature_cols].fillna(0).values
y = cc['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("\n⚖️  Applying SMOTE...")
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)

# Reshape for LSTM: (samples, timesteps=1, features)
X_train_lstm = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_test_lstm  = X_test_scaled.reshape(-1,  1, X_test_scaled.shape[1])

y_train_sm = y_train_sm.astype(np.float32)
y_test     = y_test.astype(np.float32)

# PyTorch DataLoaders
train_ds = TensorDataset(torch.tensor(X_train_lstm), torch.tensor(y_train_sm))
test_ds  = TensorDataset(torch.tensor(X_test_lstm),  torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

print(f"✅ Train size  : {len(X_train_lstm):,}")
print(f"✅ Test size   : {len(X_test_lstm):,}")
print(f"✅ Features    : {X_train_scaled.shape[1]}")
print("\n✅ Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: Build BiLSTM Model in PyTorch
# ============================================================
print("🏗️  Building Bidirectional LSTM...")
print("   Architecture: BiLSTM(128) → BiLSTM(64) → Dense(32) → 1\n")

class BiLSTMFraud(nn.Module):
    def __init__(self, input_size, hidden1=128, hidden2=64):
        super(BiLSTMFraud, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True,
                             bidirectional=True, dropout=0.3)
        self.bn1   = nn.BatchNorm1d(hidden1 * 2)
        self.drop1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(hidden1 * 2, hidden2, batch_first=True,
                             bidirectional=True, dropout=0.3)
        self.bn2   = nn.BatchNorm1d(hidden2 * 2)
        self.drop2 = nn.Dropout(0.3)

        self.fc1   = nn.Linear(hidden2 * 2, 32)
        self.relu1 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)

        self.fc2   = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.drop4 = nn.Dropout(0.2)

        self.out   = nn.Linear(16, 1)
        self.sig   = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.bn1(x[:, -1, :])
        x = self.drop1(x)

        x = x.unsqueeze(1)
        x, _ = self.lstm2(x)
        x = self.bn2(x[:, -1, :])
        x = self.drop2(x)

        x = self.drop3(self.relu1(self.fc1(x)))
        x = self.drop4(self.relu2(self.fc2(x)))
        return self.sig(self.out(x)).squeeze(1)

N_FEATURES = X_train_scaled.shape[1]
model = BiLSTMFraud(input_size=N_FEATURES).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"\n✅ Total parameters : {total_params:,}")
print(f"✅ Device           : {DEVICE}")
print("✅ BiLSTM model built!")

# COMMAND ----------

# ============================================================
# Cell 4: Train BiLSTM + MLflow Tracking
# ============================================================
print("🧠 Training BiLSTM — this may take a few minutes...\n")

# Class weights to handle imbalance
neg = np.sum(y_train_sm == 0)
pos = np.sum(y_train_sm == 1)
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE)
print(f"✅ Pos weight (fraud) : {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3)

EPOCHS     = 30
best_auc   = 0.0
patience   = 5
no_improve = 0

train_losses = []
val_aucs     = []

with mlflow.start_run(run_name="BiLSTM_CreditCard"):

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # --- Validate ---
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                probs = model(xb).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(yb.numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        val_aucs.append(val_auc)
        scheduler.step(val_auc)

        print(f"  Epoch {epoch+1:>2}/{EPOCHS} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        # Early stopping
        if val_auc > best_auc:
            best_auc   = val_auc
            no_improve = 0
            torch.save(model.state_dict(), "/tmp/best_lstm.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch+1}")
                break

    # Load best weights
    model.load_state_dict(torch.load("/tmp/best_lstm.pt"))

    # Final evaluation
    model.eval()
    lstm_probs = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            lstm_probs.extend(model(xb).cpu().numpy())

    lstm_probs = np.array(lstm_probs)
    lstm_preds = (lstm_probs >= 0.5).astype(int)
    lstm_auc   = roc_auc_score(y_test, lstm_probs)
    lstm_ap    = average_precision_score(y_test, lstm_probs)

    # Log to MLflow
    mlflow.log_param("model",        "BiLSTM_PyTorch")
    mlflow.log_param("architecture", "128-64-32-16-1")
    mlflow.log_param("n_features",   N_FEATURES)
    mlflow.log_param("batch_size",   512)
    mlflow.log_param("optimizer",    "Adam")
    mlflow.log_metric("auc_roc",     lstm_auc)
    mlflow.log_metric("avg_precision", lstm_ap)
    mlflow.log_metric("best_epoch",  EPOCHS - no_improve)
    mlflow.set_tag("project",        "FRAUDSENSE")
    mlflow.set_tag("notebook",       "NB15")

print(f"\n✅ LSTM trained!")
print(f"✅ Best AUC-ROC    : {best_auc:.4f}")
print(f"✅ Final AUC-ROC   : {lstm_auc:.4f}")
print(f"✅ Avg Precision   : {lstm_ap:.4f}")

# COMMAND ----------

# ============================================================
# Cell 5: Training Curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, color=LSTM_COLOR,  lw=2, label='Train Loss')
axes[0].set_title('BiLSTM — Training Loss', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
axes[0].grid(alpha=0.3)

axes[1].plot(val_aucs, color=LEGIT_COLOR, lw=2, label='Val AUC-ROC')
axes[1].axhline(y=best_auc, color=FRAUD_COLOR, lw=1.5, ls='--',
                label=f'Best: {best_auc:.4f}')
axes[1].set_title('BiLSTM — Validation AUC', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC-ROC')
axes[1].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
axes[1].grid(alpha=0.3)

plt.suptitle('FRAUDSENSE NB15 — BiLSTM Training History',
             fontsize=13, fontweight='bold', color='#e6edf3', y=1.02)
plt.tight_layout()

chart_path = "/Volumes/workspace/default/fraud_data/models/nb15_lstm_training.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"✅ Chart saved → {chart_path}")

# COMMAND ----------

# ============================================================
# Cell 6: Classification Report + Results Summary
# ============================================================
print("📊 Classification Report:")
print("-" * 50)
print(classification_report(y_test, lstm_preds, target_names=['Legit', 'Fraud']))

# NB13 results for reference
NB13_RESULTS = {
    "MLP (256-128-64)": {"auc": 0.9337, "ap": 0.7866},
    "TabNet":           {"auc": 0.9692, "ap": 0.5347},
}

print("\n╔══════════════════════════════════════════════╗")
print("║          NB15 — RESULTS SUMMARY              ║")
print("╠══════════════════════════════════════════════╣")
print(f"║  {'Model':<22} {'AUC-ROC':<10} {'Avg Precision'}  ║")
print("╠══════════════════════════════════════════════╣")
for m, v in NB13_RESULTS.items():
    print(f"║  {m:<22} {v['auc']:.4f}     {v['ap']:.4f}         ║")
print(f"║  {'BiLSTM':<22} {lstm_auc:.4f}     {lstm_ap:.4f}  ← NEW   ║")
print("╚══════════════════════════════════════════════╝")

# COMMAND ----------

# ============================================================
# Cell 7: Updated Full Leaderboard
# ============================================================

ALL_MODELS_UPDATED = [
    ("Extra Trees",          0.9786, "NB10", "Ensemble"),
    ("XGBoost",              0.9765, "NB12", "Boosting"),
    ("GradientBoosting",     0.9764, "NB11", "Boosting"),
    ("LightGBM",             0.9724, "NB12", "Boosting"),
    ("TabNet",               0.9692, "NB13", "Neural Net"),
    ("HistGradientBoosting", 0.9711, "NB11", "Boosting"),
    ("AdaBoost",             0.9710, "NB11", "Boosting"),
    ("CatBoost",             0.9681, "NB12", "Boosting"),
    ("Random Forest",        0.9678, "NB10", "Ensemble"),
    ("Logistic Regression",  0.9632, "NB08", "Linear"),
    ("Linear SVM",           0.9570, "NB08", "Linear"),
    ("Isolation Forest CC",  0.9515, "NB06", "Anomaly"),
    ("Gaussian Naive Bayes", 0.9500, "NB09", "Probabilistic"),
    ("Isolation Forest PS",  0.9282, "NB06", "Anomaly"),
    ("BiLSTM",               lstm_auc, "NB15", "Deep Learning"),
    ("MLP",                  0.9337, "NB13", "Neural Net"),
    ("KNN",                  0.8880, "NB09", "Instance"),
]

ALL_MODELS_UPDATED = sorted(ALL_MODELS_UPDATED, key=lambda x: x[1], reverse=True)

print("🏆 FULL BATTLE ROYALE LEADERBOARD (17 Models):")
medals = {1: "🥇", 2: "🥈", 3: "🥉"}
for i, (name, auc, nb, cat) in enumerate(ALL_MODELS_UPDATED, 1):
    medal = medals.get(i, "  ")
    tag   = "← NEW" if nb == "NB15" else ""
    print(f"   {i:<3} {medal} {name:<26} : {auc:.4f}  {tag}")

print(f"""
╔══════════════════════════════════════════════════════╗
║  FRAUDSENSE FORMULA PROGRESS                        ║
║                                                      ║
║  0.35 × Stacking    → NB19-21 (pending)             ║
║  0.25 × LSTM        → {lstm_auc:.4f} ✅ NB15 done          ║
║  0.20 × XGBoost     → 0.9765  ✅ NB12 done          ║
║  0.20 × IsoForest   → 0.9515  ✅ NB06 done          ║
║                                                      ║
║  Next  → NB16: CNN 1D (Local pattern detection)     ║
╚══════════════════════════════════════════════════════╝
""")

print("✅ NB15 complete!")
print("🚀 Next: NB16 — CNN 1D")

# COMMAND ----------

