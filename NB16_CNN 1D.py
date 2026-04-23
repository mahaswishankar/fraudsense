# Databricks notebook source
# ============================================================
# Cell 0: Install Dependencies
# ============================================================
import subprocess
packages = ["torch", "scikit-learn", "imbalanced-learn", "mlflow"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"OK {pkg}" if result.returncode == 0 else f"FAILED {pkg}: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE - NB16: CNN 1D Local Pattern Detection
# Cell 1: Banner + Setup
# ============================================================
print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB16: CNN 1D                       ║
║     Phase 4 - Deep Learning Round 2                 ║
╚══════════════════════════════════════════════════════╝
""")

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
CNN_COLOR    = '#38bdf8'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {DEVICE}")
print("All libraries loaded")
print("CNN 1D round starting...")

# COMMAND ----------

# ============================================================
# Cell 2: Load Gold Data + SMOTE
# ============================================================
GOLD_PATH = "/Volumes/workspace/default/fraud_data/gold/creditcard"
cc = spark.read.format("delta").load(GOLD_PATH).toPandas()

v_cols       = [c for c in cc.columns if c.startswith('V')]
eng_feats    = ['amount_log', 'amount_zscore', 'amount_spike',
                'is_night', 'tx_velocity_10', 'high_amount_flag',
                'v1_v2_interaction', 'v3_v4_interaction', 'v14_v17_interaction',
                'v_sum_top5', 'v_abs_sum']
eng_feats    = [f for f in eng_feats if f in cc.columns]
feature_cols = v_cols + eng_feats

X = cc[feature_cols].fillna(0).values
y = cc['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm).astype(np.float32)
X_test_scaled  = scaler.transform(X_test).astype(np.float32)

# CNN expects (batch, channels, length) - treat each feature as a signal channel
X_train_cnn = X_train_scaled.reshape(-1, 1, X_train_scaled.shape[1])
X_test_cnn  = X_test_scaled.reshape(-1,  1, X_test_scaled.shape[1])

y_train_sm = y_train_sm.astype(np.float32)
y_test     = y_test.astype(np.float32)

train_ds = TensorDataset(torch.tensor(X_train_cnn), torch.tensor(y_train_sm))
test_ds  = TensorDataset(torch.tensor(X_test_cnn),  torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

print(f"Train shape : {X_train_cnn.shape}")
print(f"Test shape  : {X_test_cnn.shape}")
print(f"Features    : {X_train_scaled.shape[1]}")
print("Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: Build CNN 1D Model
# ============================================================
print("Building CNN 1D architecture...")
print("Architecture: Conv(64) -> Conv(128) -> Conv(256) -> GlobalAvgPool -> Dense(64) -> 1")

class CNN1DFraud(nn.Module):
    def __init__(self, input_length):
        super(CNN1DFraud, self).__init__()

        # Block 1
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

        # Block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)

        # Block 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)

        # Global average pooling collapses the spatial dim
        self.gap   = nn.AdaptiveAvgPool1d(1)

        # Dense head
        self.fc1   = nn.Linear(256, 64)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.3)
        self.out   = nn.Linear(64, 1)
        self.sig   = nn.Sigmoid()

    def forward(self, x):
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.drop3(self.relu3(self.bn3(self.conv3(x))))
        x = self.gap(x).squeeze(-1)
        x = self.drop4(self.relu4(self.fc1(x)))
        return self.sig(self.out(x)).squeeze(1)

N_FEATURES = X_train_scaled.shape[1]
model = CNN1DFraud(input_length=N_FEATURES).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"\nTotal parameters : {total_params:,}")
print("CNN 1D model built!")

# COMMAND ----------

# ============================================================
# Cell 4: Train CNN 1D + MLflow Tracking
# ============================================================
print("Training CNN 1D...\n")

neg = np.sum(y_train_sm == 0)
pos = np.sum(y_train_sm == 1)
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE)
print(f"Pos weight (fraud) : {pos_weight.item():.2f}")

criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3)

EPOCHS     = 30
best_auc   = 0.0
patience   = 5
no_improve = 0

train_losses = []
val_aucs     = []

with mlflow.start_run(run_name="CNN1D_CreditCard"):

    for epoch in range(EPOCHS):
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

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                probs = model(xb.to(DEVICE)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(yb.numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        val_aucs.append(val_auc)
        scheduler.step(val_auc)

        print(f"Epoch {epoch+1:>2}/{EPOCHS} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc   = val_auc
            no_improve = 0
            torch.save(model.state_dict(), "/tmp/best_cnn1d.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load("/tmp/best_cnn1d.pt"))

    model.eval()
    cnn_probs = []
    with torch.no_grad():
        for xb, yb in test_loader:
            cnn_probs.extend(model(xb.to(DEVICE)).cpu().numpy())

    cnn_probs = np.array(cnn_probs)
    cnn_preds = (cnn_probs >= 0.5).astype(int)
    cnn_auc   = roc_auc_score(y_test, cnn_probs)
    cnn_ap    = average_precision_score(y_test, cnn_probs)

    mlflow.log_param("model",          "CNN1D_PyTorch")
    mlflow.log_param("architecture",   "64-128-256-GAP-64-1")
    mlflow.log_param("n_features",     N_FEATURES)
    mlflow.log_param("batch_size",     512)
    mlflow.log_param("optimizer",      "Adam")
    mlflow.log_metric("auc_roc",       cnn_auc)
    mlflow.log_metric("avg_precision", cnn_ap)
    mlflow.log_metric("best_epoch",    EPOCHS - no_improve)
    mlflow.set_tag("project",          "FRAUDSENSE")
    mlflow.set_tag("notebook",         "NB16")

print(f"Best AUC-ROC  : {best_auc:.4f}")
print(f"Final AUC-ROC : {cnn_auc:.4f}")
print(f"Avg Precision : {cnn_ap:.4f}")

# COMMAND ----------

# ============================================================
# Cell 5: Training Curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, color=CNN_COLOR, lw=2, label='Train Loss')
axes[0].set_title('CNN 1D - Training Loss', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
axes[0].grid(alpha=0.3)

axes[1].plot(val_aucs, color=LEGIT_COLOR, lw=2, label='Val AUC-ROC')
axes[1].axhline(y=best_auc, color=FRAUD_COLOR, lw=1.5, ls='--',
                label=f'Best: {best_auc:.4f}')
axes[1].set_title('CNN 1D - Validation AUC', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC-ROC')
axes[1].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
axes[1].grid(alpha=0.3)

plt.suptitle('FRAUDSENSE NB16 - CNN 1D Training History',
             fontsize=13, fontweight='bold', color='#e6edf3', y=1.02)
plt.tight_layout()

chart_path = "/Volumes/workspace/default/fraud_data/models/nb16_cnn1d_training.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart_path}")

# COMMAND ----------

# ============================================================
# Cell 6: Results Summary
# ============================================================
print(classification_report(y_test, cnn_preds, target_names=['Legit', 'Fraud']))

print("╔══════════════════════════════════════════════════╗")
print("║            NB16 - RESULTS SUMMARY               ║")
print("╠══════════════════════════════════════════════════╣")
print(f"║  {'Model':<24} {'AUC-ROC':<10} {'Avg Precision'} ║")
print("╠══════════════════════════════════════════════════╣")
print(f"║  {'MLP (256-128-64)':<24} {0.9337:.4f}     {0.7866:.4f}       ║")
print(f"║  {'TabNet':<24} {0.9692:.4f}     {0.5347:.4f}       ║")
print(f"║  {'BiLSTM':<24} {0.9259:.4f}     {'N/A':<10}       ║")
print(f"║  {'CNN 1D':<24} {cnn_auc:.4f}     {cnn_ap:.4f}  <- NEW ║")
print("╚══════════════════════════════════════════════════╝")

print("NB16 complete!")
print("Next: NB17 - GRU/RNN")

# COMMAND ----------

