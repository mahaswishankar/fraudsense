# Databricks notebook source
# ============================================================
# Cell 0: Install Dependencies
# ============================================================
import subprocess
packages = ["torch", "scikit-learn", "imbalanced-learn", "mlflow"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"{pkg} installed" if result.returncode == 0 else f"{pkg} failed: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE - NB18: TabTransformer
# Cell 1: Banner + Setup
# ============================================================
print("""
╔══════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB18: TabTransformer               ║
║     Phase 4 - Deep Learning Round 4 (Final)         ║
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
TRANS_COLOR  = '#818cf8'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"PyTorch  : {torch.__version__}")
print(f"Device   : {DEVICE}")
print("All libraries loaded")
print("TabTransformer round starting...")

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

y_train_sm = y_train_sm.astype(np.float32)
y_test     = y_test.astype(np.float32)

# TabTransformer treats each feature as a token — shape: (batch, n_features)
# No reshape needed, standard 2D input
X_train_t = torch.tensor(X_train_scaled)
X_test_t  = torch.tensor(X_test_scaled)
y_train_t = torch.tensor(y_train_sm)
y_test_t  = torch.tensor(y_test)

train_ds = TensorDataset(X_train_t, y_train_t)
test_ds  = TensorDataset(X_test_t,  y_test_t)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

N_FEATURES = X_train_scaled.shape[1]

print(f"Train shape : {X_train_scaled.shape}")
print(f"Test shape  : {X_test_scaled.shape}")
print(f"Features    : {N_FEATURES}")
print("Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: Build TabTransformer Model
# ============================================================
print("Building TabTransformer architecture...")
print("Architecture: FeatureEmbed -> TransformerEncoder(4 heads x 3 layers) -> MLP -> 1")

class TabTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout=0.1):
        super(TabTransformer, self).__init__()

        # Project each scalar feature into d_model dimensions
        # Each feature becomes a d_model-dim embedding (its own "token")
        self.feature_embed = nn.Linear(1, d_model)

        # Learnable positional encoding for each feature position
        self.pos_embed = nn.Parameter(torch.randn(1, n_features, d_model))

        # Transformer encoder — multi-head self-attention across features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head on top of flattened transformer output
        self.mlp = nn.Sequential(
            nn.Linear(n_features * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, n_features)
        # Embed each feature scalar independently
        x = x.unsqueeze(-1)                        # (batch, n_features, 1)
        x = self.feature_embed(x)                  # (batch, n_features, d_model)
        x = x + self.pos_embed                     # add positional encoding
        x = self.transformer(x)                    # self-attention across features
        x = x.flatten(1)                           # (batch, n_features * d_model)
        return self.mlp(x).squeeze(1)

model = TabTransformer(
    n_features=N_FEATURES,
    d_model=64,
    nhead=4,
    num_layers=3,
    dim_feedforward=256,
    dropout=0.1
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"\nTotal parameters : {total_params:,}")
print("TabTransformer model built!")

# COMMAND ----------

# ============================================================
# Cell 4: Train TabTransformer + MLflow Tracking
# ============================================================
print("Training TabTransformer...\n")

neg = np.sum(y_train_sm == 0)
pos = np.sum(y_train_sm == 1)
pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(DEVICE)
print(f"Pos weight (fraud) : {pos_weight.item():.2f}")

criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer  = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3)

EPOCHS     = 30
best_auc   = 0.0
patience   = 5
no_improve = 0

train_losses = []
val_aucs     = []

with mlflow.start_run(run_name="TabTransformer_CreditCard"):

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
            torch.save(model.state_dict(), "/tmp/best_tabtransformer.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load("/tmp/best_tabtransformer.pt"))

    model.eval()
    trans_probs = []
    with torch.no_grad():
        for xb, yb in test_loader:
            trans_probs.extend(model(xb.to(DEVICE)).cpu().numpy())

    trans_probs = np.array(trans_probs)
    trans_preds = (trans_probs >= 0.5).astype(int)
    trans_auc   = roc_auc_score(y_test, trans_probs)
    trans_ap    = average_precision_score(y_test, trans_probs)

    mlflow.log_param("model",          "TabTransformer_PyTorch")
    mlflow.log_param("d_model",        64)
    mlflow.log_param("nhead",          4)
    mlflow.log_param("num_layers",     3)
    mlflow.log_param("dim_feedforward",256)
    mlflow.log_param("n_features",     N_FEATURES)
    mlflow.log_param("batch_size",     512)
    mlflow.log_param("optimizer",      "Adam_lr0.0005")
    mlflow.log_metric("auc_roc",       trans_auc)
    mlflow.log_metric("avg_precision", trans_ap)
    mlflow.log_metric("best_epoch",    EPOCHS - no_improve)
    mlflow.set_tag("project",          "FRAUDSENSE")
    mlflow.set_tag("notebook",         "NB18")

print(f"Best AUC-ROC  : {best_auc:.4f}")
print(f"Final AUC-ROC : {trans_auc:.4f}")
print(f"Avg Precision : {trans_ap:.4f}")

# COMMAND ----------

# ============================================================
# Cell 5: Training Curves
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, color=TRANS_COLOR, lw=2, label='Train Loss')
axes[0].set_title('TabTransformer - Training Loss', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
axes[0].grid(alpha=0.3)

axes[1].plot(val_aucs, color=LEGIT_COLOR, lw=2, label='Val AUC-ROC')
axes[1].axhline(y=best_auc, color=FRAUD_COLOR, lw=1.5, ls='--',
                label=f'Best: {best_auc:.4f}')
axes[1].set_title('TabTransformer - Validation AUC', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('AUC-ROC')
axes[1].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
axes[1].grid(alpha=0.3)

plt.suptitle('FRAUDSENSE NB18 - TabTransformer Training History',
             fontsize=13, fontweight='bold', color='#e6edf3', y=1.02)
plt.tight_layout()

chart_path = "/Volumes/workspace/default/fraud_data/models/nb18_tabtransformer_training.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart_path}")

# COMMAND ----------

print(classification_report(y_test, trans_preds, target_names=['Legit', 'Fraud']))

print("╔══════════════════════════════════════════════════════╗")
print("║          NB18 - RESULTS SUMMARY (Phase 4 Final)     ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  {'Model':<24} {'AUC-ROC':<10} {'Avg Precision'}   ║")
print("╠══════════════════════════════════════════════════════╣")
print(f"║  {'MLP (256-128-64)':<24} {0.9337:.4f}     {0.7866:.4f}         ║")
print(f"║  {'BiLSTM':<24} {0.9259:.4f}     {'N/A':<12}         ║")
print(f"║  {'BiGRU':<24} {0.9479:.4f}     {0.5043:.4f}         ║")
print(f"║  {'TabNet':<24} {0.9692:.4f}     {0.5347:.4f}         ║")
print(f"║  {'CNN 1D':<24} {0.9799:.4f}     {0.7001:.4f}         ║")
print(f"║  {'TabTransformer':<24} {trans_auc:.4f}     {trans_ap:.4f}  <- NEW  ║")
print("╚══════════════════════════════════════════════════════╝")

# COMMAND ----------

