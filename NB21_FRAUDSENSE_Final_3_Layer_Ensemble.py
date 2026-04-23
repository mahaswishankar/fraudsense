# Databricks notebook source
# ============================================================
# Cell 0: Install Dependencies
# ============================================================
import subprocess
packages = ["scikit-learn", "xgboost", "lightgbm", "imbalanced-learn", "mlflow", "torch"]
for pkg in packages:
    result = subprocess.run(["pip", "install", pkg, "-q"], capture_output=True, text=True)
    print(f"{pkg} installed" if result.returncode == 0 else f"{pkg} failed: {result.stderr}")

# COMMAND ----------

# ============================================================
# FRAUDSENSE - NB21: Final 3-Layer Ensemble
# Cell 1: Banner + Setup
# ============================================================
print("""
╔══════════════════════════════════════════════════════════════╗
║     FRAUDSENSE - NB21: Final 3-Layer Ensemble               ║
║     Phase 5 - The Main Event                                ║
║     0.35xStack + 0.25xLSTM + 0.20xXGB + 0.20xIsoForest     ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import functions as F
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data        import DataLoader, TensorDataset
from sklearn.ensemble        import ExtraTreesClassifier, IsolationForest
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics         import (roc_auc_score, average_precision_score,
                                     classification_report, confusion_matrix,
                                     roc_curve, precision_recall_curve)
from imblearn.over_sampling  import SMOTE
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier

plt.rcParams['figure.facecolor']  = '#0d1117'
plt.rcParams['axes.facecolor']    = '#161b22'
plt.rcParams['axes.edgecolor']    = '#30363d'
plt.rcParams['text.color']        = '#e6edf3'
plt.rcParams['axes.labelcolor']   = '#e6edf3'
plt.rcParams['xtick.color']       = '#e6edf3'
plt.rcParams['ytick.color']       = '#e6edf3'
plt.rcParams['grid.color']        = '#21262d'
plt.rcParams['font.family']       = 'monospace'

FRAUD_COLOR    = '#ff4444'
LEGIT_COLOR    = '#00d4aa'
ACCENT_COLOR   = '#f7931a'
STACK_COLOR    = '#34d399'
FRAUDSENSE_COL = '#a78bfa'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"PyTorch : {torch.__version__}")
print(f"Device  : {DEVICE}")
print("All libraries loaded")
print("FRAUDSENSE Final Ensemble starting...")

# COMMAND ----------

# ============================================================
# Cell 2: Load Gold Data
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

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# SMOTE for supervised models
smote = SMOTE(random_state=42, sampling_strategy=0.1)
X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)

print(f"Train size  : {len(X_train_scaled):,}")
print(f"Test size   : {len(X_test_scaled):,}")
print(f"Features    : {X_train_scaled.shape[1]}")
print("Data ready!")

# COMMAND ----------

# ============================================================
# Cell 3: Component 1 — Stacking Ensemble (0.35 weight)
# ============================================================
print("Training Component 1: Stacking Ensemble (weight=0.35)...")
print("Base: XGB + LGB + ExtraTrees | Meta: XGBoost\n")

N_FOLDS  = 5
skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

stack_base = {
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
        eval_metric='auc', random_state=42, verbosity=0),
    "LightGBM": LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=10,
        random_state=42, verbose=-1),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=300, class_weight='balanced',
        random_state=42, n_jobs=-1),
}

oof_train_s = np.zeros((len(X_train_scaled), len(stack_base)))
oof_test_s  = np.zeros((len(X_test_scaled),  len(stack_base)))

for col_idx, (name, clf) in enumerate(stack_base.items()):
    print(f"  {name}...")
    fold_test_preds = np.zeros((len(X_test_scaled), N_FOLDS))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
        X_ft, y_ft = X_train_scaled[tr_idx], y_train[tr_idx]
        X_fv, y_fv = X_train_scaled[val_idx], y_train[val_idx]

        sm = SMOTE(random_state=42, sampling_strategy=0.1)
        X_ft_sm, y_ft_sm = sm.fit_resample(X_ft, y_ft)

        clf.fit(X_ft_sm, y_ft_sm)
        oof_train_s[val_idx, col_idx] = clf.predict_proba(X_fv)[:, 1]
        fold_test_preds[:, fold]      = clf.predict_proba(X_test_scaled)[:, 1]

    oof_test_s[:, col_idx] = fold_test_preds.mean(axis=1)
    print(f"    CV AUC: {roc_auc_score(y_train, oof_train_s[:, col_idx]):.4f}")

# Train XGBoost meta on OOF
meta_xgb = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    scale_pos_weight=(np.sum(y_train==0)/np.sum(y_train==1)),
    eval_metric='auc', random_state=42, verbosity=0)
meta_xgb.fit(oof_train_s, y_train)

stack_probs = meta_xgb.predict_proba(oof_test_s)[:, 1]
stack_auc   = roc_auc_score(y_test, stack_probs)

print(f"\nComponent 1 (Stack) AUC : {stack_auc:.4f}")

# COMMAND ----------

# ============================================================
# Cell 4: Component 2 — BiLSTM (0.25 weight)
# ============================================================
print("Training Component 2: BiLSTM (weight=0.25)...")

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

X_train_lstm = X_train_sm.reshape(-1, 1, N_FEATURES).astype(np.float32)
X_test_lstm  = X_test_scaled.reshape(-1, 1, N_FEATURES).astype(np.float32)

train_ds = TensorDataset(torch.tensor(X_train_lstm), torch.tensor(y_train_sm.astype(np.float32)))
test_ds  = TensorDataset(torch.tensor(X_test_lstm),  torch.tensor(y_test.astype(np.float32)))
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False)

lstm_model = BiLSTMFraud(N_FEATURES).to(DEVICE)
neg = np.sum(y_train_sm == 0)
pos = np.sum(y_train_sm == 1)
pos_weight  = torch.tensor([neg/pos], dtype=torch.float32).to(DEVICE)
criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer   = torch.optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3)

best_auc, no_improve = 0.0, 0
for epoch in range(30):
    lstm_model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(lstm_model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
        optimizer.step()

    lstm_model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            probs.extend(lstm_model(xb.to(DEVICE)).cpu().numpy())
            labels.extend(yb.numpy())

    val_auc = roc_auc_score(labels, probs)
    scheduler.step(val_auc)
    print(f"  Epoch {epoch+1:>2}/30 | Val AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc   = val_auc
        no_improve = 0
        torch.save(lstm_model.state_dict(), "/tmp/fraudsense_lstm.pt")
    else:
        no_improve += 1
        if no_improve >= 5:
            print(f"  Early stopping at epoch {epoch+1}")
            break

lstm_model.load_state_dict(torch.load("/tmp/fraudsense_lstm.pt"))
lstm_model.eval()
lstm_probs = []
with torch.no_grad():
    for xb, yb in test_loader:
        lstm_probs.extend(lstm_model(xb.to(DEVICE)).cpu().numpy())

lstm_probs = np.array(lstm_probs)
lstm_auc   = roc_auc_score(y_test, lstm_probs)
print(f"\nComponent 2 (LSTM) AUC : {lstm_auc:.4f}")

# COMMAND ----------

# ============================================================
# Cell 5: Component 3 — XGBoost (0.20 weight)
# ============================================================
print("Training Component 3: XGBoost (weight=0.20)...")

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10,
    eval_metric='auc',
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train_sm, y_train_sm)

xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_probs)
print(f"Component 3 (XGBoost) AUC : {xgb_auc:.4f}")

# COMMAND ----------

# ============================================================
# Cell 6: Component 4 — Isolation Forest (0.20 weight)
# ============================================================
print("Training Component 4: Isolation Forest (weight=0.20)...")

iso_forest = IsolationForest(
    n_estimators=300,
    contamination=0.002,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_scaled)

# Convert anomaly scores to probabilities (higher = more anomalous = more fraudulent)
iso_raw    = iso_forest.decision_function(X_test_scaled)
iso_probs  = 1 - (iso_raw - iso_raw.min()) / (iso_raw.max() - iso_raw.min())
iso_auc    = roc_auc_score(y_test, iso_probs)
print(f"Component 4 (IsoForest) AUC : {iso_auc:.4f}")

# COMMAND ----------

# ============================================================
# Cell 7: FRAUDSENSE Final Weighted Ensemble
# ============================================================
print("Computing FRAUDSENSE weighted ensemble...")
print("Formula: 0.35 x Stack + 0.25 x LSTM + 0.20 x XGB + 0.20 x IsoForest\n")

# FRAUDSENSE formula
W_STACK = 0.35
W_LSTM  = 0.25
W_XGB   = 0.20
W_ISO   = 0.20

fraudsense_probs = (W_STACK * stack_probs +
                    W_LSTM  * lstm_probs  +
                    W_XGB   * xgb_probs   +
                    W_ISO   * iso_probs)

fraudsense_preds = (fraudsense_probs >= 0.5).astype(int)
fraudsense_auc   = roc_auc_score(y_test, fraudsense_probs)
fraudsense_ap    = average_precision_score(y_test, fraudsense_probs)

print(f"Component scores:")
print(f"  Stack     (w=0.35) : {stack_auc:.4f}")
print(f"  LSTM      (w=0.25) : {lstm_auc:.4f}")
print(f"  XGBoost   (w=0.20) : {xgb_auc:.4f}")
print(f"  IsoForest (w=0.20) : {iso_auc:.4f}")
print(f"\nFRAUDSENSE AUC-ROC  : {fraudsense_auc:.4f}")
print(f"FRAUDSENSE Avg Prec : {fraudsense_ap:.4f}")

# Log to MLflow
with mlflow.start_run(run_name="FRAUDSENSE_Final_Ensemble"):
    mlflow.log_metric("stack_auc",       stack_auc)
    mlflow.log_metric("lstm_auc",        lstm_auc)
    mlflow.log_metric("xgb_auc",         xgb_auc)
    mlflow.log_metric("isoforest_auc",   iso_auc)
    mlflow.log_metric("fraudsense_auc",  fraudsense_auc)
    mlflow.log_metric("fraudsense_ap",   fraudsense_ap)
    mlflow.log_param("w_stack",          W_STACK)
    mlflow.log_param("w_lstm",           W_LSTM)
    mlflow.log_param("w_xgb",            W_XGB)
    mlflow.log_param("w_isoforest",      W_ISO)
    mlflow.set_tag("project",            "FRAUDSENSE")
    mlflow.set_tag("notebook",           "NB21")
    mlflow.set_tag("phase",              "Phase5_FinalEnsemble")

# COMMAND ----------

# ============================================================
# Cell 8: Visualization — ROC + PR Curves + Component Comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC Curves for all components + FRAUDSENSE
components = {
    "Stack":          (stack_probs,       ACCENT_COLOR),
    "LSTM":           (lstm_probs,        '#a78bfa'),
    "XGBoost":        (xgb_probs,         LEGIT_COLOR),
    "IsoForest":      (iso_probs,         '#38bdf8'),
    "FRAUDSENSE":     (fraudsense_probs,  FRAUDSENSE_COL),
}

for name, (probs, color) in components.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val      = roc_auc_score(y_test, probs)
    lw           = 3 if name == "FRAUDSENSE" else 1.5
    ls           = '-' if name == "FRAUDSENSE" else '--'
    axes[0].plot(fpr, tpr, color=color, lw=lw, ls=ls,
                 label=f"{name} ({auc_val:.4f})")

axes[0].plot([0,1],[0,1], color='#30363d', lw=1, ls=':')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves — All Components', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=8, facecolor='#161b22', edgecolor='#30363d', labelcolor='#e6edf3')
axes[0].grid(alpha=0.3)

# Precision-Recall Curve for FRAUDSENSE
precision, recall, _ = precision_recall_curve(y_test, fraudsense_probs)
axes[1].plot(recall, precision, color=FRAUDSENSE_COL, lw=2)
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('FRAUDSENSE — Precision-Recall Curve', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3)

# Component AUC comparison bar chart
names_bar  = list(components.keys())
scores_bar = [roc_auc_score(y_test, p) for p, _ in components.values()]
colors_bar = [c for _, c in components.values()]
bars       = axes[2].bar(names_bar, scores_bar, color=colors_bar,
                         edgecolor='#30363d', linewidth=0.5)
axes[2].set_ylim(min(scores_bar) - 0.02, 1.0)
axes[2].set_title('Component AUC Comparison', fontsize=12, fontweight='bold')
axes[2].set_ylabel('AUC-ROC')
for bar, score in zip(bars, scores_bar):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{score:.4f}', ha='center', va='bottom', fontsize=8)
axes[2].tick_params(axis='x', rotation=15)
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('FRAUDSENSE NB21 - Final Ensemble Analysis',
             fontsize=13, fontweight='bold', color='#e6edf3', y=1.02)
plt.tight_layout()

chart_path = "/Volumes/workspace/default/fraud_data/models/nb21_fraudsense_final.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print(f"Chart saved -> {chart_path}")

# COMMAND ----------

# ============================================================
# Cell 9: Confusion Matrix
# ============================================================
cm = confusion_matrix(y_test, fraudsense_preds)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted Legit', 'Predicted Fraud'])
ax.set_yticklabels(['Actual Legit', 'Actual Fraud'])
ax.set_title('FRAUDSENSE - Confusion Matrix', fontsize=12, fontweight='bold')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=14, color='white' if cm[i,j] > cm.max()/2 else '#e6edf3')
plt.colorbar(im, ax=ax)
plt.tight_layout()

cm_path = "/Volumes/workspace/default/fraud_data/models/nb21_confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()

print(f"True Negatives  (Legit correctly identified) : {tn:,}")
print(f"False Positives (Legit flagged as fraud)     : {fp:,}")
print(f"False Negatives (Fraud missed)               : {fn:,}")
print(f"True Positives  (Fraud correctly caught)     : {tp:,}")
print(f"Fraud Catch Rate                             : {tp/(tp+fn)*100:.1f}%")

# COMMAND ----------

# ============================================================
# Cell 10: Final Results Summary
# ============================================================
print(classification_report(y_test, fraudsense_preds, target_names=['Legit', 'Fraud']))

print("╔══════════════════════════════════════════════════════════════╗")
print("║            FRAUDSENSE - FINAL RESULTS                       ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  {'Component':<28} {'Weight':<8} {'AUC-ROC'}              ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  {'Stacking Ensemble':<28} {'0.35':<8} {stack_auc:.4f}              ║")
print(f"║  {'BiLSTM':<28} {'0.25':<8} {lstm_auc:.4f}              ║")
print(f"║  {'XGBoost':<28} {'0.20':<8} {xgb_auc:.4f}              ║")
print(f"║  {'Isolation Forest':<28} {'0.20':<8} {iso_auc:.4f}              ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  {'FRAUDSENSE FINAL':<28} {'1.00':<8} {fraudsense_auc:.4f}  <- FINAL  ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  Best individual model: Extra Trees @ 0.9786                ║")
print(f"║  FRAUDSENSE improvement: {(fraudsense_auc - 0.9786)*100:+.2f}%                        ║")
print("╚══════════════════════════════════════════════════════════════╝")

print("NB21 complete!")
print("Next: NB22 - FRAUDSENSE vs XGBoost Benchmark")

# COMMAND ----------

