# Databricks notebook source
# Install NetworkX
import subprocess
result = subprocess.run(["pip", "install", "networkx", "-q"], capture_output=True, text=True)
print("✅ NetworkX installed!" if result.returncode == 0 else f"❌ Error: {result.stderr}")

# COMMAND ----------


# ============================================================
# FRAUDSENSE — NB07: Graph Fraud Detection
# Cell 1: Banner + Setup
# ============================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║     FRAUDSENSE — NB07: Graph Fraud Detection                ║
║     PageRank + Connected Components on Transaction Network  ║
╚══════════════════════════════════════════════════════════════╝
""")

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

spark = SparkSession.builder.getOrCreate()

GOLD_PATH    = "/Volumes/workspace/default/fraud_data/gold"

# ── Plot styling ─────────────────────────────────────────────
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor']   = '#161b22'
plt.rcParams['axes.edgecolor']   = '#30363d'
plt.rcParams['text.color']       = '#e6edf3'
plt.rcParams['axes.labelcolor']  = '#e6edf3'
plt.rcParams['xtick.color']      = '#e6edf3'
plt.rcParams['ytick.color']      = '#e6edf3'
plt.rcParams['grid.color']       = '#21262d'
plt.rcParams['font.family']      = 'monospace'

FRAUD_COLOR  = '#ff4444'
LEGIT_COLOR  = '#00d4aa'
ACCENT_COLOR = '#f7931a'

print("✅ Libraries loaded")
print("✅ NetworkX ready for graph analysis")
print("\n🕸️  Starting Graph Fraud Detection...")

# COMMAND ----------

# ============================================================
# Cell 2: Build Transaction Network from PaySim
# ============================================================

print("📥 Loading PaySim Gold dataset...")

paysim = spark.read.format("delta").load(f"{GOLD_PATH}/paysim").toPandas()

# ── Filter only TRANSFER and CASH_OUT (fraud-relevant types) ─
network_df = paysim[paysim['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

print(f"✅ Total PaySim transactions : {len(paysim):,}")
print(f"✅ Network transactions      : {len(network_df):,} (TRANSFER + CASH_OUT only)")
print(f"✅ Fraud in network          : {network_df['Class'].sum():,}")

# ── Sample for graph construction (full 6M too large for NetworkX) ──
# Stratified sample — keep all fraud, sample legit
fraud_df    = network_df[network_df['Class'] == 1]
legit_df    = network_df[network_df['Class'] == 0].sample(
                min(50000, len(network_df[network_df['Class']==0])),
                random_state=42)
graph_df    = pd.concat([fraud_df, legit_df]).reset_index(drop=True)

print(f"\n📊 Graph sample:")
print(f"   Fraud transactions  : {len(fraud_df):,}")
print(f"   Legit transactions  : {len(legit_df):,}")
print(f"   Total graph edges   : {len(graph_df):,}")

# ── Build directed graph ──────────────────────────────────────
print("\n🕸️  Building transaction graph...")
G = nx.DiGraph()

for _, row in graph_df.iterrows():
    src  = str(row['nameOrig'])
    dst  = str(row['nameDest'])
    G.add_edge(src, dst,
               amount=row['Amount'],
               is_fraud=int(row['Class']),
               tx_type=row['type'])

print(f"✅ Graph built!")
print(f"   Nodes (accounts)    : {G.number_of_nodes():,}")
print(f"   Edges (transactions): {G.number_of_edges():,}")
print(f"   Is directed         : {G.is_directed()}")

# COMMAND ----------

# ============================================================
# Cell 3: PageRank — Find High Influence Fraud Accounts
# ============================================================

print("📊 Computing PageRank scores...")
print("   (Higher PageRank = more central/influential in network)\n")

# ── Compute PageRank ──────────────────────────────────────────
pagerank_scores = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)

# ── Convert to DataFrame ──────────────────────────────────────
pr_df = pd.DataFrame([
    {'account': node, 'pagerank': score}
    for node, score in pagerank_scores.items()
]).sort_values('pagerank', ascending=False)

# ── Tag accounts as fraud-connected ──────────────────────────
fraud_accounts = set(
    graph_df[graph_df['Class'] == 1]['nameOrig'].tolist() +
    graph_df[graph_df['Class'] == 1]['nameDest'].tolist()
)
pr_df['is_fraud_account'] = pr_df['account'].isin(fraud_accounts).astype(int)

# ── Analysis ──────────────────────────────────────────────────
fraud_pr    = pr_df[pr_df['is_fraud_account'] == 1]['pagerank']
legit_pr    = pr_df[pr_df['is_fraud_account'] == 0]['pagerank']

print(f"✅ PageRank computed for {len(pr_df):,} accounts")
print(f"\n📊 PageRank Statistics:")
print(f"   Fraud accounts — Mean PR : {fraud_pr.mean():.6f}  Max PR: {fraud_pr.max():.6f}")
print(f"   Legit accounts — Mean PR : {legit_pr.mean():.6f}  Max PR: {legit_pr.max():.6f}")
print(f"\n🎯 Top 15 highest PageRank accounts:")
print(f"{'Rank':<6} {'Account':<20} {'PageRank':>12} {'Fraud?':>8}")
print("─" * 50)
for i, (_, row) in enumerate(pr_df.head(15).iterrows(), 1):
    fraud_tag = "🚨 FRAUD" if row['is_fraud_account'] else "✅ LEGIT"
    print(f"{i:<6} {row['account'][:18]:<20} {row['pagerank']:>12.6f} {fraud_tag:>8}")

# COMMAND ----------

# ============================================================
# Cell 4: Connected Components — Find Fraud Clusters/Rings
# ============================================================

print("🔍 Computing Connected Components...")
print("   (Groups of accounts all connected to each other)\n")

# ── Work on undirected version for connected components ───────
G_undirected = G.to_undirected()
components   = list(nx.connected_components(G_undirected))
components_sorted = sorted(components, key=len, reverse=True)

print(f"✅ Total connected components : {len(components):,}")
print(f"   Largest component size     : {len(components_sorted[0]):,} accounts")
print(f"   Smallest component size    : {len(components_sorted[-1]):,} accounts")

# ── Analyze fraud concentration in components ─────────────────
component_stats = []
for i, comp in enumerate(components_sorted[:200]):  # Analyze top 200
    comp_accounts = comp
    comp_edges    = [(u, v, d) for u, v, d in G.edges(data=True)
                     if u in comp_accounts or v in comp_accounts]
    fraud_edges   = [e for e in comp_edges if e[2].get('is_fraud', 0) == 1]
    total_amount  = sum(e[2].get('amount', 0) for e in comp_edges)
    fraud_amount  = sum(e[2].get('amount', 0) for e in fraud_edges)

    component_stats.append({
        'component_id'   : i,
        'size'           : len(comp_accounts),
        'total_edges'    : len(comp_edges),
        'fraud_edges'    : len(fraud_edges),
        'fraud_rate'     : len(fraud_edges) / max(len(comp_edges), 1),
        'total_amount'   : total_amount,
        'fraud_amount'   : fraud_amount
    })

comp_df = pd.DataFrame(component_stats)
high_fraud_comps = comp_df[comp_df['fraud_rate'] > 0.5].sort_values('fraud_rate', ascending=False)

print(f"\n🚨 High-fraud components (>50% fraud rate): {len(high_fraud_comps)}")
print(f"\n{'ID':<6} {'Size':<8} {'Edges':<8} {'Fraud%':>8} {'Fraud Amount':>15}")
print("─" * 50)
for _, row in high_fraud_comps.head(10).iterrows():
    print(f"{int(row['component_id']):<6} {int(row['size']):<8} "
          f"{int(row['total_edges']):<8} {row['fraud_rate']*100:>7.1f}% "
          f"${row['fraud_amount']:>14,.0f}")

# COMMAND ----------

# ============================================================
# Cell 5: Graph Visualization — Fraud Network
# ============================================================

print("🎨 Visualizing fraud network...")

# ── Sample a small subgraph for visualization ─────────────────
# Take the most interesting component with high fraud rate
if len(high_fraud_comps) > 0:
    target_comp_id = int(high_fraud_comps.iloc[0]['component_id'])
    target_comp    = components_sorted[target_comp_id]
else:
    target_comp    = components_sorted[0]

# Limit to 80 nodes max for clean visualization
target_nodes  = list(target_comp)[:80]
subgraph      = G.subgraph(target_nodes)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('FRAUDSENSE — Transaction Network: Fraud Ring Detection',
             fontsize=14, fontweight='bold', color='#e6edf3')

# ── Left: Full component colored by fraud ─────────────────────
pos = nx.spring_layout(subgraph, seed=42, k=0.8)

node_colors = []
node_sizes  = []
for node in subgraph.nodes():
    is_fraud = node in fraud_accounts
    node_colors.append(FRAUD_COLOR if is_fraud else LEGIT_COLOR)
    pr_score = pagerank_scores.get(node, 0)
    node_sizes.append(500 + pr_score * 50000)

edge_colors = []
for u, v, data in subgraph.edges(data=True):
    edge_colors.append(FRAUD_COLOR if data.get('is_fraud', 0) == 1 else '#30363d')

nx.draw_networkx_nodes(subgraph, pos, ax=axes[0],
                       node_color=node_colors,
                       node_size=node_sizes, alpha=0.85)
nx.draw_networkx_edges(subgraph, pos, ax=axes[0],
                       edge_color=edge_colors,
                       arrows=True, arrowsize=10,
                       width=1.5, alpha=0.6)
axes[0].set_title('Fraud Network — Node size = PageRank\nRed = Fraud account, Teal = Legit',
                   fontsize=11, fontweight='bold', color='#e6edf3')
axes[0].axis('off')

# ── Right: PageRank distribution ──────────────────────────────
axes[1].hist(legit_pr, bins=60, color=LEGIT_COLOR, alpha=0.6,
             label='Legit Accounts', density=True)
axes[1].hist(fraud_pr, bins=60, color=FRAUD_COLOR, alpha=0.7,
             label='Fraud Accounts', density=True)
axes[1].set_title('PageRank Score Distribution\nFraud vs Legitimate Accounts',
                   fontsize=11, fontweight='bold', color='#e6edf3')
axes[1].set_xlabel('PageRank Score', fontsize=10)
axes[1].set_ylabel('Density', fontsize=10)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/fraud_network.png', dpi=150,
            bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("✅ Network visualization saved!")

# COMMAND ----------

# ============================================================
# Cell 6: Extract Graph Features → Save for ML Models
# ============================================================

print("💾 Extracting graph features for ML models...")

# ── Compute additional graph metrics ─────────────────────────
print("   Computing degree centrality...")
in_degree   = dict(G.in_degree())
out_degree  = dict(G.out_degree())

print("   Computing clustering coefficients...")
G_undir     = G.to_undirected()
clustering  = nx.clustering(G_undir)

# ── Build feature dataframe per account ──────────────────────
graph_features = []
for node in G.nodes():
    graph_features.append({
        'account'           : node,
        'pagerank'          : pagerank_scores.get(node, 0),
        'in_degree'         : in_degree.get(node, 0),
        'out_degree'        : out_degree.get(node, 0),
        'total_degree'      : in_degree.get(node, 0) + out_degree.get(node, 0),
        'clustering_coef'   : clustering.get(node, 0),
        'is_fraud_account'  : int(node in fraud_accounts)
    })

graph_feat_df = pd.DataFrame(graph_features)

# ── Save as Delta table ───────────────────────────────────────
graph_spark_df = spark.createDataFrame(graph_feat_df)
graph_spark_df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(f"{GOLD_PATH}/graph_features")

# ── Summary stats ─────────────────────────────────────────────
fraud_nodes = graph_feat_df[graph_feat_df['is_fraud_account'] == 1]
legit_nodes = graph_feat_df[graph_feat_df['is_fraud_account'] == 0]

print(f"\n✅ Graph features saved!")
print(f"   Total accounts analyzed : {len(graph_feat_df):,}")
print(f"   Fraud-connected accounts: {len(fraud_nodes):,}")
print(f"\n📊 Graph Feature Stats — Fraud vs Legit:")
print(f"{'Feature':<22} {'Fraud Mean':>12} {'Legit Mean':>12}")
print("─" * 48)
for feat in ['pagerank', 'in_degree', 'out_degree', 'clustering_coef']:
    print(f"{feat:<22} {fraud_nodes[feat].mean():>12.6f} {legit_nodes[feat].mean():>12.6f}")

print(f"\n✅ NB07 Complete — Graph Fraud Detection done!")
print(f"🚀 Next → NB08: Classical ML — Logistic Regression + Linear SVM")

# COMMAND ----------

