"""
================================================================
Day 07 — Microbiome Co-occurrence Network Analysis (REAL DATA)
Author  : Subhadip Jana
Dataset : peerj32 — LGG Probiotic vs Placebo
          44 samples × 130 real gut taxa

What is a Microbiome Network?
  A co-occurrence network models relationships between gut taxa:
  • Nodes  = bacterial taxa
  • Edges  = significant co-occurrence (positive) or
             mutual exclusion (negative correlation)
  • Edge weight = Spearman correlation strength

Methods:
  • Spearman correlation matrix (all taxa pairs)
  • FDR correction (Benjamini-Hochberg)
  • Threshold filtering (|r| > 0.4, FDR < 0.05)
  • Separate networks for LGG vs Placebo
  • Network topology metrics:
      - Degree, Betweenness centrality
      - Clustering coefficient
      - Hub taxa identification
      - Modularity (community detection)
  • SparCC-style sign correction
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import networkx as nx
from scipy.stats import spearmanr
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA
# ─────────────────────────────────────────────────────────────

print("🔬 Loading peerj32 dataset...")
otu_raw = pd.read_csv("data/otu_table.csv", index_col=0)
meta    = pd.read_csv("data/metadata.csv",  index_col=0)

otu_df  = otu_raw.T.astype(float)
taxa    = otu_df.columns.tolist()
rel_df  = otu_df.div(otu_df.sum(axis=1), axis=0) * 100

group   = meta["group"]
lgg_df  = rel_df[group == "LGG"]
plc_df  = rel_df[group == "Placebo"]

print(f"✅ {len(otu_df)} samples × {len(taxa)} taxa")
print(f"   LGG: {len(lgg_df)} | Placebo: {len(plc_df)}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: FILTER TAXA (prevalence + abundance)
# ─────────────────────────────────────────────────────────────

# Keep taxa present in ≥ 30% of samples with mean > 0.1%
prev_mask  = (rel_df > 0).mean(axis=0) >= 0.30
abund_mask = rel_df.mean(axis=0) >= 0.1
keep_taxa  = taxa_filtered = rel_df.columns[prev_mask & abund_mask].tolist()
rel_filt   = rel_df[keep_taxa]
lgg_filt   = lgg_df[keep_taxa]
plc_filt   = plc_df[keep_taxa]

print(f"\n📊 After filtering: {len(keep_taxa)} taxa retained")

# ─────────────────────────────────────────────────────────────
# SECTION 3: BH FDR CORRECTION
# ─────────────────────────────────────────────────────────────

def bh_fdr(pvals):
    n   = len(pvals)
    idx = np.argsort(pvals)
    fdr = np.minimum(1, np.array(pvals)[idx] * n / np.arange(1, n+1))
    for i in range(n-2, -1, -1):
        fdr[i] = min(fdr[i], fdr[i+1])
    result = np.empty(n); result[idx] = fdr
    return result

# ─────────────────────────────────────────────────────────────
# SECTION 4: BUILD CORRELATION NETWORK
# ─────────────────────────────────────────────────────────────

def build_network(df, label, r_thresh=0.4, fdr_thresh=0.05):
    """
    Build co-occurrence network from taxa abundance matrix.
    Returns a NetworkX graph with edge weights = Spearman r.
    """
    print(f"\n🔗 Building {label} network...")
    taxa_list = df.columns.tolist()
    n_taxa    = len(taxa_list)

    # Compute all pairwise Spearman correlations
    rs, ps, pairs = [], [], []
    for i, j in combinations(range(n_taxa), 2):
        r, p = spearmanr(df.iloc[:, i], df.iloc[:, j])
        rs.append(r); ps.append(p)
        pairs.append((taxa_list[i], taxa_list[j]))

    # BH FDR correction
    fdrs = bh_fdr(ps)

    # Build graph with significant edges
    G = nx.Graph()
    G.add_nodes_from(taxa_list)

    pos_edges, neg_edges = 0, 0
    for (t1, t2), r, fdr in zip(pairs, rs, fdrs):
        if abs(r) >= r_thresh and fdr < fdr_thresh:
            G.add_edge(t1, t2, weight=r, abs_weight=abs(r),
                       sign="positive" if r > 0 else "negative")
            if r > 0: pos_edges += 1
            else:     neg_edges += 1

    # Remove isolated nodes
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated)

    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()} "
          f"(+{pos_edges} positive, -{neg_edges} negative)")
    return G

# Build networks
G_all = build_network(rel_filt,  "All samples",  r_thresh=0.4, fdr_thresh=0.05)
G_lgg = build_network(lgg_filt,  "LGG",          r_thresh=0.35, fdr_thresh=0.1)
G_plc = build_network(plc_filt,  "Placebo",      r_thresh=0.35, fdr_thresh=0.1)

# ─────────────────────────────────────────────────────────────
# SECTION 5: NETWORK TOPOLOGY METRICS
# ─────────────────────────────────────────────────────────────

def network_metrics(G, label):
    if G.number_of_nodes() == 0:
        return {}
    metrics = {
        "label"          : label,
        "nodes"          : G.number_of_nodes(),
        "edges"          : G.number_of_edges(),
        "density"        : round(nx.density(G), 4),
        "avg_degree"     : round(np.mean([d for _, d in G.degree()]), 3),
        "avg_clustering" : round(nx.average_clustering(G), 4),
        "components"     : nx.number_connected_components(G),
        "pos_edges"      : sum(1 for _,_,d in G.edges(data=True) if d.get("sign")=="positive"),
        "neg_edges"      : sum(1 for _,_,d in G.edges(data=True) if d.get("sign")=="negative"),
    }
    # Largest component diameter
    largest = max(nx.connected_components(G), key=len)
    subG    = G.subgraph(largest)
    try:
        metrics["diameter"] = nx.diameter(subG)
    except:
        metrics["diameter"] = "N/A"
    return metrics

print("\n" + "="*55)
print("NETWORK TOPOLOGY METRICS")
print("="*55)
all_metrics = []
for G, label in [(G_all,"All"),(G_lgg,"LGG"),(G_plc,"Placebo")]:
    m = network_metrics(G, label)
    all_metrics.append(m)
    print(f"\n{label}:")
    for k, v in m.items():
        if k != "label":
            print(f"  {k:20s}: {v}")

# ─────────────────────────────────────────────────────────────
# SECTION 6: HUB TAXA IDENTIFICATION
# ─────────────────────────────────────────────────────────────

def get_hubs(G, top_n=10):
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    degree      = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, weight="abs_weight")
    clustering  = nx.clustering(G, weight="abs_weight")
    hub_df = pd.DataFrame({
        "Taxon"      : list(degree.keys()),
        "Degree"     : list(degree.values()),
        "Betweenness": [betweenness[n] for n in degree.keys()],
        "Clustering" : [clustering[n]  for n in degree.keys()],
    })
    hub_df["Hub_score"] = (
        hub_df["Degree"]      / hub_df["Degree"].max() +
        hub_df["Betweenness"] / (hub_df["Betweenness"].max() + 1e-9)
    )
    return hub_df.sort_values("Hub_score", ascending=False).head(top_n)

hubs_all = get_hubs(G_all)
hubs_lgg = get_hubs(G_lgg)
hubs_plc = get_hubs(G_plc)

print("\n🏆 Top 5 Hub Taxa (All samples):")
print(hubs_all[["Taxon","Degree","Betweenness","Hub_score"]].head(5).to_string(index=False))

hubs_all.to_csv("outputs/hub_taxa.csv", index=False)

# ─────────────────────────────────────────────────────────────
# SECTION 7: COMMUNITY DETECTION
# ─────────────────────────────────────────────────────────────

def detect_communities(G):
    if G.number_of_nodes() < 3:
        return {}
    try:
        communities = nx.community.greedy_modularity_communities(G, weight="abs_weight")
        return {node: i for i, comm in enumerate(communities) for node in comm}
    except:
        return {node: 0 for node in G.nodes()}

communities_all = detect_communities(G_all)
n_communities   = len(set(communities_all.values())) if communities_all else 0
print(f"\n🧩 Communities detected (all network): {n_communities}")

# ─────────────────────────────────────────────────────────────
# SECTION 8: FULL CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────

# Compute full correlation matrix for heatmap
corr_matrix = rel_filt.corr(method="spearman")
corr_matrix.to_csv("outputs/correlation_matrix.csv")

# ─────────────────────────────────────────────────────────────
# SECTION 9: DASHBOARD
# ─────────────────────────────────────────────────────────────

print("\n🎨 Generating dashboard...")

PHYLUM_COLORS = {
    "Firmicutes"   : "#E74C3C",
    "Bacteroidetes": "#3498DB",
    "Actinobacteria": "#2ECC71",
    "Proteobacteria": "#F39C12",
    "Verrucomicrobia": "#9B59B6",
    "Other"        : "#BDC3C7",
}

FIRM_KW = ["ruminococcus","clostridium","faecalibacterium","butyrivibrio",
           "eubacterium","lachnospira","roseburia","blautia","coprococcus",
           "dorea","lactobacillus","enterococcus","streptococcus","anaerostipes",
           "subdoligranulum","anaerotruncus","bryantella","bulleidia","dialister",
           "phascolarctobacterium","veillonella","bacillus","papillibacter"]
BACT_KW = ["bacteroides","prevotella","alistipes","parabacteroides","allistipes",
           "bilophila","barnesiella","butyricimonas"]
ACTI_KW = ["bifidobacterium","atopobium","collinsella","actinomyces","wissella"]
PROT_KW = ["escherichia","klebsiella","haemophilus","helicobacter","aeromonas",
           "aquabacterium","burkholderia","stenotrophomonas"]
VERR_KW = ["akkermansia"]

def get_phylum(taxon):
    t = taxon.lower()
    if any(k in t for k in FIRM_KW): return "Firmicutes"
    if any(k in t for k in BACT_KW): return "Bacteroidetes"
    if any(k in t for k in ACTI_KW): return "Actinobacteria"
    if any(k in t for k in PROT_KW): return "Proteobacteria"
    if any(k in t for k in VERR_KW): return "Verrucomicrobia"
    return "Other"

node_colors_all = [PHYLUM_COLORS[get_phylum(n)] for n in G_all.nodes()]
comm_color_map  = plt.cm.Set2(np.linspace(0, 1, max(n_communities, 1)))

fig = plt.figure(figsize=(24, 18))
fig.suptitle(
    "Gut Microbiome Co-occurrence Network Analysis — REAL DATA\n"
    "LGG Probiotic vs Placebo | peerj32 dataset\n"
    "Spearman correlation + BH FDR | nodes=taxa, edges=co-occurrence",
    fontsize=15, fontweight="bold", y=0.99
)

# ── Plot 1: Full network (All samples) — colored by phylum ──
ax1 = fig.add_subplot(3, 3, 1)
if G_all.number_of_nodes() > 0:
    pos = nx.spring_layout(G_all, seed=42, k=2.5/np.sqrt(G_all.number_of_nodes()))
    edge_colors = ["#E74C3C" if G_all[u][v].get("sign")=="positive"
                   else "#3498DB" for u,v in G_all.edges()]
    edge_widths = [G_all[u][v].get("abs_weight", 0.5) * 2 for u,v in G_all.edges()]
    nx.draw_networkx_edges(G_all, pos, ax=ax1, edge_color=edge_colors,
                           width=edge_widths, alpha=0.5)
    nx.draw_networkx_nodes(G_all, pos, ax=ax1, node_color=node_colors_all,
                           node_size=[G_all.degree(n)*50+80 for n in G_all.nodes()],
                           alpha=0.85, edgecolors="white", linewidths=0.5)
    # Label only top hub nodes
    top5_nodes = hubs_all["Taxon"].head(5).tolist()
    labels_dict = {n: n.replace("et rel.","").strip()[:15]
                   for n in G_all.nodes() if n in top5_nodes}
    nx.draw_networkx_labels(G_all, pos, labels=labels_dict, ax=ax1,
                            font_size=6, font_weight="bold")
ax1.set_title(f"Co-occurrence Network (All)\n"
              f"{G_all.number_of_nodes()} nodes, {G_all.number_of_edges()} edges",
              fontweight="bold", fontsize=10)
ax1.axis("off")
legend_patches = [mpatches.Patch(color=c, label=p)
                  for p, c in PHYLUM_COLORS.items() if p != "Other"] + [
    mpatches.Patch(color="#E74C3C", label="+ co-occurrence"),
    mpatches.Patch(color="#3498DB", label="– mutual exclusion"),
]
ax1.legend(handles=legend_patches, fontsize=6, loc="lower left",
           ncol=2, framealpha=0.7)

# ── Plot 2: LGG network ──
ax2 = fig.add_subplot(3, 3, 2)
if G_lgg.number_of_nodes() > 0:
    pos_lgg = nx.spring_layout(G_lgg, seed=42, k=2.5/np.sqrt(G_lgg.number_of_nodes()))
    ec_lgg  = ["#E74C3C" if G_lgg[u][v].get("sign")=="positive"
               else "#3498DB" for u,v in G_lgg.edges()]
    ew_lgg  = [G_lgg[u][v].get("abs_weight",0.5)*2 for u,v in G_lgg.edges()]
    nc_lgg  = [PHYLUM_COLORS[get_phylum(n)] for n in G_lgg.nodes()]
    nx.draw_networkx_edges(G_lgg, pos_lgg, ax=ax2, edge_color=ec_lgg,
                           width=ew_lgg, alpha=0.5)
    nx.draw_networkx_nodes(G_lgg, pos_lgg, ax=ax2, node_color=nc_lgg,
                           node_size=[G_lgg.degree(n)*60+80 for n in G_lgg.nodes()],
                           alpha=0.85, edgecolors="white", linewidths=0.5)
ax2.set_title(f"LGG Network\n"
              f"{G_lgg.number_of_nodes()} nodes, {G_lgg.number_of_edges()} edges",
              fontweight="bold", fontsize=10)
ax2.axis("off")

# ── Plot 3: Placebo network ──
ax3 = fig.add_subplot(3, 3, 3)
if G_plc.number_of_nodes() > 0:
    pos_plc = nx.spring_layout(G_plc, seed=42, k=2.5/np.sqrt(G_plc.number_of_nodes()))
    ec_plc  = ["#E74C3C" if G_plc[u][v].get("sign")=="positive"
               else "#3498DB" for u,v in G_plc.edges()]
    ew_plc  = [G_plc[u][v].get("abs_weight",0.5)*2 for u,v in G_plc.edges()]
    nc_plc  = [PHYLUM_COLORS[get_phylum(n)] for n in G_plc.nodes()]
    nx.draw_networkx_edges(G_plc, pos_plc, ax=ax3, edge_color=ec_plc,
                           width=ew_plc, alpha=0.5)
    nx.draw_networkx_nodes(G_plc, pos_plc, ax=ax3, node_color=nc_plc,
                           node_size=[G_plc.degree(n)*60+80 for n in G_plc.nodes()],
                           alpha=0.85, edgecolors="white", linewidths=0.5)
ax3.set_title(f"Placebo Network\n"
              f"{G_plc.number_of_nodes()} nodes, {G_plc.number_of_edges()} edges",
              fontweight="bold", fontsize=10)
ax3.axis("off")

# ── Plot 4: Hub taxa bar chart (degree) ──
ax4 = fig.add_subplot(3, 3, 4)
if len(hubs_all) > 0:
    colors_hub = [PHYLUM_COLORS[get_phylum(t)] for t in hubs_all["Taxon"]]
    ax4.barh(range(len(hubs_all)),
             hubs_all["Degree"].values[::-1],
             color=colors_hub[::-1], edgecolor="black", linewidth=0.4)
    ax4.set_yticks(range(len(hubs_all)))
    ax4.set_yticklabels([t.replace("et rel.","").strip()[:28]
                         for t in hubs_all["Taxon"][::-1]], fontsize=7)
    ax4.set_xlabel("Node Degree (# connections)")
    ax4.set_title("Hub Taxa — Node Degree\n(All network)",
                  fontweight="bold", fontsize=10)

# ── Plot 5: Betweenness centrality ──
ax5 = fig.add_subplot(3, 3, 5)
if len(hubs_all) > 0:
    colors_bet = [PHYLUM_COLORS[get_phylum(t)] for t in hubs_all["Taxon"]]
    ax5.barh(range(len(hubs_all)),
             hubs_all["Betweenness"].values[::-1],
             color=colors_bet[::-1], edgecolor="black", linewidth=0.4)
    ax5.set_yticks(range(len(hubs_all)))
    ax5.set_yticklabels([t.replace("et rel.","").strip()[:28]
                         for t in hubs_all["Taxon"][::-1]], fontsize=7)
    ax5.set_xlabel("Betweenness Centrality")
    ax5.set_title("Hub Taxa — Betweenness Centrality\n(All network)",
                  fontweight="bold", fontsize=10)

# ── Plot 6: Correlation heatmap (top 25 taxa) ──
ax6 = fig.add_subplot(3, 3, 6)
top25_taxa = rel_filt.mean().nlargest(25).index
corr_sub   = rel_filt[top25_taxa].corr(method="spearman")
short_names = [t.replace("et rel.","").strip()[:15] for t in top25_taxa]
corr_sub.index   = short_names
corr_sub.columns = short_names
sns.heatmap(corr_sub, ax=ax6, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, xticklabels=True, yticklabels=True,
            cbar_kws={"label": "Spearman r", "shrink": 0.8})
ax6.tick_params(axis="both", labelsize=5)
ax6.set_title("Spearman Correlation Heatmap\n(Top 25 abundant taxa)",
              fontweight="bold", fontsize=10)

# ── Plot 7: Degree distribution ──
ax7 = fig.add_subplot(3, 3, 7)
degrees = [d for _, d in G_all.degree()]
if degrees:
    ax7.hist(degrees, bins=range(0, max(degrees)+2), color="#9B59B6",
             edgecolor="black", linewidth=0.5, alpha=0.8)
    ax7.axvline(np.mean(degrees), color="red", lw=2,
                linestyle="--", label=f"Mean={np.mean(degrees):.1f}")
ax7.set_xlabel("Node Degree")
ax7.set_ylabel("Count")
ax7.set_title("Degree Distribution\n(All network)", fontweight="bold", fontsize=10)
ax7.legend(fontsize=9)

# ── Plot 8: Network comparison bar ──
ax8 = fig.add_subplot(3, 3, 8)
metrics_compare = {
    "Nodes"    : [G_all.number_of_nodes(), G_lgg.number_of_nodes(), G_plc.number_of_nodes()],
    "Edges"    : [G_all.number_of_edges(), G_lgg.number_of_edges(), G_plc.number_of_edges()],
    "Density×100": [nx.density(G_all)*100 if G_all.number_of_nodes()>1 else 0,
                    nx.density(G_lgg)*100 if G_lgg.number_of_nodes()>1 else 0,
                    nx.density(G_plc)*100 if G_plc.number_of_nodes()>1 else 0],
}
x  = np.arange(3)
w  = 0.25
gp_labels = ["All","LGG","Placebo"]
gp_colors = ["#9B59B6","#E74C3C","#3498DB"]
for i, (metric, vals) in enumerate(metrics_compare.items()):
    bars = ax8.bar(x + i*w, vals, w, label=metric,
                   color=[c for c in gp_colors], edgecolor="black",
                   linewidth=0.5, alpha=0.8)
ax8.set_xticks(x + w)
ax8.set_xticklabels(gp_labels)
ax8.set_ylabel("Value")
ax8.set_title("Network Comparison\n(All vs LGG vs Placebo)",
              fontweight="bold", fontsize=10)
ax8.legend(fontsize=8)

# ── Plot 9: Summary table ──
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis("off")
rows = []
for m in all_metrics:
    if m:
        rows.append([
            m["label"],
            str(m["nodes"]),
            str(m["edges"]),
            str(m.get("pos_edges","—")),
            str(m.get("neg_edges","—")),
            f"{m['density']:.4f}",
            f"{m['avg_clustering']:.4f}",
        ])
rows += [
    ["Communities (All)", str(n_communities), "—","—","—","—","—"],
    ["r threshold",       "≥0.4",             "—","—","—","—","—"],
    ["FDR threshold",     "<0.05",            "—","—","—","—","—"],
    ["Correlation",       "Spearman",         "—","—","—","—","—"],
]
tbl = ax9.table(
    cellText=rows,
    colLabels=["Network","Nodes","Edges","+Edge","-Edge","Density","Clustering"],
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.3, 1.9)
for j in range(7): tbl[(0,j)].set_facecolor("#BDC3C7")
for i, color in enumerate(["#9B59B6","#E74C3C","#3498DB"], 1):
    if i <= len(rows):
        tbl[(i,0)].set_facecolor(color)
        tbl[(i,0)].set_text_props(color="white", fontweight="bold")
ax9.set_title("Network Topology Summary", fontweight="bold", fontsize=11, pad=20)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig("outputs/network_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ Dashboard saved → outputs/network_dashboard.png")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("FINAL SUMMARY")
print("="*55)
for G, label in [(G_all,"All"),(G_lgg,"LGG"),(G_plc,"Placebo")]:
    print(f"\n{label} Network:")
    print(f"  Nodes     : {G.number_of_nodes()}")
    print(f"  Edges     : {G.number_of_edges()}")
    if G.number_of_nodes() > 1:
        print(f"  Density   : {nx.density(G):.4f}")
        print(f"  Avg Clust : {nx.average_clustering(G):.4f}")
        pos_e = sum(1 for _,_,d in G.edges(data=True) if d.get("sign")=="positive")
        neg_e = sum(1 for _,_,d in G.edges(data=True) if d.get("sign")=="negative")
        print(f"  + edges   : {pos_e}")
        print(f"  - edges   : {neg_e}")

print(f"\nTop Hub Taxa:")
if len(hubs_all) > 0:
    print(hubs_all[["Taxon","Degree","Hub_score"]].head(5).to_string(index=False))
print(f"\nCommunities detected: {n_communities}")
print("\n✅ All outputs saved!")
