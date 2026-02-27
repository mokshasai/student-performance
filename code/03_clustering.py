"""
Student Performance Analysis
=============================
Script 03 — Clustering Analysis

Methods : K-Means, Hierarchical (Agglomerative / Ward), DBSCAN
Goal    : Discover natural student groupings without using labels.
          Compare all three algorithms.
Dataset : UCI Student Performance (student-mat.csv)
Author  : Data Science Project 2025
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
#    Same preprocessing as PCA — numeric only, no labels
# ─────────────────────────────────────────────────────────────
raw_df = pd.read_csv("../data/student-mat.csv", sep=";")

binary_cols = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]
for col in binary_cols:
    raw_df[col] = LabelEncoder().fit_transform(raw_df[col])

raw_df = pd.get_dummies(raw_df, columns=["Mjob", "Fjob", "reason", "guardian"])

# Save true labels for later comparison — NOT used during clustering
def perf_label(g):
    if g >= 14: return "High"
    elif g >= 10: return "Medium"
    else: return "Low"

true_labels = raw_df["G3"].apply(perf_label)
pca_features = raw_df.drop(columns=["G3", "G1", "G2"])

# Normalize
X_scaled = StandardScaler().fit_transform(pca_features)

# Reduce to 3D via PCA for visualization & computation
pca3 = PCA(n_components=3)
X_3d = pca3.fit_transform(X_scaled)
var3 = pca3.explained_variance_ratio_
print(f"PCA 3D input — retained variance: {var3.sum()*100:.1f}%")
print(f"Clustering input shape: {X_3d.shape}")

# Color palette
CLUSTER_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
                  "#1abc9c", "#e67e22", "#34495e"]

plt.rcParams["figure.facecolor"] = "#f8f9fa"

# ─────────────────────────────────────────────────────────────
# 2. KMEANS — SILHOUETTE METHOD TO CHOOSE k
# ─────────────────────────────────────────────────────────────
print("\n── K-Means: Silhouette Method ──")
sil_scores = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_3d)
    sil_scores[k] = silhouette_score(X_3d, lbl)
    print(f"  k={k}: silhouette = {sil_scores[k]:.4f}")

# Pick top 3 k values
best_ks = sorted(sil_scores, key=sil_scores.get, reverse=True)[:3]
best_ks.sort()
print(f"\n  Best 3 k values: {best_ks}")

# Plot silhouette scores
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_facecolor("#f8f9fa")
ax.plot(list(sil_scores.keys()), list(sil_scores.values()),
        marker="o", color="#2c3e50", linewidth=2, markersize=8)
for k, s in sil_scores.items():
    ax.text(k, s + 0.003, f"{s:.3f}", ha="center", fontsize=8)
for k in best_ks:
    ax.axvline(k, color="#e74c3c", linestyle="--", alpha=0.4)
ax.set_xlabel("Number of Clusters (k)", fontsize=11)
ax.set_ylabel("Silhouette Score", fontsize=11)
ax.set_title("Silhouette Method — Choosing Optimal k", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../Website/fig_cluster_silhouette.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_cluster_silhouette.png")

# ─────────────────────────────────────────────────────────────
# 3. KMEANS CLUSTERING — 3 BEST k VALUES
# ─────────────────────────────────────────────────────────────
print("\n── K-Means Clustering ──")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#f8f9fa")

for ax, k in zip(axes, best_ks):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X_3d)
    centroids = km.cluster_centers_
    sil = silhouette_score(X_3d, cluster_labels)
    print(f"  k={k}: silhouette = {sil:.4f}")

    ax.set_facecolor("#f8f9fa")
    for ci in range(k):
        mask = cluster_labels == ci
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1],
                   c=CLUSTER_COLORS[ci % len(CLUSTER_COLORS)],
                   alpha=0.5, s=40, label=f"Cluster {ci + 1}",
                   edgecolors="white", linewidth=0.4)
    ax.scatter(centroids[:, 0], centroids[:, 1],
               c="black", marker="X", s=180, zorder=5,
               label="Centroid", edgecolors="white", linewidth=1)
    ax.set_title(f"KMeans k={k}  |  Sil: {sil:.3f}", fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

fig.suptitle("KMeans Clustering on PCA-Reduced Student Data",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("../Website/fig_cluster_kmeans.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_cluster_kmeans.png")

# ─────────────────────────────────────────────────────────────
# 4. HIERARCHICAL CLUSTERING — WARD LINKAGE
# ─────────────────────────────────────────────────────────────
print("\n── Hierarchical Clustering (Ward Linkage) ──")

# Dendrogram (use first 100 students for readability)
Z = linkage(X_3d[:100], method="ward")
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")
dendrogram(Z, ax=ax, truncate_mode="lastp", p=20,
           leaf_rotation=45, leaf_font_size=9,
           color_threshold=0.7 * max(Z[:, 2]))
ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage, 100 students)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Sample Index / Cluster", fontsize=11)
ax.set_ylabel("Distance (Ward)", fontsize=11)
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
plt.savefig("../Website/fig_cluster_dendrogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_cluster_dendrogram.png")

# Fit k=3 and scatter
hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
hc_labels = hc.fit_predict(X_3d)
hc_sil = silhouette_score(X_3d, hc_labels)
print(f"  Hierarchical k=3 silhouette: {hc_sil:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")
for ci in range(3):
    mask = hc_labels == ci
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1],
               c=CLUSTER_COLORS[ci], alpha=0.6, s=50,
               label=f"Cluster {ci + 1}", edgecolors="white", linewidth=0.4)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title(f"Hierarchical Clustering (Ward, k=3)  |  Sil: {hc_sil:.3f}",
             fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../Website/fig_cluster_hierarchical.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_cluster_hierarchical.png")

# ─────────────────────────────────────────────────────────────
# 5. DBSCAN — DENSITY BASED CLUSTERING
# ─────────────────────────────────────────────────────────────
print("\n── DBSCAN (eps=0.8, min_samples=5) ──")
dbscan = DBSCAN(eps=0.8, min_samples=5)
db_labels = dbscan.fit_predict(X_3d)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = list(db_labels).count(-1)
print(f"  Clusters found: {n_clusters}")
print(f"  Noise points  : {n_noise} ({n_noise/len(db_labels)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")
db_color_map = {-1: "#aaaaaa"}
for i, l in enumerate(sorted([x for x in set(db_labels) if x != -1])):
    db_color_map[l] = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

for label in sorted(set(db_labels)):
    mask = db_labels == label
    name = "Noise" if label == -1 else f"Cluster {label + 1}"
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1],
               c=db_color_map[label],
               alpha=0.7 if label != -1 else 0.3,
               s=50 if label != -1 else 20,
               label=name, edgecolors="white", linewidth=0.4)

ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title(
    f"DBSCAN (eps=0.8, min_samples=5)\n"
    f"{n_clusters} Clusters  |  {n_noise} Noise Points",
    fontsize=12, fontweight="bold"
)
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../Website/fig_cluster_dbscan.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_cluster_dbscan.png")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("CLUSTERING SUMMARY")
print("="*50)
print(f"  Input              : PCA 3D (19.7% variance)")
print(f"  Best KMeans k      : {best_ks}")
print(f"  Hierarchical k=3   : Sil = {hc_sil:.4f}")
print(f"  DBSCAN clusters    : {n_clusters} clusters + {n_noise} noise")
print("="*50)
print("\nScript 03 complete.")
