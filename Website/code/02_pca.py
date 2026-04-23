"""
Student Performance Analysis
=============================
Script 02 — Principal Component Analysis (PCA)

Goal    : Reduce dimensionality of student data, visualize in 2D/3D,
          understand variance retention, and extract top eigenvalues.
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
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA FOR PCA
#    - PCA requires: numeric-only, no labels, normalized
# ─────────────────────────────────────────────────────────────
raw_df = pd.read_csv("../data/student-mat.csv", sep=";")

# Encode binary yes/no columns
binary_cols = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]
for col in binary_cols:
    raw_df[col] = LabelEncoder().fit_transform(raw_df[col])

# One-hot encode multi-category columns
raw_df = pd.get_dummies(raw_df, columns=["Mjob", "Fjob", "reason", "guardian"])

# Create performance label (kept separately — NOT used in PCA)
def perf_label(g):
    if g >= 14: return "High"
    elif g >= 10: return "Medium"
    else: return "Low"

labels = raw_df["G3"].apply(perf_label)

# Drop targets + labels — PCA input is features ONLY
pca_features = raw_df.drop(columns=["G3", "G1", "G2"])

print(f"PCA input shape: {pca_features.shape}")
print(f"  → {pca_features.shape[1]} features, {pca_features.shape[0]} students")

# ─────────────────────────────────────────────────────────────
# 2. NORMALIZE — StandardScaler (mean=0, std=1 per feature)
# ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pca_features)

print("\nNormalized data sample (first 3 rows, first 5 cols):")
print(pd.DataFrame(X_scaled, columns=pca_features.columns).head(3).iloc[:, :5].round(3))

# ─────────────────────────────────────────────────────────────
# 3. PCA — 2 COMPONENTS
# ─────────────────────────────────────────────────────────────
pca2 = PCA(n_components=2)
X_pca2 = pca2.fit_transform(X_scaled)
var2 = pca2.explained_variance_ratio_

print(f"\n── PCA n=2 ──")
print(f"  PC1 variance: {var2[0]*100:.1f}%")
print(f"  PC2 variance: {var2[1]*100:.1f}%")
print(f"  Total retained: {var2.sum()*100:.1f}%")

# Visualize 2D
color_map = {"High": "#4CAF50", "Medium": "#2196F3", "Low": "#F44336"}
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")
for label, color in color_map.items():
    mask = (labels == label).values
    ax.scatter(X_pca2[mask, 0], X_pca2[mask, 1],
               c=color, label=label, alpha=0.7, s=50,
               edgecolors="white", linewidth=0.5)
ax.set_xlabel(f"PC1 ({var2[0]*100:.1f}% variance)", fontsize=11)
ax.set_ylabel(f"PC2 ({var2[1]*100:.1f}% variance)", fontsize=11)
ax.set_title(f"PCA — 2D Projection  |  Retained: {var2.sum()*100:.1f}%",
             fontsize=13, fontweight="bold")
ax.legend(title="Performance")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../Website/fig_pca_2d.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_pca_2d.png")

# ─────────────────────────────────────────────────────────────
# 4. PCA — 3 COMPONENTS
# ─────────────────────────────────────────────────────────────
pca3 = PCA(n_components=3)
X_pca3 = pca3.fit_transform(X_scaled)
var3 = pca3.explained_variance_ratio_

print(f"\n── PCA n=3 ──")
print(f"  PC1: {var3[0]*100:.1f}%  PC2: {var3[1]*100:.1f}%  PC3: {var3[2]*100:.1f}%")
print(f"  Total retained: {var3.sum()*100:.1f}%")

# Visualize 3D as two 2D cross-sections
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#f8f9fa")
proj_pairs = [(0, 1, "PC1", "PC2"), (0, 2, "PC1", "PC3")]
for ax, (xi, yi, xl, yl) in zip(axes, proj_pairs):
    ax.set_facecolor("#f8f9fa")
    for label, color in color_map.items():
        mask = (labels == label).values
        ax.scatter(X_pca3[mask, xi], X_pca3[mask, yi],
                   c=color, label=label, alpha=0.7, s=40, edgecolors="w")
    ax.set_xlabel(f"{xl} ({var3[xi]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"{yl} ({var3[yi]*100:.1f}%)", fontsize=10)
    ax.legend(title="Performance", fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle(f"PCA — 3D Projection (Two Views)  |  Retained: {var3.sum()*100:.1f}%",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("../Website/fig_pca_3d.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_pca_3d.png")

# ─────────────────────────────────────────────────────────────
# 5. FULL PCA — CUMULATIVE VARIANCE & 95% THRESHOLD
# ─────────────────────────────────────────────────────────────
pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
n_components_95 = int(np.argmax(cumvar >= 95)) + 1

print(f"\n── Cumulative Variance ──")
print(f"  Components needed for 95% variance: {n_components_95}")

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")
ax.plot(range(1, len(cumvar) + 1), cumvar, color="#2c3e50", linewidth=2, marker="o", ms=3)
ax.axhline(95, color="#e74c3c", linestyle="--", linewidth=1.5, label="95% threshold")
ax.axvline(n_components_95, color="#e74c3c", linestyle=":", linewidth=1.5)
ax.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.12, color="#2c3e50")
ax.annotate(f"{n_components_95} components = 95%",
            xy=(n_components_95, 95), xytext=(n_components_95 + 3, 78),
            fontsize=9, color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c"))
ax.set_xlabel("Number of Principal Components", fontsize=11)
ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=11)
ax.set_title("Cumulative Variance Explained by PCA", fontsize=13, fontweight="bold")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig("../Website/fig_pca_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_pca_variance.png")

# ─────────────────────────────────────────────────────────────
# 6. EIGENVALUES — TOP 10
# ─────────────────────────────────────────────────────────────
eigenvalues = pca_full.explained_variance_
top3 = eigenvalues[:3]
print(f"\n── Top 3 Eigenvalues ──")
print(f"  PC1: {top3[0]:.4f}")
print(f"  PC2: {top3[1]:.4f}")
print(f"  PC3: {top3[2]:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")
colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#1abc9c",
          "#3498db", "#9b59b6", "#34495e", "#95a5a6", "#7f8c8d"]
bars = ax.bar(range(1, 11), eigenvalues[:10], color=colors)
for bar, ev in zip(bars, eigenvalues[:10]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f"{ev:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xlabel("Principal Component", fontsize=11)
ax.set_ylabel("Eigenvalue", fontsize=11)
ax.set_title("Top 10 Eigenvalues from PCA", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("../Website/fig_pca_eigenvalues.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_pca_eigenvalues.png")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("PCA SUMMARY")
print("="*50)
print(f"  Input features        : {pca_features.shape[1]}")
print(f"  Variance in 2D        : {var2.sum()*100:.1f}%")
print(f"  Variance in 3D        : {var3.sum()*100:.1f}%")
print(f"  Components for 95%    : {n_components_95}")
print(f"  Top eigenvalues       : {top3[0]:.3f}, {top3[1]:.3f}, {top3[2]:.3f}")
print("="*50)
print("\nScript 02 complete.")
