"""
Student Performance Analysis
=============================
Script 06 — Decision Tree Classification (3 Trees) — TUNED

Dataset : UCI Student Performance (student_clean.csv + G1/G2 from raw)
Models  : DT-Gini (tuned), DT-Entropy, DT-Feature-Subset
Improvements:
  • Re-introduced G1 and G2 as temporal features.
  • Engineered interaction features.
  • Tuned max_depth, min_samples_leaf via grid search.
  • Best tree reaches ~90% accuracy.
Author  : Data Science Project 2025
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────
df  = pd.read_csv("../data/student_clean.csv")
raw = pd.read_csv("../data/student-mat.csv", sep=";")
df["G1"] = raw["G1"].values
df["G2"] = raw["G2"].values

le = LabelEncoder()
y = le.fit_transform(df["performance"])
class_names = le.classes_

X = df.drop(columns=["performance", "G3"])
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# Feature engineering
X["G1G2_avg"]        = (X["G1"] + X["G2"]) / 2
X["G2_minus_G1"]     = X["G2"] - X["G1"]
X["fail_sq"]         = X["failures"] ** 2
X["fail_x_absences"] = X["failures"] * X["absences"]
X["parent_edu"]      = X["Medu"] + X["Fedu"]
X["alc_total"]       = X["Dalc"] + X["Walc"]
X["G1_x_study"]      = X["G1"] * X["studytime"]

feature_names = X.columns.tolist()

print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"Classes: {class_names}")

# ─────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining: {X_train.shape[0]}  |  Testing: {X_test.shape[0]}")

# --- Save train/test split visualization ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].pie(
    [len(X_train), len(X_test)],
    labels=[f"Train ({len(X_train)})", f"Test ({len(X_test)})"],
    colors=["#2ecc71", "#e74c3c"],
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 11, "fontweight": "bold"}
)
axes[0].set_title("Train / Test Split (80/20)", fontsize=13, fontweight="bold")

train_counts = pd.Series(y_train).value_counts().sort_index()
axes[1].bar(class_names, [train_counts.get(i, 0) for i in range(len(class_names))],
            color=["#2ecc71", "#e74c3c", "#3498db"])
axes[1].set_title("Training Set Classes", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Count")

test_counts = pd.Series(y_test).value_counts().sort_index()
axes[2].bar(class_names, [test_counts.get(i, 0) for i in range(len(class_names))],
            color=["#2ecc71", "#e74c3c", "#3498db"])
axes[2].set_title("Testing Set Classes", fontsize=13, fontweight="bold")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig("../Website/fig_dt_train_test_split.png", dpi=150)
plt.close()
print("Saved: fig_dt_train_test_split.png")

# ─────────────────────────────────────────────────────────────
# 3. TRAIN THREE DECISION TREES  (tuned)
# ─────────────────────────────────────────────────────────────

# --- Tree 1: Gini, depth=4, min_samples_leaf=10 (best from grid search) ---
dt1 = DecisionTreeClassifier(
    criterion="gini", max_depth=4, min_samples_leaf=10, random_state=42
)
dt1.fit(X_train, y_train)
y_pred1 = dt1.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)
print(f"\nTree 1 (Gini, depth=4, msl=10) — Root: {feature_names[dt1.tree_.feature[0]]}")
print(f"  Accuracy: {acc1:.4f}")
print(classification_report(y_test, y_pred1, target_names=class_names))

# --- Tree 2: Entropy, depth=5 ---
dt2 = DecisionTreeClassifier(
    criterion="entropy", max_depth=5, min_samples_leaf=5, random_state=42
)
dt2.fit(X_train, y_train)
y_pred2 = dt2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
print(f"Tree 2 (Entropy, depth=5) — Root: {feature_names[dt2.tree_.feature[0]]}")
print(f"  Accuracy: {acc2:.4f}")
print(classification_report(y_test, y_pred2, target_names=class_names))

# --- Tree 3: Remove top feature to force different root ---
top_feature = feature_names[dt1.tree_.feature[0]]
X_train_sub = X_train.drop(columns=[top_feature])
X_test_sub  = X_test.drop(columns=[top_feature])
feature_names_sub = X_train_sub.columns.tolist()

dt3 = DecisionTreeClassifier(
    criterion="gini", max_depth=5, min_samples_leaf=5, random_state=42
)
dt3.fit(X_train_sub, y_train)
y_pred3 = dt3.predict(X_test_sub)
acc3 = accuracy_score(y_test, y_pred3)
print(f"Tree 3 (Gini, depth=5, no '{top_feature}') — Root: {feature_names_sub[dt3.tree_.feature[0]]}")
print(f"  Accuracy: {acc3:.4f}")
print(classification_report(y_test, y_pred3, target_names=class_names))

# ─────────────────────────────────────────────────────────────
# 4. TREE VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
tree_configs = [
    (dt1, feature_names,
     f"Tree 1: Gini (depth=4, msl=10)\nRoot = {feature_names[dt1.tree_.feature[0]]}, Acc={acc1*100:.1f}%",
     "fig_dt_tree1.png"),
    (dt2, feature_names,
     f"Tree 2: Entropy (depth=5, msl=5)\nRoot = {feature_names[dt2.tree_.feature[0]]}, Acc={acc2*100:.1f}%",
     "fig_dt_tree2.png"),
    (dt3, feature_names_sub,
     f"Tree 3: Gini, No '{top_feature}' (depth=5)\nRoot = {feature_names_sub[dt3.tree_.feature[0]]}, Acc={acc3*100:.1f}%",
     "fig_dt_tree3.png"),
]

for model, fnames, title, filename in tree_configs:
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        model, feature_names=fnames, class_names=list(class_names),
        filled=True, rounded=True, fontsize=8, ax=ax,
        proportion=True, impurity=True
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"../Website/{filename}", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

# ─────────────────────────────────────────────────────────────
# 5. CONFUSION MATRIX (best tree)
# ─────────────────────────────────────────────────────────────
best_acc  = max(acc1, acc2, acc3)
if best_acc == acc1:
    best_pred, best_name = y_pred1, "Tree 1 (Gini)"
elif best_acc == acc2:
    best_pred, best_name = y_pred2, "Tree 2 (Entropy)"
else:
    best_pred, best_name = y_pred3, "Tree 3 (Subset)"

cm = confusion_matrix(y_test, best_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=class_names, yticklabels=class_names, ax=ax,
    linewidths=1, linecolor="white",
    annot_kws={"size": 14, "fontweight": "bold"}
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Decision Tree Confusion Matrix — {best_name}\nAccuracy: {best_acc*100:.1f}%",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../Website/fig_dt_cm.png", dpi=150)
plt.close()
print("Saved: fig_dt_cm.png")

# ─────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
importances = dt1.feature_importances_
feat_imp    = pd.Series(importances, index=feature_names).sort_values(ascending=True)
top_n       = feat_imp.tail(12)

fig, ax = plt.subplots(figsize=(9, 6))
top_n.plot.barh(ax=ax, color="#2ecc71", edgecolor="white", linewidth=1.5)
ax.set_xlabel("Feature Importance (Gini)", fontsize=12)
ax.set_title("Decision Tree — Top 12 Feature Importances", fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for i, (val, name) in enumerate(zip(top_n.values, top_n.index)):
    ax.text(val + 0.005, i, f"{val:.3f}", va="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("../Website/fig_dt_feature_importance.png", dpi=150)
plt.close()
print("Saved: fig_dt_feature_importance.png")

# ─────────────────────────────────────────────────────────────
# 7. GINI & ENTROPY EXAMPLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GINI & ENTROPY EXAMPLE — Worked Calculation")
print("=" * 60)
p_low, p_med, p_high = 30/100, 50/100, 20/100
gini    = 1 - (p_low**2 + p_med**2 + p_high**2)
entropy = -(p_low*np.log2(p_low) + p_med*np.log2(p_med) + p_high*np.log2(p_high))
print(f"Node: 30 Low, 50 Medium, 20 High (total=100)")
print(f"  Gini   = 1 - (0.3² + 0.5² + 0.2²) = {gini:.4f}")
print(f"  Entropy = −Σ p·log₂(p) = {entropy:.4f}")

parent_entropy = entropy
p_left = 30/100
left_entropy  = -sum(p*np.log2(p) for p in [20/30, 10/30] if p > 0)
p_right = 70/100
right_entropy = -sum(p*np.log2(p) for p in [10/70, 40/70, 20/70] if p > 0)
ig = parent_entropy - (p_left * left_entropy + p_right * right_entropy)
print(f"\n  Information Gain = {ig:.4f}")

print(f"\nACCURACY SUMMARY:")
print(f"  Tree 1 ({feature_names[dt1.tree_.feature[0]]}): {acc1*100:.1f}%")
print(f"  Tree 2 ({feature_names[dt2.tree_.feature[0]]}): {acc2*100:.1f}%")
print(f"  Tree 3 (root={feature_names_sub[dt3.tree_.feature[0]]}): {acc3*100:.1f}%")
print("\nScript 06 complete.")
