"""
Student Performance Analysis
=============================
Script 05 — Naïve Bayes Classification (3 Flavors) — TUNED

Dataset : UCI Student Performance (student_clean.csv + G1/G2 from raw)
Models  : Multinomial NB, Gaussian NB (tuned), Bernoulli NB (tuned)
Improvements:
  • Re-introduced G1 (period-1 grade) and G2 (period-2 grade) as
    legitimate temporal features available before the final exam.
  • Engineered interaction features (G1G2_avg, G2−G1 trend, etc.)
  • Tuned var_smoothing for GNB, alpha/binarize for BNB, alpha for MNB.
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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA  (re-introduce G1, G2 + feature eng.)
# ─────────────────────────────────────────────────────────────
df   = pd.read_csv("../data/student_clean.csv")
raw  = pd.read_csv("../data/student-mat.csv", sep=";")

# Add G1 and G2 back — they are temporally prior to G3
df["G1"] = raw["G1"].values
df["G2"] = raw["G2"].values

# Target
le = LabelEncoder()
y = le.fit_transform(df["performance"])          # High=0, Low=1, Medium=2
class_names = le.classes_

# Features: drop only target columns, keep G1 & G2
X = df.drop(columns=["performance", "G3"])
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# ── Feature Engineering ───────────────────────────────────────
X["G1G2_avg"]       = (X["G1"] + X["G2"]) / 2
X["G2_minus_G1"]    = X["G2"] - X["G1"]          # grade trend
X["fail_sq"]        = X["failures"] ** 2
X["fail_x_absences"] = X["failures"] * X["absences"]
X["parent_edu"]     = X["Medu"] + X["Fedu"]
X["alc_total"]      = X["Dalc"] + X["Walc"]
X["G1_x_study"]     = X["G1"] * X["studytime"]

print(f"Features shape: {X.shape}")
print(f"Class distribution:\n{pd.Series(y).value_counts()}")
print(f"Classes: {class_names}")

# ─────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT  (80/20 stratified)
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing  set: {X_test.shape[0]} samples")

# --- Save train/test split visualization ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].pie(
    [len(X_train), len(X_test)],
    labels=[f"Train ({len(X_train)})", f"Test ({len(X_test)})"],
    colors=["#3498db", "#e74c3c"],
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 11, "fontweight": "bold"}
)
axes[0].set_title("Train / Test Split (80/20)", fontsize=13, fontweight="bold")

train_counts = pd.Series(y_train).value_counts().sort_index()
axes[1].bar(class_names, [train_counts.get(i, 0) for i in range(len(class_names))],
            color=["#2ecc71", "#e74c3c", "#3498db"])
axes[1].set_title("Training Set Class Distribution", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Count")

test_counts = pd.Series(y_test).value_counts().sort_index()
axes[2].bar(class_names, [test_counts.get(i, 0) for i in range(len(class_names))],
            color=["#2ecc71", "#e74c3c", "#3498db"])
axes[2].set_title("Testing Set Class Distribution", fontsize=13, fontweight="bold")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig("../Website/fig_nb_train_test_split.png", dpi=150)
plt.close()
print("Saved: fig_nb_train_test_split.png")

# ─────────────────────────────────────────────────────────────
# 3. SCALE DATA
# ─────────────────────────────────────────────────────────────
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# 4. TRAIN ALL THREE NB MODELS  (tuned hyper-parameters)
# ─────────────────────────────────────────────────────────────
models = {
    "Multinomial NB": MultinomialNB(alpha=10.0),
    "Gaussian NB":    GaussianNB(var_smoothing=0.01),
    "Bernoulli NB":   BernoulliNB(alpha=10.0, binarize=0.0),
}

results = {}

for name, model in models.items():
    if name == "Multinomial NB":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif name == "Gaussian NB":
        # GNB works best on raw (unscaled) features here
        model.fit(X_train.values, y_train)
        y_pred = model.predict(X_test.values)
    else:
        model.fit(X_train.values, y_train)
        y_pred = model.predict(X_test.values)

    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    results[name] = {"accuracy": acc, "cm": cm, "report": report, "y_pred": y_pred}

    print(f"\n{'='*50}")
    print(f"{name}  —  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"{'='*50}")
    print(report)

# ─────────────────────────────────────────────────────────────
# 5. CONFUSION MATRIX VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
cm_files = {
    "Multinomial NB": "fig_nb_cm_multinomial.png",
    "Gaussian NB":    "fig_nb_cm_gaussian.png",
    "Bernoulli NB":   "fig_nb_cm_bernoulli.png",
}

for name, filename in cm_files.items():
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        results[name]["cm"], annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        linewidths=1, linecolor="white",
        annot_kws={"size": 14, "fontweight": "bold"}
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"{name} — Confusion Matrix\nAccuracy: {results[name]['accuracy']*100:.1f}%",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"../Website/{filename}", dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# ─────────────────────────────────────────────────────────────
# 6. ACCURACY COMPARISON BAR CHART
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
names = list(results.keys())
accs  = [results[n]["accuracy"] * 100 for n in names]
colors = ["#e74c3c", "#3498db", "#2ecc71"]
bars  = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=2)

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{acc:.1f}%", ha="center", va="bottom",
            fontweight="bold", fontsize=13)

ax.set_ylim(0, max(accs) + 12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Naïve Bayes — Accuracy Comparison (3 Flavors)\n(with feature engineering & hyper-parameter tuning)",
             fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("../Website/fig_nb_accuracy_comparison.png", dpi=150)
plt.close()
print("Saved: fig_nb_accuracy_comparison.png")

# ─────────────────────────────────────────────────────────────
# 7. PRINT DATA PREVIEWS
# ─────────────────────────────────────────────────────────────
print("\n--- Feature Matrix (first 5 rows) ---")
print(X.head())
print(f"\nTarget labels (first 20): {y[:20]}")
print(f"Train indices (first 10): {X_train.index[:10].tolist()}")
print(f"Test  indices (first 10): {X_test.index[:10].tolist()}")

print("\nScript 05 complete.")
