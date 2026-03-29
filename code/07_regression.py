"""
Student Performance Analysis
=============================
Script 07 — Logistic Regression vs. Multinomial NB — TUNED

Dataset : UCI Student Performance (student_clean.csv + G1/G2 from raw)
Task    : Binary classification — Pass (G3 >= 10) vs. Fail (G3 < 10)
Improvements:
  • Re-introduced G1/G2 as temporal features.
  • Engineered interaction features.
  • StandardScaler + L1 regularisation for LogReg.
  • Alpha tuning for MNB.
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE BINARY-LABELED DATA
# ─────────────────────────────────────────────────────────────
df  = pd.read_csv("../data/student_clean.csv")
raw = pd.read_csv("../data/student-mat.csv", sep=";")
df["G1"] = raw["G1"].values
df["G2"] = raw["G2"].values

# Binary label: Pass (1) if G3 >= 10, else Fail (0)
df["pass_fail"] = (df["G3"] >= 10).astype(int)
class_names = ["Fail", "Pass"]

y = df["pass_fail"].values
X = df.drop(columns=["performance", "G3", "pass_fail"])
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# Feature engineering
X["G1G2_avg"]        = (X["G1"] + X["G2"]) / 2
X["G2_minus_G1"]     = X["G2"] - X["G1"]
X["fail_sq"]         = X["failures"] ** 2
X["fail_x_absences"] = X["failures"] * X["absences"]
X["parent_edu"]      = X["Medu"] + X["Fedu"]
X["alc_total"]       = X["Dalc"] + X["Walc"]
X["G1_x_study"]      = X["G1"] * X["studytime"]

print(f"Features shape: {X.shape}")
print(f"Class distribution:\n  Pass: {(y==1).sum()}, Fail: {(y==0).sum()}")

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
    colors=["#f39c12", "#e74c3c"],
    autopct="%1.1f%%", startangle=90,
    textprops={"fontsize": 11, "fontweight": "bold"}
)
axes[0].set_title("Train / Test Split (80/20)", fontsize=13, fontweight="bold")

for idx, (data, title) in enumerate([
    (y_train, "Training Set"), (y_test, "Testing Set")
]):
    counts = [np.sum(data == 0), np.sum(data == 1)]
    axes[idx + 1].bar(class_names, counts, color=["#e74c3c", "#2ecc71"])
    axes[idx + 1].set_title(f"{title} Class Distribution", fontsize=13, fontweight="bold")
    axes[idx + 1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("../Website/fig_reg_train_test_split.png", dpi=150)
plt.close()
print("Saved: fig_reg_train_test_split.png")

# ─────────────────────────────────────────────────────────────
# 3. SCALE DATA
# ─────────────────────────────────────────────────────────────
# StandardScaler for Logistic Regression (best with regularisation)
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std  = std_scaler.transform(X_test)

# MinMaxScaler for Multinomial NB (requires non-negative)
mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train)
X_test_mm  = mm_scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# 4. LOGISTIC REGRESSION  (tuned: C=0.1, L1, StandardScaler)
# ─────────────────────────────────────────────────────────────
lr = LogisticRegression(
    C=0.1, penalty="l1", solver="saga",
    max_iter=5000, random_state=42
)
lr.fit(X_train_std, y_train)
y_pred_lr = lr.predict(X_test_std)
acc_lr    = accuracy_score(y_test, y_pred_lr)
cm_lr     = confusion_matrix(y_test, y_pred_lr)

print(f"\n{'='*50}")
print(f"Logistic Regression — Accuracy: {acc_lr:.4f} ({acc_lr*100:.1f}%)")
print(f"{'='*50}")
print(classification_report(y_test, y_pred_lr, target_names=class_names))

# ─────────────────────────────────────────────────────────────
# 5. MULTINOMIAL NAÏVE BAYES  (tuned alpha)
# ─────────────────────────────────────────────────────────────
mnb = MultinomialNB(alpha=0.01)
mnb.fit(X_train_mm, y_train)
y_pred_mnb = mnb.predict(X_test_mm)
acc_mnb    = accuracy_score(y_test, y_pred_mnb)
cm_mnb     = confusion_matrix(y_test, y_pred_mnb)

print(f"\n{'='*50}")
print(f"Multinomial Naïve Bayes — Accuracy: {acc_mnb:.4f} ({acc_mnb*100:.1f}%)")
print(f"{'='*50}")
print(classification_report(y_test, y_pred_mnb, target_names=class_names))

# ─────────────────────────────────────────────────────────────
# 6. CONFUSION MATRIX — LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm_lr, annot=True, fmt="d", cmap="Oranges",
    xticklabels=class_names, yticklabels=class_names, ax=ax,
    linewidths=1, linecolor="white",
    annot_kws={"size": 16, "fontweight": "bold"}
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Logistic Regression — Confusion Matrix\nAccuracy: {acc_lr*100:.1f}%",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../Website/fig_reg_cm_logistic.png", dpi=150)
plt.close()
print("Saved: fig_reg_cm_logistic.png")

# ─────────────────────────────────────────────────────────────
# 7. CONFUSION MATRIX — MULTINOMIAL NB
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm_mnb, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names, ax=ax,
    linewidths=1, linecolor="white",
    annot_kws={"size": 16, "fontweight": "bold"}
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Multinomial Naïve Bayes — Confusion Matrix\nAccuracy: {acc_mnb*100:.1f}%",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../Website/fig_reg_cm_mnb.png", dpi=150)
plt.close()
print("Saved: fig_reg_cm_mnb.png")

# ─────────────────────────────────────────────────────────────
# 8. ACCURACY COMPARISON
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
model_names = ["Logistic Regression", "Multinomial NB"]
accuracies  = [acc_lr * 100, acc_mnb * 100]
colors      = ["#f39c12", "#3498db"]
bars = ax.bar(model_names, accuracies, color=colors, edgecolor="white", linewidth=2)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{acc:.1f}%", ha="center", va="bottom",
            fontweight="bold", fontsize=14)

ax.set_ylim(0, max(accuracies) + 12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Logistic Regression vs. Multinomial NB\n(Binary: Pass / Fail — with feature engineering)",
             fontsize=14, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("../Website/fig_reg_comparison.png", dpi=150)
plt.close()
print("Saved: fig_reg_comparison.png")

# ─────────────────────────────────────────────────────────────
# 9. SIGMOID FUNCTION ILLUSTRATION
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
z = np.linspace(-7, 7, 300)
sigmoid = 1 / (1 + np.exp(-z))
ax.plot(z, sigmoid, color="#f39c12", linewidth=3, label="σ(z) = 1 / (1 + e⁻ᶻ)")
ax.axhline(y=0.5, color="#e74c3c", linestyle="--", alpha=0.6, label="Threshold = 0.5")
ax.axvline(x=0, color="gray", linestyle=":", alpha=0.4)
ax.fill_between(z, sigmoid, 0.5, where=(sigmoid > 0.5),
                alpha=0.1, color="#2ecc71", label="Predict: Pass")
ax.fill_between(z, sigmoid, 0.5, where=(sigmoid < 0.5),
                alpha=0.1, color="#e74c3c", label="Predict: Fail")
ax.set_xlabel("z = β₀ + β₁x₁ + β₂x₂ + ...", fontsize=12)
ax.set_ylabel("σ(z) = P(Pass)", fontsize=12)
ax.set_title("The Sigmoid Function in Logistic Regression", fontsize=14, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("../Website/fig_reg_sigmoid.png", dpi=150)
plt.close()
print("Saved: fig_reg_sigmoid.png")

print(f"\n\nFINAL COMPARISON:")
print(f"  Logistic Regression: {acc_lr*100:.1f}%")
print(f"  Multinomial NB:      {acc_mnb*100:.1f}%")
winner = "Logistic Regression" if acc_lr > acc_mnb else "Multinomial NB"
print(f"  Winner: {winner}")

print("\nScript 07 complete.")
