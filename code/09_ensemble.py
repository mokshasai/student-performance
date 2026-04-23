"""
09_ensemble.py — Random Forest Ensemble Learning Classification
Student Performance Analysis Project
Module 4 Assignment

Trains a Random Forest classifier on the student dataset and
generates all figures for the Ensemble tab of the website.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# ── Chalkboard style ────────────────────────────────────────────
BOARD      = '#2c3e2d'
BOARD_LT   = '#354a36'
CHALK      = '#e8e4d9'
CHALK_DIM  = '#b5b0a0'
CHALK_YEL  = '#f0d87a'
CHALK_BLUE = '#7ec8e3'
CHALK_PINK = '#e8a0b0'
CHALK_GRN  = '#8bc99a'
CHALK_ORG  = '#e8b86d'
CHALK_PURP = '#b39ddb'

plt.rcParams.update({
    'figure.facecolor': BOARD,
    'axes.facecolor':   BOARD_LT,
    'axes.edgecolor':   CHALK_DIM,
    'axes.labelcolor':  CHALK,
    'xtick.color':      CHALK_DIM,
    'ytick.color':      CHALK_DIM,
    'text.color':       CHALK,
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'grid.color':       '#354a36',
    'grid.alpha':       0.4,
})

OUT = 'Website/'

# ── 1. Load & prepare data ──────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/student_clean.csv')

LABEL_COL = 'performance'
DROP_COLS  = [c for c in ['G3', 'performance', 'G1', 'G2'] if c in df.columns]

X = df.drop(columns=DROP_COLS)
y = df[LABEL_COL]

for col in X.select_dtypes(include='bool').columns:
    X[col] = X[col].astype(int)

class_order = ['Low', 'Medium', 'High']
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print(f"Class distribution:\n{y.value_counts()}")

# ── 2. Train/Test Split ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── 3. Train Random Forest ──────────────────────────────────────
print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred, labels=class_order)

print(f"Random Forest Accuracy: {acc*100:.1f}%")
print(classification_report(y_test, y_pred, target_names=class_order))

# ── 4. Figure: Confusion Matrix ─────────────────────────────────
from matplotlib.colors import LinearSegmentedColormap

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(BOARD)
ax.set_facecolor(BOARD_LT)

cmap = LinearSegmentedColormap.from_list('chalk', [BOARD_LT, CHALK_GRN], N=256)
sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
            xticklabels=class_order, yticklabels=class_order,
            ax=ax, linewidths=0.5, linecolor=CHALK_DIM,
            annot_kws={'size': 14, 'color': CHALK, 'weight': 'bold'},
            cbar_kws={'shrink': 0.8})

ax.set_title(f'Random Forest — Confusion Matrix\nAccuracy: {acc*100:.1f}%',
             color=CHALK_YEL, fontsize=13, pad=10)
ax.set_xlabel('Predicted Class', color=CHALK_DIM, fontsize=11)
ax.set_ylabel('True Class', color=CHALK_DIM, fontsize=11)
ax.tick_params(colors=CHALK_DIM)

cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color=CHALK_DIM)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=CHALK_DIM)

plt.tight_layout()
plt.savefig(f'{OUT}fig_ens_cm.png', dpi=130, bbox_inches='tight', facecolor=BOARD)
plt.close()
print("Saved: fig_ens_cm.png")

# ── 5. Figure: Feature Importance ───────────────────────────────
importances = pd.Series(rf.feature_importances_, index=X.columns)
top15 = importances.nlargest(15).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BOARD)
ax.set_facecolor(BOARD_LT)

colors = [CHALK_GRN if v > top15.median() else CHALK_BLUE for v in top15.values]
bars = ax.barh(top15.index, top15.values,
               color=colors, edgecolor=CHALK_DIM, linewidth=0.8, height=0.65)

for bar, val in zip(bars, top15.values):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left',
            color=CHALK_DIM, fontsize=9)

ax.set_title('Random Forest — Top 15 Feature Importances',
             color=CHALK, fontsize=13, pad=12)
ax.set_xlabel('Importance Score (Mean Decrease in Impurity)', color=CHALK_DIM)
ax.set_ylabel('Feature', color=CHALK_DIM)
ax.grid(axis='x', linestyle='--', alpha=0.3, color=CHALK_DIM)
for spine in ax.spines.values():
    spine.set_edgecolor(CHALK_DIM)
    spine.set_linewidth(0.5)

green_patch = mpatches.Patch(color=CHALK_GRN, label='Above Median Importance')
blue_patch  = mpatches.Patch(color=CHALK_BLUE, label='Below Median Importance')
ax.legend(handles=[green_patch, blue_patch],
          facecolor=BOARD, edgecolor=CHALK_DIM, labelcolor=CHALK,
          fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig(f'{OUT}fig_ens_feature_importance.png', dpi=130, bbox_inches='tight',
            facecolor=BOARD)
plt.close()
print("Saved: fig_ens_feature_importance.png")

# ── 6. Figure: Learning Curve ────────────────────────────────────
print("Computing learning curve (this may take a moment)...")
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                           class_weight='balanced'),
    X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1) * 100
train_std  = train_scores.std(axis=1)  * 100
val_mean   = val_scores.mean(axis=1)   * 100
val_std    = val_scores.std(axis=1)    * 100

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor(BOARD)
ax.set_facecolor(BOARD_LT)

ax.plot(train_sizes, train_mean, 'o-', color=CHALK_GRN, linewidth=2.2,
        markersize=7, label='Training Score')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.12, color=CHALK_GRN)

ax.plot(train_sizes, val_mean, 'o-', color=CHALK_BLUE, linewidth=2.2,
        markersize=7, label='Cross-Validation Score')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                alpha=0.12, color=CHALK_BLUE)

ax.set_title('Random Forest — Learning Curve (5-Fold CV)',
             color=CHALK, fontsize=13, pad=12)
ax.set_xlabel('Training Set Size (number of samples)', color=CHALK_DIM)
ax.set_ylabel('Accuracy (%)', color=CHALK_DIM)
ax.legend(facecolor=BOARD, edgecolor=CHALK_DIM, labelcolor=CHALK, fontsize=11)
ax.set_ylim(0, 110)
ax.grid(True, linestyle='--', alpha=0.3, color=CHALK_DIM)
for spine in ax.spines.values():
    spine.set_edgecolor(CHALK_DIM)
    spine.set_linewidth(0.5)

plt.tight_layout()
plt.savefig(f'{OUT}fig_ens_learning_curve.png', dpi=130, bbox_inches='tight',
            facecolor=BOARD)
plt.close()
print("Saved: fig_ens_learning_curve.png")

# ── 7. Figure: Model Comparison (RF vs DT vs NB) ────────────────
# Approximate accuracies from prior tabs
model_names = ['Naïve Bayes\n(Gaussian)', 'Decision Tree\n(Gini)', 'SVM\n(Best)', 'Random Forest']
model_accs  = [81.0, 89.9, None, round(acc * 100, 1)]

# We'll compute SVM reference here too for comparison
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler as SS
sc2 = SS()
Xtr_sc = sc2.fit_transform(X_train)
Xte_sc = sc2.transform(X_test)
svm_ref = SVC(kernel='rbf', C=10, gamma='scale', decision_function_shape='ovr', random_state=42)
svm_ref.fit(Xtr_sc, y_train)
svm_acc = accuracy_score(y_test, svm_ref.predict(Xte_sc)) * 100
model_accs[2] = round(svm_acc, 1)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BOARD)
ax.set_facecolor(BOARD_LT)

bar_colors = [CHALK_BLUE, CHALK_GRN, CHALK_PINK, CHALK_YEL]
bars = ax.bar(model_names, model_accs,
              color=bar_colors, edgecolor=CHALK_DIM, linewidth=1.2, width=0.5)

for bar, val in zip(bars, model_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val}%', ha='center', va='bottom',
            color=CHALK, fontsize=12, fontweight='bold')

ax.set_ylim(0, 110)
ax.set_ylabel('Test Accuracy (%)', color=CHALK_DIM)
ax.set_title('Model Comparison — Accuracy Across ML Methods',
             color=CHALK, fontsize=13, pad=12)
ax.grid(axis='y', linestyle='--', alpha=0.3, color=CHALK_DIM)
for spine in ax.spines.values():
    spine.set_edgecolor(CHALK_DIM)
    spine.set_linewidth(0.5)

plt.tight_layout()
plt.savefig(f'{OUT}fig_ens_accuracy_comparison.png', dpi=130, bbox_inches='tight',
            facecolor=BOARD)
plt.close()
print("Saved: fig_ens_accuracy_comparison.png")

# ── 8. Hyperparameter exploration: n_estimators ─────────────────
n_trees_range = [10, 25, 50, 100, 150, 200, 300]
tree_accs = []
for n in n_trees_range:
    rf_tmp = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1,
                                    class_weight='balanced')
    rf_tmp.fit(X_train, y_train)
    tree_accs.append(accuracy_score(y_test, rf_tmp.predict(X_test)) * 100)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BOARD)
ax.set_facecolor(BOARD_LT)

ax.plot(n_trees_range, tree_accs, 'o-', color=CHALK_YEL, linewidth=2.2,
        markersize=8, markerfacecolor=CHALK)

best_n_idx = np.argmax(tree_accs)
ax.plot(n_trees_range[best_n_idx], tree_accs[best_n_idx],
        'o', color=CHALK_GRN, markersize=13, zorder=5,
        label=f'Best: {n_trees_range[best_n_idx]} trees ({tree_accs[best_n_idx]:.1f}%)')

ax.set_title('Random Forest — Accuracy vs Number of Trees (n_estimators)',
             color=CHALK, fontsize=13, pad=12)
ax.set_xlabel('Number of Decision Trees', color=CHALK_DIM)
ax.set_ylabel('Test Accuracy (%)', color=CHALK_DIM)
ax.set_ylim(0, 110)
ax.legend(facecolor=BOARD, edgecolor=CHALK_DIM, labelcolor=CHALK, fontsize=11)
ax.grid(True, linestyle='--', alpha=0.3, color=CHALK_DIM)
for spine in ax.spines.values():
    spine.set_edgecolor(CHALK_DIM)
    spine.set_linewidth(0.5)

plt.tight_layout()
plt.savefig(f'{OUT}fig_ens_ntrees.png', dpi=130, bbox_inches='tight', facecolor=BOARD)
plt.close()
print("Saved: fig_ens_ntrees.png")

# ── Summary ─────────────────────────────────────────────────────
print("\n" + "="*50)
print("RANDOM FOREST RESULTS SUMMARY")
print("="*50)
print(f"n_estimators : 200")
print(f"Test Accuracy: {acc*100:.1f}%")
print(f"Top 3 features: {', '.join(importances.nlargest(3).index.tolist())}")
print("="*50)
print("\nAll Ensemble figures generated successfully!")
