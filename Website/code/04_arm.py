"""
Student Performance Analysis
=============================
Script 04 — Association Rule Mining (ARM)

Algorithm : Apriori (via mlxtend)
Goal      : Discover if-then rules linking student behaviors to grades.
            Report top 15 rules by support, confidence, and lift.
Dataset   : UCI Student Performance (student-mat.csv)
Author    : Data Science Project 2025

Install   : pip install mlxtend
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.facecolor"] = "#f8f9fa"

# ─────────────────────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("../data/student-mat.csv", sep=";")
print(f"Loaded {len(df)} student records.")

# ─────────────────────────────────────────────────────────────
# 2. BUILD TRANSACTION DATA
#    ARM needs: one list of items per student (no numeric columns).
#    Each attribute value becomes a labeled "item", e.g. "StudyTime:High"
# ─────────────────────────────────────────────────────────────
def build_transaction(row):
    """Convert one student row into a list of descriptive items."""
    items = []

    # Study time (1–4 scale)
    st_map = {1: "StudyTime:Low", 2: "StudyTime:Medium",
              3: "StudyTime:High", 4: "StudyTime:VeryHigh"}
    items.append(st_map.get(row["studytime"], "StudyTime:Low"))

    # School absences (threshold: >6 = High)
    items.append("Absences:High" if row["absences"] > 6 else "Absences:Low")

    # Weekend alcohol (1–5 scale)
    walc_map = {1: "Alcohol:None", 2: "Alcohol:Low", 3: "Alcohol:Medium",
                4: "Alcohol:High", 5: "Alcohol:VeryHigh"}
    items.append(walc_map.get(row["Walc"], "Alcohol:Low"))

    # Internet access
    items.append("Internet:Yes" if row["internet"] == "yes" else "Internet:No")

    # Past failures
    items.append("Failures:None" if row["failures"] == 0 else "Failures:Some")

    # Family support
    items.append("FamSupport:Yes" if row["famsup"] == "yes" else "FamSupport:No")

    # Higher education aspiration
    items.append("Higher:Yes" if row["higher"] == "yes" else "Higher:No")

    # Romantic relationship
    items.append("Romantic:Yes" if row["romantic"] == "yes" else "Romantic:No")

    # Free time (1–5 scale)
    ft_map = {1: "FreeTime:VeryLow", 2: "FreeTime:Low", 3: "FreeTime:Medium",
              4: "FreeTime:High", 5: "FreeTime:VeryHigh"}
    items.append(ft_map.get(row["freetime"], "FreeTime:Medium"))

    # Final grade performance label
    g3 = row["G3"]
    if g3 >= 14:
        items.append("Grade:High")
    elif g3 >= 10:
        items.append("Grade:Medium")
    else:
        items.append("Grade:Low")

    return items


transactions = [build_transaction(row) for _, row in df.iterrows()]

# Show sample transactions
print("\nSample transactions (first 3 students):")
for i, t in enumerate(transactions[:3]):
    print(f"  Student {i + 1}: {t}")

# Save sample to CSV for the website
sample_csv = pd.DataFrame({
    "Student": range(1, 11),
    "Transaction": [", ".join(t) for t in transactions[:10]]
})
sample_csv.to_csv("../Website/arm_transactions_sample.csv", index=False)
print("\nSaved sample: arm_transactions_sample.csv")

# ─────────────────────────────────────────────────────────────
# 3. ENCODE TRANSACTIONS FOR APRIORI
# ─────────────────────────────────────────────────────────────
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
trans_df = pd.DataFrame(te_array, columns=te.columns_)
print(f"\nTransaction matrix shape: {trans_df.shape}")

# ─────────────────────────────────────────────────────────────
# 4. APRIORI — MINE FREQUENT ITEMSETS
# ─────────────────────────────────────────────────────────────
MIN_SUPPORT    = 0.10   # at least 10% of students
MIN_CONFIDENCE = 0.30   # rule correct at least 30% of the time

freq_itemsets = apriori(trans_df, min_support=MIN_SUPPORT, use_colnames=True)
print(f"\nFrequent itemsets found: {len(freq_itemsets)}")
print(f"  Min support used: {MIN_SUPPORT} ({MIN_SUPPORT*100:.0f}%)")

# ─────────────────────────────────────────────────────────────
# 5. GENERATE ASSOCIATION RULES
# ─────────────────────────────────────────────────────────────
rules = association_rules(freq_itemsets, metric="confidence",
                          min_threshold=MIN_CONFIDENCE)
rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

print(f"Total rules generated  : {len(rules)}")
print(f"  Min confidence used  : {MIN_CONFIDENCE} ({MIN_CONFIDENCE*100:.0f}%)")

# ─────────────────────────────────────────────────────────────
# 6. TOP 15 BY SUPPORT, CONFIDENCE, LIFT
# ─────────────────────────────────────────────────────────────
cols = ["antecedents_str", "consequents_str", "support", "confidence", "lift"]

top_support    = rules.nlargest(15, "support")[cols].reset_index(drop=True)
top_confidence = rules.nlargest(15, "confidence")[cols].reset_index(drop=True)
top_lift       = rules.nlargest(15, "lift")[cols].reset_index(drop=True)

print("\nTop 5 rules by SUPPORT:")
print(top_support[cols].head().to_string(index=False))

print("\nTop 5 rules by CONFIDENCE:")
print(top_confidence[cols].head().to_string(index=False))

print("\nTop 5 rules by LIFT:")
print(top_lift[cols].head().to_string(index=False))

# Save CSVs for website viewing
top_support.to_csv("../Website/arm_top_support.csv", index=False)
top_confidence.to_csv("../Website/arm_top_confidence.csv", index=False)
top_lift.to_csv("../Website/arm_top_lift.csv", index=False)
print("\nSaved rule CSVs to Website/")

# ─────────────────────────────────────────────────────────────
# 7. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────

# --- 7a. Support vs Confidence scatter (colored by Lift) ---
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor("#f8f9fa")
scatter = ax.scatter(rules["support"], rules["confidence"],
                     c=rules["lift"], cmap="RdYlGn",
                     alpha=0.7, s=60, edgecolors="white", linewidth=0.4)
plt.colorbar(scatter, ax=ax, label="Lift")
ax.set_xlabel("Support", fontsize=11)
ax.set_ylabel("Confidence", fontsize=11)
ax.set_title("Association Rules — Support vs Confidence\n(colored by Lift)",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../Website/fig_arm_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_arm_scatter.png")

# --- 7b. Top 10 rules by Lift — horizontal bar chart ---
top10_lift = rules.nlargest(10, "lift").reset_index(drop=True)
labels_b = [f"{r['antecedents_str']} → {r['consequents_str']}"[:55]
            for _, r in top10_lift.iterrows()]

fig, ax = plt.subplots(figsize=(11, 6))
ax.set_facecolor("#f8f9fa")
ax.barh(range(len(labels_b)), top10_lift["lift"].values[::-1],
        color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(labels_b))))
ax.set_yticks(range(len(labels_b)))
ax.set_yticklabels(labels_b[::-1], fontsize=8)
ax.set_xlabel("Lift", fontsize=11)
ax.set_title("Top 10 Association Rules by Lift", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("../Website/fig_arm_top_lift.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_arm_top_lift.png")

# --- 7c. Network visualization (matplotlib — no networkx required) ---
top_net = rules.nlargest(20, "lift").reset_index(drop=True)
nodes = list(set(top_net["antecedents_str"].tolist() +
                 top_net["consequents_str"].tolist()))
n = len(nodes)
node_idx = {node: i for i, node in enumerate(nodes)}
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
nx_pos = np.cos(angles) * 1.4
ny_pos = np.sin(angles) * 1.4
max_lift = top_net["lift"].max()
cmap = plt.cm.YlOrRd

fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor("#f8f9fa")
ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
ax.set_aspect("equal")

for _, r in top_net.iterrows():
    i = node_idx[r["antecedents_str"]]
    j = node_idx[r["consequents_str"]]
    color = cmap(r["lift"] / max_lift)
    ax.annotate("", xy=(nx_pos[j], ny_pos[j]),
                xytext=(nx_pos[i], ny_pos[i]),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=r["lift"] / max_lift * 2.5,
                                connectionstyle="arc3,rad=0.2"))

for node, idx in node_idx.items():
    x, y = nx_pos[idx], ny_pos[idx]
    if "Grade" in node:       nc = "#e74c3c"
    elif "StudyTime" in node or "Failures" in node: nc = "#3498db"
    elif "Alcohol" in node:   nc = "#f39c12"
    else:                     nc = "#2ecc71"
    circle = plt.Circle((x, y), 0.12, color=nc, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, node.replace(":", ":\n"),
            ha="center", va="center", fontsize=6, fontweight="bold",
            zorder=4, color="white")

legend = [mpatches.Patch(color="#e74c3c", label="Grade outcome"),
          mpatches.Patch(color="#3498db", label="Academic factor"),
          mpatches.Patch(color="#f39c12", label="Alcohol/lifestyle"),
          mpatches.Patch(color="#2ecc71", label="Other factor")]
ax.legend(handles=legend, loc="lower right", fontsize=9)
ax.set_title("Association Rules Network (Top 20 by Lift)\n"
             "Arrow thickness and color = Lift strength",
             fontsize=13, fontweight="bold")
ax.axis("off")
plt.tight_layout()
plt.savefig("../Website/fig_arm_network.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig_arm_network.png")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("ARM SUMMARY")
print("="*50)
print(f"  Students (transactions)  : {len(transactions)}")
print(f"  Frequent itemsets found  : {len(freq_itemsets)}")
print(f"  Total rules generated    : {len(rules)}")
print(f"  Min support threshold    : {MIN_SUPPORT} ({MIN_SUPPORT*100:.0f}%)")
print(f"  Min confidence threshold : {MIN_CONFIDENCE} ({MIN_CONFIDENCE*100:.0f}%)")
print(f"  Max lift achieved        : {rules['lift'].max():.3f}")
print(f"  Top lift rule            : {top_lift.loc[0,'antecedents_str']} → {top_lift.loc[0,'consequents_str']}")
print("="*50)
print("\nScript 04 complete.")
