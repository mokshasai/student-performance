"""
Student Performance Analysis
=============================
Script 01 — Data Loading, Cleaning & Exploratory Data Analysis (EDA)

Dataset : UCI Student Performance (student-mat.csv)
Author  : Data Science Project 2025
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
raw_df = pd.read_csv("../data/student-mat.csv", sep=";")

print("Shape:", raw_df.shape)
print("\nFirst 5 rows:")
print(raw_df.head())

# Export raw snapshot
raw_df.to_csv("../data/student_raw.csv", index=False)

# ─────────────────────────────────────────────────────────────
# 2. DATA VALIDATION
# ─────────────────────────────────────────────────────────────
print("\nNull values per column:")
print(raw_df.isnull().sum())

print("\nDuplicate rows:", raw_df.duplicated().sum())

# ─────────────────────────────────────────────────────────────
# 3. TARGET VARIABLE — PERFORMANCE LABEL (Classification)
# ─────────────────────────────────────────────────────────────
def performance_label(grade):
    """Convert numeric G3 grade (0–20) to categorical performance label."""
    if grade >= 14:
        return "High"
    elif grade >= 10:
        return "Medium"
    else:
        return "Low"

raw_df["performance"] = raw_df["G3"].apply(performance_label)
print("\nPerformance class distribution:")
print(raw_df["performance"].value_counts())

# ─────────────────────────────────────────────────────────────
# 4. DATA CLEANING & PREPARATION
# ─────────────────────────────────────────────────────────────

# 4a. Drop G1 and G2 to prevent data leakage when predicting G3
clean_df = raw_df.drop(columns=["G1", "G2"])

# 4b. Binary encode yes/no columns (LabelEncoder equivalent)
binary_cols = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic"
]
binary_map = {"yes": 1, "no": 0, "GP": 0, "MS": 1,
              "F": 0, "M": 1, "U": 0, "R": 1,
              "LE3": 0, "GT3": 1, "T": 0, "A": 1}

for col in binary_cols:
    clean_df[col] = clean_df[col].map(binary_map).fillna(clean_df[col])

# 4c. One-hot encode multi-category columns
multi_cat_cols = ["Mjob", "Fjob", "reason", "guardian"]
clean_df = pd.get_dummies(clean_df, columns=multi_cat_cols)

print("\nCleaned dataset shape:", clean_df.shape)
print(clean_df.head())

# Export cleaned dataset
clean_df.to_csv("../data/student_clean.csv", index=False)

# ─────────────────────────────────────────────────────────────
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams["figure.facecolor"] = "#f8f9fa"

# --- 5a. Grade Distribution ---
plt.figure(figsize=(8, 5))
sns.histplot(raw_df["G3"], bins=21, kde=True, color="#2c3e50")
plt.title("Distribution of Final Grades (G3)", fontsize=14, fontweight="bold")
plt.xlabel("Final Grade (0–20)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("../Website/fig_grade_distribution.png", dpi=150)
plt.close()

# --- 5b. Study Time vs Final Grade ---
plt.figure(figsize=(8, 5))
sns.boxplot(x="studytime", y="G3", data=raw_df, palette="Blues")
plt.title("Study Time vs Final Grade", fontsize=14, fontweight="bold")
plt.xlabel("Study Time (1=<2h, 2=2-5h, 3=5-10h, 4=>10h)")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("../Website/fig_studytime_vs_grade.png", dpi=150)
plt.close()

# --- 5c. Absences vs Final Grade ---
plt.figure(figsize=(8, 5))
sns.scatterplot(x="absences", y="G3", data=raw_df, alpha=0.6, color="#e74c3c")
plt.title("Absences vs Final Grade", fontsize=14, fontweight="bold")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("../Website/fig_absences_vs_grade.png", dpi=150)
plt.close()

# --- 5d. Gender Distribution ---
plt.figure(figsize=(6, 4))
raw_df["sex"].value_counts().plot(kind="bar", color=["#3498db", "#e74c3c"])
plt.title("Gender Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Gender (F = Female, M = Male)")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../Website/fig_gender_distribution.png", dpi=150)
plt.close()

# --- 5e. Parental Education vs Grade ---
plt.figure(figsize=(8, 5))
sns.boxplot(x="Medu", y="G3", data=raw_df, palette="Greens")
plt.title("Mother's Education vs Final Grade", fontsize=14, fontweight="bold")
plt.xlabel("Mother Education Level (0=none … 4=higher)")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("../Website/fig_parent_edu_vs_grade.png", dpi=150)
plt.close()

# --- 5f. Alcohol Consumption vs Grade ---
plt.figure(figsize=(8, 5))
sns.boxplot(x="Walc", y="G3", data=raw_df, palette="Reds")
plt.title("Weekend Alcohol Consumption vs Final Grade", fontsize=14, fontweight="bold")
plt.xlabel("Weekend Alcohol (1=very low … 5=very high)")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.savefig("../Website/fig_alcohol_vs_grade.png", dpi=150)
plt.close()

# --- 5g. Internet Access vs Performance ---
plt.figure(figsize=(7, 5))
internet_perf = pd.crosstab(raw_df["internet"], raw_df["performance"])
internet_perf.plot(kind="bar", color=["#e74c3c", "#3498db", "#2ecc71"])
plt.title("Internet Access vs Performance Category", fontsize=14, fontweight="bold")
plt.xlabel("Internet Access")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(title="Performance")
plt.tight_layout()
plt.savefig("../Website/fig_internet_vs_performance.png", dpi=150)
plt.close()

# --- 5h. Family Support vs Performance ---
plt.figure(figsize=(7, 5))
famsup_perf = pd.crosstab(raw_df["famsup"], raw_df["performance"])
famsup_perf.plot(kind="bar", color=["#e74c3c", "#3498db", "#2ecc71"])
plt.title("Family Support vs Performance Category", fontsize=14, fontweight="bold")
plt.xlabel("Family Educational Support")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(title="Performance")
plt.tight_layout()
plt.savefig("../Website/fig_family_support_vs_performance.png", dpi=150)
plt.close()

# --- 5i. Age Distribution ---
plt.figure(figsize=(7, 4))
raw_df["age"].value_counts().sort_index().plot(kind="bar", color="#9b59b6")
plt.title("Age Distribution of Students", fontsize=14, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("../Website/fig_age_distribution.png", dpi=150)
plt.close()

print("\nEDA complete. All figures saved to Website/")

# ─────────────────────────────────────────────────────────────
# 6. API DATA COLLECTION — World Bank Education Indicators
# ─────────────────────────────────────────────────────────────
url = "https://api.worldbank.org/v2/country/PRT/indicator/SE.XPD.TOTL.GD.ZS?format=json"
try:
    response = requests.get(url, timeout=10)
    data = response.json()
    edu_spending = [
        entry["value"] for entry in data[1]
        if entry["value"] is not None
    ][0]  # Most recent non-null value
    print(f"\nPortugal Education Spending (% of GDP): {edu_spending:.2f}%")
    clean_df["edu_spending_gdp"] = edu_spending
except Exception as e:
    print(f"\nAPI call failed (offline?): {e}")

print("\nScript 01 complete.")
