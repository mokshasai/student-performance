# 🎓 Student Performance Analysis

A data science project analyzing student academic performance using machine learning techniques on the [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance).

## 🌐 Live Site
**[View Website →](https://your-site.netlify.app)**  
*(Replace with your Netlify URL after deploying)*

---

## 📁 Project Structure

```
Student-Performance-Analysis/
│
├── Website/                    ← Static site hosted on Netlify
│   ├── index.html              ← Introduction (3 paragraphs, 2 images, 10 questions)
│   ├── data-prep.html          ← Data Prep & EDA
│   ├── pca.html                ← PCA: 2D/3D projections, variance, eigenvalues
│   ├── clustering.html         ← KMeans, Hierarchical, DBSCAN clustering
│   ├── arm.html                ← Association Rule Mining (Apriori)
│   ├── style.css               ← Dark theme design system
│   ├── script.js               ← Animations & nav interactions
│   ├── code/                   ← Syntax-highlighted code viewer pages
│   │   ├── code_eda.html
│   │   ├── code_pca.html
│   │   ├── code_clustering.html
│   │   └── code_arm.html
│   └── [all figures .png]      ← Generated visualizations
│
├── code/                       ← Individual Python analysis scripts
│   ├── 01_data_prep_eda.py     ← Data loading, cleaning, EDA charts
│   ├── 02_pca.py               ← PCA analysis & visualizations
│   ├── 03_clustering.py        ← KMeans, Hierarchical, DBSCAN
│   └── 04_arm.py               ← Association Rule Mining (Apriori)
│
├── data/                       ← Raw & processed datasets
│   ├── student-mat.csv         ← Primary dataset (Math, 395 students)
│   ├── student-por.csv         ← Secondary dataset (Portuguese)
│   ├── student_raw.csv         ← Unprocessed export
│   ├── student_clean.csv       ← Cleaned & encoded export
│   ├── student-attributes.txt  ← Feature descriptions
│   └── student-merge.R         ← R script from UCI (reference only)
│
├── notebooks/
│   └── Student_performance.ipynb  ← Full exploratory notebook
│
├── .gitignore
├── netlify.toml                ← Netlify publish config (publish = Website/)
└── README.md
```

---

## 🔬 Analysis Sections

| Section | Description | Code File |
|---------|-------------|-----------|
| **Data Prep & EDA** | Loading, cleaning, encoding, and visualizing | `code/01_data_prep_eda.py` |
| **PCA** | 2D/3D dimensionality reduction, variance, eigenvalues | `code/02_pca.py` |
| **Clustering** | KMeans (Silhouette), Hierarchical (Ward), DBSCAN | `code/03_clustering.py` |
| **ARM** | Apriori rules by support, confidence, and lift | `code/04_arm.py` |

---

## ⚙️ Running the Code

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy mlxtend requests
```

### Run individual scripts
```bash
cd code

# Data preparation & EDA
python 01_data_prep_eda.py

# PCA analysis
python 02_pca.py

# Clustering
python 03_clustering.py

# Association Rule Mining
python 04_arm.py
```

> **Note:** Scripts assume the dataset is at `../data/student-mat.csv` and save figures to `../Website/`.

---

## 🚀 Deploying to Netlify

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial project commit"
   git remote add origin https://github.com/YOUR_USERNAME/student-performance-analysis.git
   git push -u origin main
   ```

2. **Connect to Netlify:**
   - Go to [app.netlify.com](https://app.netlify.com) → New site → Import from GitHub
   - Select your repo
   - Build settings are pre-configured via `netlify.toml` → **Publish dir:** `Website`
   - Click **Deploy site**

3. Your site will be live at `https://your-project.netlify.app`

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
- **File:** `student-mat.csv` (Math course)
- **Size:** 395 students × 33 attributes
- **Target:** G3 final grade → Low / Medium / High

---

## 🛠️ Tech Stack

- **Analysis:** Python 3, pandas, NumPy, scikit-learn, scipy, mlxtend, matplotlib, seaborn
- **Website:** Vanilla HTML/CSS/JS, Highlight.js (code viewer), Google Fonts
- **Hosting:** Netlify (static site, continuous deployment from GitHub)
