# 🎬 Movie Dataset — ML & Data Analysis Project

A comprehensive machine learning project that applies multiple classification and regression algorithms on a movies dataset (`movies.csv`) to predict movie ratings and revenue.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Algorithms Implemented](#algorithms-implemented)
- [Visualizations](#visualizations)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)

---

## Overview

This project explores a movies dataset through the full machine learning pipeline — from data cleaning and preprocessing to training multiple ML models and evaluating their performance. The primary goals are:

- **Regression**: Predict a movie's `revenue` based on features like budget, popularity, runtime, and vote count.
- **Classification**: Predict whether a movie is *high rated* (`vote_average >= 7.0`) using various classifiers.

---

## Dataset

**File:** `movies.csv`

**Key columns used:**

| Column | Description |
|---|---|
| `budget` | Production budget of the movie |
| `revenue` | Total box office revenue |
| `popularity` | Popularity score |
| `runtime` | Movie duration in minutes |
| `vote_average` | Average user rating (0–10) |
| `vote_count` | Total number of votes |
| `title` | Movie title |

**Target Variables:**
- `revenue` — used for regression
- `is_high_rated` — binary label (`1` if `vote_average >= 7.0`, else `0`) used for classification

---

## Project Workflow

```
1. Data Loading          →  Load movies.csv using pandas
2. Exploratory Analysis  →  .info(), .isnull(), descriptive stats
3. Data Preprocessing    →  Handle nulls, StandardScaler, LabelEncoder
4. Feature Engineering   →  Select relevant numeric features
5. Train/Test Split      →  80% train / 20% test (random_state=42)
6. Model Training        →  Train 7 ML algorithms
7. Evaluation            →  Accuracy, Confusion Matrix, Classification Report
8. Visualization         →  Charts, heatmaps, ROC/AUC curves
```

---

## Algorithms Implemented

### Regression
| Algorithm | Target | Key Metric |
|---|---|---|
| Linear Regression | `revenue` | R² Score, MAE |

### Classification
| Algorithm | Notes |
|---|---|
| Logistic Regression | Feature scaling applied |
| K-Nearest Neighbors (KNN) | `n_neighbors=5`, scaling required |
| Support Vector Machine (SVM/SVC) | RBF kernel, `C=1.0` |
| Decision Tree | `max_depth=3` to prevent overfitting |
| Random Forest | `n_estimators=100`, includes feature importance |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.1` |

---

## Visualizations

The notebook generates the following plots:

- **Boxplots** — Distribution of `vote_count` and `popularity`
- **Correlation Heatmap** — Pearson correlation between numeric features
- **Scatter Plot** — Budget vs Revenue
- **Bar Chart** — Top 10 movies by revenue
- **Distribution Plots** — Histograms with KDE for budget, revenue, runtime, vote_average, popularity
- **ROC Curve** — Model discriminative ability (with AUC score highlighted)
- **Feature Importance** — Bar chart from Random Forest and Gradient Boosting
- **Decision Tree** — Visual tree structure (max_depth=3)
- **Success vs Unsuccessful** — Movies categorized by profit/loss (revenue > budget)
- **VIF Analysis** — Multicollinearity check using Variance Inflation Factor

---

## Requirements

```bash
pip install pandas scikit-learn matplotlib seaborn statsmodels
```

> **Note:** The notebook was originally developed on **Google Colab**. The first cell uses `google.colab.files.upload()` for file upload — replace this with a local `pd.read_csv()` path if running locally.

---

## How to Run

### On Google Colab (Recommended)
1. Open the `.ipynb` file in [Google Colab](https://colab.research.google.com/)
2. Run the first cell to upload `movies.csv`
3. Run all remaining cells in order

### Locally (Jupyter Notebook)
1. Clone or download the repository
2. Place `movies.csv` in the same directory as the notebook
3. Remove/comment out the Colab file upload cell
4. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn statsmodels jupyter
   ```
5. Launch the notebook:
   ```bash
   jupyter notebook ML_DL_PROJECT.ipynb
   ```

---

## Results

All classification models predict the binary label `is_high_rated` (vote_average ≥ 7.0).

| Model | Evaluated By |
|---|---|
| Linear Regression | R² Score + MAE |
| Logistic Regression | Accuracy, Confusion Matrix, ROC-AUC |
| KNN | Accuracy, Confusion Matrix, Classification Report |
| SVM / SVC | Accuracy, Confusion Matrix, Classification Report |
| Decision Tree | Accuracy, Confusion Matrix + Tree Visualization |
| Random Forest | Accuracy, Feature Importance |
| Gradient Boosting | Accuracy, Feature Importance |

> Refer to the individual cells in the notebook for exact metric outputs after running.

---

## Author

*BCA Final Year Project — Machine Learning & Data Analysis*