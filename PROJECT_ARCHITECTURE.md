# Ship Performance ML Project — Architecture

---

## Overview

An end-to-end unsupervised + supervised machine learning study on 2,600 commercial
shipping voyages. The project flows through three sequential notebooks: exploratory
analysis, fleet clustering, and maintenance-status classification.

---

## Project Structure

```
ShippingPerformace/
│
├── data/
│   ├── Ship_Performance_Dataset.csv              ← Raw dataset (2,600 rows × 20 cols)
│   └── Ship_Performance_Dataset_Modernized.csv   ← Modernised variant
│
├── EDA/                                          ← Stage 1: Exploratory Data Analysis
│   ├── Ship_EDA.Rmd
│   ├── Ship_EDA.html
│   ├── Ship_EDA_files/                           ← HTML dependencies
│   ├── EDA_Plot_Walkthrough.md
│   └── plots/                                    ← 22 saved EDA plots
│
├── clustering/                                   ← Stage 2: Unsupervised Clustering
│   ├── Ship_Clustering.Rmd
│   ├── Ship_Clustering.html
│   ├── Clustering_Analysis_Report.md             ← Iterative optimisation log (v1→v6)
│   ├── ML_Clustering_Plan.md
│   ├── Clustering_Ship.R
│   └── plots/                                    ← 23 saved clustering plots
│
├── classification/                               ← Stage 3: Supervised Classification
│   ├── Ship_Classifier.Rmd
│   └── Ship_Classifier.html
│
├── PROJECT_ARCHITECTURE.md                       ← This file
└── PLAN.md
```

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Raw Dataset                                   │
│         data/Ship_Performance_Dataset.csv                       │
│         2,600 voyages × 20 features                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Exploratory Data Analysis        Ship_EDA.Rmd        │
│                                                                  │
│  • Data quality audit (missing values, distributions)           │
│  • Univariate analysis — all 20 features                        │
│  • Bivariate analysis — correlations, group comparisons         │
│  • Multivariate analysis — PCA, parallel coordinates            │
│  • Clustering recommendations derived from EDA                  │
└────────────────────────┬────────────────────────────────────────┘
                         │  findings inform feature selection
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Fleet Clustering                Ship_Clustering.Rmd  │
│                                                                  │
│  Feature Engineering                                            │
│  ├── 5 domain ratio features (Profit_Margin, Cost_per_nm, …)   │
│  ├── One-hot encoding (Maintenance_Status, Weather_Condition)   │
│  └── Winsorization (1st–99th percentile clip)                   │
│                                                                  │
│  Full Pipeline (documented, 19 features → 15 PCA components)   │
│  └── OHE → Winsorise → Z-score scale → Corr filter → PCA       │
│                                                                  │
│  Focused Subset (final clustering input)                        │
│  └── 5 pure efficiency & financial ratios → winsorise → scale  │
│                                                                  │
│  Algorithms (K = 3)                                             │
│  ├── K-Means          sil = 0.2844                              │
│  ├── Hierarchical     sil = 0.5240  ◀ best for reporting        │
│  ├── DBSCAN           sil = 0.6436  ◀ highest score             │
│  ├── GMM (EM)         sil = 0.0513                              │
│  └── PAM (K-Medoids)  sil = 0.2843                              │
│                                                                  │
│  Output: 3 fleet performance archetypes                         │
│  ├── Cluster 1 — High-Efficiency Performers                     │
│  ├── Cluster 2 — Balanced Mid-Range Fleet                       │
│  └── Cluster 3 — At-Risk Underperformers                        │
└────────────────────────┬────────────────────────────────────────┘
                         │  cluster labels + engineered features
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Maintenance Classifier          Ship_Classifier.Rmd  │
│                                                                  │
│  Target: Maintenance_Status (Good / Fair / Critical)            │
│                                                                  │
│  Preprocessing                                                   │
│  ├── Drop leakage columns (Date, Efficiency_nm_per_kWh)         │
│  ├── 80/20 stratified train/test split                          │
│  └── 5-fold cross-validation                                    │
│                                                                  │
│  Algorithms                                                      │
│  ├── Multinomial Logistic Regression                            │
│  ├── K-Nearest Neighbours (KNN)                                 │
│  ├── Random Forest                                              │
│  └── Gradient Boosting Machine (GBM)                            │
│                                                                  │
│  Evaluation                                                      │
│  ├── Accuracy / F1 / ROC-AUC (one-vs-rest)                      │
│  ├── Confusion matrices                                         │
│  ├── Per-class precision / recall                               │
│  └── Feature importance comparison                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Value |
|---|---|
| File | `data/Ship_Performance_Dataset.csv` |
| Rows | 2,600 voyages |
| Columns | 20 |
| Date range | Jan 2025 |
| Missing values | None (complete dataset) |

### Feature Catalogue

| Feature | Type | Role |
|---|---|---|
| `Date` | Date | Dropped (leakage risk) |
| `Ship_Type` | Categorical | Bulk Carrier / Container / Tanker / … |
| `Route_Type` | Categorical | Short / Medium / Long |
| `Engine_Type` | Categorical | Diesel / HFO / LNG |
| `Maintenance_Status` | Categorical | **Classifier target** — Good / Fair / Critical |
| `Weather_Condition` | Categorical | Calm / Moderate / Rough |
| `Speed_Over_Ground_knots` | Numeric | Clustering + classifier input |
| `Engine_Power_kW` | Numeric | Clustering + classifier input |
| `Distance_Traveled_nm` | Numeric | Clustering + classifier input |
| `Draft_meters` | Numeric | Clustering + classifier input |
| `Cargo_Weight_tons` | Numeric | Clustering + classifier input |
| `Operational_Cost_USD` | Numeric | Clustering + classifier input |
| `Revenue_per_Voyage_USD` | Numeric | Clustering + classifier input |
| `Turnaround_Time_hours` | Numeric | Clustering + classifier input |
| `Efficiency_nm_per_kWh` | Numeric | Clustering input / **dropped from classifier** |
| `Seasonal_Impact_Score` | Numeric | Clustering + classifier input |
| `Weekly_Voyage_Count` | Numeric | Clustering + classifier input |
| `Average_Load_Percentage` | Numeric | Clustering + classifier input |

---

## Stage 2 — Clustering Detail

### Preprocessing Decisions

| Step | Decision | Rationale |
|---|---|---|
| Categorical encoding | One-hot (not ordinal) | Ordinal integers impose false linear distance; OHE treats categories equally |
| Outlier handling | Winsorize at 1st–99th pct | K-Means centroids are mean-based — extremes drag them away from true centres |
| Scaling | Z-score standardisation | Prevents large-unit features (USD, kWh) from dominating Euclidean distance |
| Correlation filter | Remove \|r\| > 0.75 | Eliminates redundant dimensions that duplicate distance signal |
| Dimensionality reduction | PCA → 85% variance threshold | Mitigates curse of dimensionality in 19-feature space |
| **Focused subset** | **5 efficiency/financial ratios** | **Concentrates cluster signal; removes noise dimensions** |

### Domain-Engineered Features

```
Profit_Margin        = (Revenue - Cost) / Revenue
Cost_per_nm          = Operational_Cost / Distance_Traveled
Power_per_Cargo_ton  = Engine_Power / Cargo_Weight
Revenue_per_nm       = Revenue / Distance_Traveled
Speed_Load_ratio     = Speed / Avg_Load_Pct  ← removed by corr filter
```

### Focused 5-Feature Subset (final clustering input)

```
Profit_Margin         — profitability per voyage
Cost_per_nm           — cost efficiency normalised by route length
Power_per_Cargo_ton   — power utilisation relative to payload
Revenue_per_nm        — revenue yield per nautical mile
Efficiency_nm_per_kWh — direct fuel efficiency (original metric)
```

### Algorithm Comparison

| Algorithm | Silhouette | Clusters | Notes |
|---|---|---|---|
| K-Means | 0.2844 | 3 | Fast, balanced, deployable |
| Hierarchical (Ward.D2) | 0.5240 | 3 | Best for presentation / dendrogram |
| **DBSCAN** | **0.6436** | **3 + 139 noise** | **Highest score; noise = anomalous voyages** |
| GMM (EM) | 0.0513 | 3 | Soft assignments; not suited to this space |
| PAM (K-Medoids) | 0.2843 | 3 | Interpretable medoids; outlier-robust |

### Silhouette Score Evolution

| Iteration | Key Change | K-Means | Hierarchical | DBSCAN |
|---|---|---|---|---|
| v1 | Ordinal encoding (baseline) | 0.049 | 0.009 | 1 cluster |
| v2 | One-hot encoding | 0.097 | 0.091 | 1 cluster |
| v3 | PCA reduction | 0.097 | 0.068 | 1 cluster |
| v4 | Domain features | 0.090 | 0.062 | 1 cluster |
| v5 | Winsorization | 0.116 | 0.130 | 1 cluster |
| **v6** | **Focused 5-feature subset** | **0.284** | **0.524** | **0.644** |

### Fleet Archetypes (K=3)

| Cluster | Name | Signal | Action |
|---|---|---|---|
| 1 | High-Efficiency Performers | High margin, low cost/nm, high efficiency | Benchmark — replicate |
| 2 | Balanced Mid-Range Fleet | Moderate across all 5 ratios | Monitor — find uplift levers |
| 3 | At-Risk Underperformers | Low margin, high cost/nm, low efficiency | Intervene — cost & maintenance review |

---

## Stage 3 — Classifier Detail

### Problem Setup

| Property | Value |
|---|---|
| Task | Multi-class classification |
| Target | `Maintenance_Status` (Good / Fair / Critical) |
| Split | 80% train / 20% test (stratified) |
| Validation | 5-fold cross-validation |
| Leakage guard | `Date` and `Efficiency_nm_per_kWh` dropped |

### Algorithms

| Algorithm | Type | Key Hyperparameters |
|---|---|---|
| Logistic Regression | Linear, probabilistic | Multinomial, L2 regularisation |
| KNN | Instance-based | K tuned via CV |
| Random Forest | Ensemble (bagging) | ntree=500, mtry tuned via CV |
| GBM | Ensemble (boosting) | n.trees, interaction.depth, shrinkage tuned |

### Evaluation Metrics

- **Accuracy** — overall proportion correct
- **F1 (macro)** — harmonic mean of precision and recall, averaged across classes
- **ROC-AUC (one-vs-rest)** — discrimination ability per class
- **Confusion matrix** — per-class error breakdown
- **Feature importance** — compared across RF and GBM

---

## Key Technical Choices & Justifications

| Choice | Alternative considered | Why this was chosen |
|---|---|---|
| One-hot encoding for categoricals | Ordinal integers | Distance metrics treat ordinal values as magnitudes — OHE avoids false ordering |
| Winsorize before scale | Remove outliers | Winsorizing keeps all rows; removing reduces sample size |
| Focused 5-feature subset | Full 19-feature PCA space | Pure ratio signals have sharper cluster boundaries; noise dims dilute distance |
| Ward.D2 linkage | Average / complete linkage | Minimises total within-cluster variance — produces compact, equal-size clusters |
| DBSCAN eps = 1.0 | eps = 5.5 (original) | Tuned to 5-dimensional focused space; original eps collapses everything into 1 cluster |
| K = 3 | K = 2 (silhouette peak) | 3 clusters yield 3 operationally meaningful archetypes; K=2 is too coarse |
| Stratified train/test split | Random split | Preserves class proportions — critical for imbalanced `Maintenance_Status` |

---

## Dependencies

### R Packages

| Package | Used in | Purpose |
|---|---|---|
| `tidyverse` | All | Data wrangling |
| `ggplot2` | All | Visualisation |
| `cluster` | Clustering | Silhouette, PAM |
| `factoextra` | Clustering | fviz_nbclust, fviz_cluster |
| `dbscan` | Clustering | DBSCAN algorithm |
| `mclust` | Clustering | GMM / EM algorithm |
| `dendextend` | Clustering | Dendrogram styling |
| `fastDummies` | Clustering | One-hot encoding |
| `caret` | Clustering + Classifier | Correlation filter, train/test split, CV |
| `randomForest` | Classifier | Random Forest |
| `gbm` | Classifier | Gradient Boosting |
| `pROC` | Classifier | ROC-AUC curves |
| `knitr` + `kableExtra` | All | Table rendering |
| `corrplot` + `GGally` | EDA | Correlation visualisation |
| `skimr` | EDA | Summary statistics |

---

*Architecture document — Ship Performance ML Project — last updated 2026-03-26*
