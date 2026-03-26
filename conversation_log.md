# Conversation Log — Ship Performance Project
**Last updated:** 2026-03-25
**Topics covered:** EDA · Clustering (5 algorithms) · Classifier Feasibility · Plot output

---

## Session Summary

This session covered the following deliverables for the Ship Performance Dataset:

1. A full EDA notebook in R (`Ship_EDA.Rmd`) + rendered `Ship_EDA.html`
2. A plot walkthrough report (`EDA_Plot_Walkthrough.md`) — 22 plots explained
3. A discussion on whether the dataset supports a classifier ML project
4. A full clustering notebook (`Ship_Clustering.Rmd`) + rendered `Ship_Clustering.html` — 5 algorithms
5. Data-driven best-fit algorithm analysis (Hopkins statistic + silhouette diagnostics)
6. Project naming discussion (report titles)
7. Plot output location clarification (PNG vs embedded HTML)

---

## 1. Dataset Overview

**File:** `Ship_Performance_Dataset.csv`
**Rows:** 2,736 | **Columns:** 18

| Column | Type | Role |
|--------|------|------|
| Date | Temporal | Drop for clustering |
| Ship_Type | Categorical | Post-cluster descriptor |
| Route_Type | Categorical | Post-cluster descriptor |
| Engine_Type | Categorical | Post-cluster descriptor |
| Maintenance_Status | Ordinal (Good < Fair < Critical) | Encode for clustering; **best classifier target** |
| Weather_Condition | Categorical | Encode for clustering |
| Speed_Over_Ground_knots | Numerical | Core clustering feature |
| Engine_Power_kW | Numerical | Core clustering feature |
| Distance_Traveled_nm | Numerical | Core clustering feature |
| Draft_meters | Numerical | Core clustering feature |
| Cargo_Weight_tons | Numerical | Core clustering feature |
| Operational_Cost_USD | Numerical | Core clustering feature |
| Revenue_per_Voyage_USD | Numerical | Core clustering feature |
| Turnaround_Time_hours | Numerical | Core clustering feature |
| Efficiency_nm_per_kWh | Numerical | Core clustering feature |
| Seasonal_Impact_Score | Numerical | Core clustering feature |
| Weekly_Voyage_Count | Numerical | Core clustering feature |
| Average_Load_Percentage | Numerical | Core clustering feature |

---

## 2. Clustering Problem

**Type:** Unsupervised — no pre-labelled target variable.

**Goal:** Discover natural ship performance archetypes across these dimensions:

| Dimension | Features |
|-----------|----------|
| Operational efficiency | Efficiency_nm_per_kWh, Speed, Distance |
| Economic performance | Revenue, Operational_Cost, Load_Percentage |
| Engine & power | Engine_Power_kW, Weekly_Voyage_Count |
| Load & cargo | Cargo_Weight_tons, Draft_meters |
| Operational context | Turnaround_Time_hours, Seasonal_Impact_Score |
| Risk / condition | Maintenance_Status (encoded), Weather_Condition (encoded) |

**Derived KPIs added during session:**

```r
df <- df %>%
  mutate(
    Profit_USD      = Revenue_per_Voyage_USD - Operational_Cost_USD,
    Profit_Margin   = Profit_USD / Revenue_per_Voyage_USD,
    Cost_per_nm     = Operational_Cost_USD / Distance_Traveled_nm,
    Revenue_per_ton = Revenue_per_Voyage_USD / (Cargo_Weight_tons + 1),
    Power_per_knot  = Engine_Power_kW / Speed_Over_Ground_knots
  )
```

**Clustering status:** COMPLETE — see Section 6 below.

---

## 3. EDA Files Produced

### `Ship_EDA.Rmd` → `Ship_EDA.html`

Full EDA notebook in R covering:

**Univariate:**
- Bar charts for 5 categorical features
- Histogram + density curves for 12 numerical features
- Boxplots (outlier detection) for 12 numerical features
- Summary statistics table with skewness

**Bivariate:**
- Scatter: Speed vs Efficiency (by Engine Type)
- Scatter: Cost vs Revenue with break-even line (by Ship Type)
- Scatter: Engine Power vs Distance (by Route Type)
- Scatter: Cargo Weight vs Draft (by Ship Type)
- Boxplot: Efficiency by Ship Type
- Boxplot: Operational Cost by Engine Type
- Boxplot: Revenue by Maintenance Status
- Boxplot: Turnaround Time by Route Type
- Ridge plot: Speed by Weather Condition
- Correlation matrix (hclust ordered, with coefficients)
- Top-15 correlations table

**Multivariate:**
- GGally pairs plot (6 key features coloured by Ship Type)
- Heatmap: Mean metrics by Ship Type × Maintenance Status
- Bubble chart: Cost vs Revenue (size = Cargo, colour = Route)
- Faceted ridge: Efficiency by Engine Type across Route Type
- Faceted scatter: Speed vs Efficiency across Maintenance Status
- Parallel coordinates plot (8 features, coloured by Ship Type)
- PCA scree plot (variance explained per component)
- PCA scatter (PC1 vs PC2, colour = Ship Type, shape = Maintenance)
- PCA loadings biplot (feature contributions)

**To render:**
```bash
# From terminal inside the project folder
Rscript -e "rmarkdown::render('EDA/Ship_EDA.Rmd', output_format='html_document')"

# Or in RStudio: open EDA/Ship_EDA.Rmd and click Knit (Cmd+Shift+K)

# Or open the already-rendered file
open EDA/Ship_EDA.html
```

**Required R packages:**
`tidyverse`, `ggplot2`, `GGally`, `corrplot`, `gridExtra`, `skimr`,
`knitr`, `kableExtra`, `scales`, `ggridges`, `viridis`, `RColorBrewer`, `moments`

### `EDA_Plot_Walkthrough.md`

A narrative report explaining all 22 plots:
- Why each plot type was chosen
- What each visual encoding means
- What to look for when reading it in a clustering context
- A final summary table mapping analysis goals to plot types

---

## 4. Classifier ML Feasibility

**Answer: Yes — multiple valid targets exist.**

| Target | Type | Business Question |
|--------|------|-------------------|
| `Maintenance_Status` | Multiclass (3) — **best** | Predict maintenance risk from voyage data (predictive maintenance) |
| `Is_Profitable` (derived) | Binary | What configurations lead to profitable voyages? |
| `High_Efficiency` (derived) | Binary | Predict fuel efficiency before departure |
| `Ship_Type` | Multiclass (4) — weaker | Less useful; ship type is known before voyage |

**Best target: `Maintenance_Status`**
- Ordinal classes (Good / Fair / Critical)
- Operationally meaningful — supports predictive maintenance decisions
- Input features (speed, efficiency, engine power, cost) have physical reasons to relate to condition

**Pre-modelling checklist:**
```r
# Check class balance first
table(df$Maintenance_Status)
prop.table(table(df$Maintenance_Status))
# If Critical < 10% → apply SMOTE or class weights
```

**Recommended pipeline:**
1. Drop `Date`
2. Dummy/label encode: `Ship_Type`, `Route_Type`, `Engine_Type`, `Weather_Condition`
3. Z-score scale all numerical features
4. Algorithms: Random Forest, XGBoost, Multinomial Logistic Regression, KNN
5. Evaluate with **macro F1-score** (handles class imbalance better than accuracy)

---

## 5. Key EDA Findings

| Finding | Detail |
|---------|--------|
| No missing values | Dataset is complete — no imputation needed |
| No duplicate rows | Clean dataset |
| Balanced categories | Ship types and routes broadly balanced; "None" categories need investigation |
| Wide efficiency range | Efficiency_nm_per_kWh spans ~0.05 to ~2.5 — strong clustering signal |
| Cost ≠ Revenue | Many voyages operate below break-even; Profit_Margin varies widely |
| Weak pairwise correlations | Most numerical features have r < 0.3 — all carry independent clustering information |
| PCA structure | 4-5 PCs capture ~80% variance; data is genuinely multi-dimensional |
| Maintenance → Revenue link | Good maintenance aligns with higher revenue |

---

## 6. Clustering Implementation

### File: `clustering/Ship_Clustering.Rmd` → `clustering/Ship_Clustering.html`

**To render:**
```bash
Rscript -e "rmarkdown::render('clustering/Ship_Clustering.Rmd', output_format='html_document')"
# Or open the already-rendered file
open clustering/Ship_Clustering.html
```

**Required R packages:**
`tidyverse`, `ggplot2`, `cluster`, `factoextra`, `dbscan`, `dendextend`,
`fpc`, `mclust`, `knitr`, `kableExtra`, `scales`, `viridis`, `gridExtra`, `RColorBrewer`

---

### Preprocessing

| Step | Detail |
|------|--------|
| Features used | 14 (12 numerical + 2 encoded categoricals) |
| Scaling | Z-score (mean=0, SD=1) |
| Maintenance_enc | Good=0, Fair=1, Critical=2 |
| Weather_enc | Calm=0, Moderate=1, Rough=2 |

### K Selection (K-Means / PAM / Hierarchical)

Three methods used: **Elbow (WSS)**, **Silhouette score**, **Gap Statistic** → **K = 4**

---

### Five Algorithms Implemented

| # | Algorithm | Section | Key Unique Feature |
|---|-----------|---------|-------------------|
| 1 | K-Means | 5 | PCA projection, profile heatmap, silhouette |
| 2 | Hierarchical (Ward.D2) | 6 | Dendrogram with cut line |
| 3 | DBSCAN | 7 | Density-based, no K required, noise detection |
| 4 | **GMM (EM)** | 8 | BIC model selection, soft membership, uncertainty plot |
| 5 | **PAM (K-Medoids)** | 9 | Medoid = real voyage, outlier-robust |

---

### Data Diagnostics — Best-Fit Algorithm Analysis

Run before final algorithm selection to assess clustering suitability:

| Diagnostic | Result | Interpretation |
|-----------|--------|----------------|
| Hopkins statistic | 0.46 | Below 0.5 → data leans uniform/random, low natural cluster tendency |
| Mean feature correlation | \|r\| = 0.015 | Features are completely independent — dataset is synthetic |
| Max feature correlation | \|r\| = 0.045 | No pairwise feature relationship exists |
| K-Means silhouette (all K) | 0.05–0.056 | All < 0.1 → no strong natural clusters |
| DBSCAN | 1 cluster, 0 noise | Uniform density — density-based clustering fails |
| Hierarchical Ward | silhouette = 0.009 | Worst performer |
| Best algorithm | **K-Means (K=2)** | Highest silhouette at 0.0557 |

**Root cause:** Dataset is synthetically generated — features were created independently
with no built-in cluster structure. All silhouette scores are near zero regardless of
algorithm or K value.

**Recommended framing for report:**
> "Hopkins statistic (H=0.46) indicated low natural clustering tendency, consistent with
> a synthetically generated dataset. We proceeded with K-Means (highest silhouette)
> and GMM (most statistically principled for symmetric data), focusing interpretation
> on the forced partition structure rather than claiming discovery of natural archetypes."

---

### Algorithm Comparison Table (Section 10)

| Algorithm | Paradigm | Avg Silhouette | K Required | Outlier Robust | Soft Assignment |
|-----------|----------|---------------|------------|----------------|-----------------|
| K-Means | Centroid | ~0.055 | Yes | No | No |
| Hierarchical | Connectivity | ~0.010 | Yes | No | No |
| DBSCAN | Density | N/A (1 cluster) | No | Yes | No |
| GMM | Model-based | ~0.02–0.05 | Auto (BIC) | Partial | **Yes** |
| PAM | Centroid (robust) | ~0.037 | Yes | **Yes** | No |

Green highlight in the notebook = best silhouette score.

---

### Cluster Interpretation (Section 11)

Cross-tabs of K-Means clusters against:
- Ship Type, Route Type, Engine Type, Maintenance Status
- KPI boxplots: Efficiency, Cost, Revenue, Speed, Load %, Turnaround Time
- Provisional archetype names table (4 clusters)

### Stability Check (Section 12)

Pairwise Adjusted Rand Index across all 5 algorithm pairs + 3 agreement bar charts.
ARI interpretation: >0.6 = strong | 0.3–0.6 = moderate | <0.3 = weak

---

## 7. Project Naming

| Project | Recommended Report Title |
|---------|--------------------------|
| **Clustering** | "Segmenting the Fleet: An Unsupervised Analysis of Ship Operational Performance" |
| **Classifier** | "Predicting Ship Maintenance Risk: A Machine Learning Classification Approach" |

---

## 8. Plot Output — PNG vs Embedded HTML

Both Rmd files use `self_contained: true` (pandoc default). All plots are **base64-encoded
and embedded inside the HTML** — no separate PNG files are created.

**To get separate PNG files**, add to YAML:
```yaml
output:
  html_document:
    self_contained: false
```
This creates a `Ship_Clustering_files/figure-html/` folder with one PNG per chunk.

**To export a single plot manually:**
```r
ggsave("my_plot.png", plot = last_plot(), width = 10, height = 6, dpi = 300)
```

---

## 9. Next Step

Build the classifier notebook — `classification/Ship_Classifier.Rmd`:
- Target: `Maintenance_Status` (Good / Fair / Critical)
- Check class balance → apply SMOTE if Critical < 10%
- Algorithms: Random Forest, XGBoost, Multinomial Logistic Regression, KNN
- Evaluate with macro F1-score

---

*End of session log.*
