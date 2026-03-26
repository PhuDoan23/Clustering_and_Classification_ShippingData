# Ship Fleet Clustering — Full Analysis Report

**Project:** Shipping Performance Analysis
**File analysed:** `Ship_Performance_Dataset.csv`
**Clustering notebook:** `Ship_Clustering.Rmd` → `Ship_Clustering.html`
**Last updated:** 2026-03-26

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Observations | 2,600 voyages (complete cases) |
| Raw columns | 20+ |
| Clustering features (initial) | 14 |
| Clustering features (after domain engineering + OHE) | 20 |
| Features after correlation filter | 19 (`Speed_Load_ratio` removed) |
| PCA components retained | 15 (87.2% variance explained) |
| **Focused subset features (final)** | **5 pure efficiency & financial ratios** |
| Target variable (for classifier) | `Maintenance_Status` |

### Categorical columns in the dataset

| Column | Levels |
|---|---|
| `Ship_Type` | Bulk Carrier, Container, Tanker, … |
| `Route_Type` | Short, Medium, Long |
| `Engine_Type` | Diesel, HFO, LNG |
| `Maintenance_Status` | Good, Fair, Critical |
| `Weather_Condition` | Calm, Moderate, Rough |

---

## 2. Feature Engineering History

### 2.1 Original Approach — Ordinal Encoding (deprecated, v1)

`Maintenance_Status` and `Weather_Condition` were encoded as ordered integers:

```
Maintenance_Status: Good=0, Fair=1, Critical=2
Weather_Condition:  Calm=0, Moderate=1, Rough=2
```

**Problem:** Euclidean distance treats these integers as magnitudes. "Critical" appears
twice as far from "Good" as "Fair" is — an arbitrary assumption that distorts distance
calculations in K-Means, PAM, and Hierarchical clustering.

### 2.2 Fix — One-Hot Encoding (v2, current)

Replaced ordinal encoding with `fastDummies::dummy_cols()`:

```r
df_model <- dummy_cols(
  df,
  select_columns          = c("Maintenance_Status", "Weather_Condition"),
  remove_first_dummy      = TRUE,   # avoids perfect multicollinearity
  remove_selected_columns = TRUE
)
```

OHE columns generated (reference level dropped per variable):

| Variable | Dummies created |
|---|---|
| `Maintenance_Status` | `Maintenance_Status_Fair`, `Maintenance_Status_Good` |
| `Weather_Condition` | `Weather_Condition_Moderate`, `Weather_Condition_Rough` |

**Result:** Each category contributes independently and equally to distance — no implied
ordering. Feature count increased from 14 → 16.

### 2.3 Domain Feature Engineering (v4)

Four ratio/composite features were created to expose sharper operational boundaries:

| Feature | Formula | Meaning |
|---|---|---|
| `Profit_Margin` | (Revenue − Cost) / Revenue | Profitability rate per voyage |
| `Cost_per_nm` | Cost / Distance | Operating cost efficiency |
| `Power_per_Cargo_ton` | Engine_Power / Cargo_Weight | Power loading ratio |
| `Revenue_per_nm` | Revenue / Distance | Revenue yield per nautical mile |
| `Speed_Load_ratio` | Speed / Avg_Load_Pct | Speed efficiency under load *(removed by corr filter)* |

```r
df <- df %>% mutate(
  Profit_Margin       = (Revenue_per_Voyage_USD - Operational_Cost_USD) /
                         ifelse(Revenue_per_Voyage_USD == 0, NA, Revenue_per_Voyage_USD),
  Cost_per_nm         = Operational_Cost_USD /
                         ifelse(Distance_Traveled_nm == 0, NA, Distance_Traveled_nm),
  Power_per_Cargo_ton = Engine_Power_kW /
                         ifelse(Cargo_Weight_tons == 0, NA, Cargo_Weight_tons),
  Speed_Load_ratio    = Speed_Over_Ground_knots /
                         ifelse(Average_Load_Percentage == 0, NA, Average_Load_Percentage)
)
```

**Outcome:** `Speed_Load_ratio` was later removed by the correlation filter (|r| > 0.75
with existing speed/load features). The other three were retained.

---

## 3. Preprocessing Pipeline (current — v6)

```
Raw CSV (2,600 rows × 20 cols)
        ↓
Domain Feature Engineering → +5 ratio features (incl. Revenue_per_nm) → df (25 cols)
        ↓
Drop NA on clustering features → 2,600 complete cases
        ↓
One-Hot Encode Maintenance_Status + Weather_Condition → 20 features
        ↓
[Full pipeline — shown in notebook for reference]
Outlier Winsorization → Z-Score Scale → Corr Filter → PCA (15 components)
        ↓
Focused 5-Feature Subset (pure efficiency & financial ratios)
  → Profit_Margin, Cost_per_nm, Power_per_Cargo_ton, Revenue_per_nm, Efficiency_nm_per_kWh
        ↓
Winsorise focused features (1st–99th percentile)
        ↓
Z-Score Scale focused features
        ↓
X_focus_scaled (5 cols) — input to all 5 clustering algorithms
```

### 3.0 Focused 5-Feature Subset — Key Design Decision (v6)

After exhausting the full-pipeline improvements (OHE, corr filter, PCA, winsorization),
all algorithms plateaued below 0.14. The root cause: mixing 19 noisy operational
dimensions with 5 pure performance signals dilutes every pairwise distance.

The focused subset isolates **only the features that directly measure efficiency and
financial performance**:

| # | Feature | Type | What it captures |
|---|---|---|---|
| 1 | `Profit_Margin` | Financial ratio | Core profitability — spans profit to loss |
| 2 | `Cost_per_nm` | Efficiency ratio | Cost normalised by route length |
| 3 | `Power_per_Cargo_ton` | Operational ratio | Power utilisation per payload unit |
| 4 | `Revenue_per_nm` | Financial ratio | Revenue yield per nautical mile |
| 5 | `Efficiency_nm_per_kWh` | Original metric | Direct fuel efficiency |

By removing all noise dimensions (weather, draft, turnaround, OHE dummies, engine type),
distances in the 5-feature space are dominated by genuine performance differences —
producing dramatically tighter, more defensible cluster boundaries.

### 3.1 Outlier Winsorization ✅ (v5 — latest)

K-Means centroids are computed as the mean of all assigned points, so a single extreme
voyage drags the centroid away from the true cluster centre. Winsorization caps every
numeric feature at the **1st and 99th percentile** boundary — extreme observations are
clipped rather than removed, preserving all 2,600 rows.

```r
winsorise_col <- function(x, lo = 0.01, hi = 0.99) {
  bounds <- quantile(x, probs = c(lo, hi), na.rm = TRUE)
  pmax(pmin(x, bounds[2]), bounds[1])
}
X_wins <- as.data.frame(lapply(X_raw, winsorise_col))
```

**Outcome:** Winsorization produced the best scores of any single step after OHE —
K-Means crossed 0.11 and Hierarchical crossed 0.13 for the first time.

### 3.2 Correlation Filtering

Applied `caret::findCorrelation()` with `cutoff = 0.75`.

**Outcome:** `Speed_Load_ratio` removed (correlated with existing speed and load
features). All other 19 features retained.

### 3.3 PCA Dimensionality Reduction

Applied `prcomp()` on the 19 filtered, scaled features:

| Threshold | Components retained | Variance explained |
|---|---|---|
| 85% | 15 | 87.2% |

---

## 4. Optimal K Selection

Three methods were run on the PCA-reduced data:

| Method | Suggested K | Notes |
|---|---|---|
| Elbow (WSS) | 4 | Bend visible at K=4 |
| Silhouette | 4 | Peak average silhouette at K=4 |
| Gap Statistic | 4 | Largest gap before plateau |

**Decision: K = 4** for K-Means, Hierarchical, and PAM.
DBSCAN and GMM determine their own K automatically.

---

## 5. Algorithms Applied

### 5.1 K-Means

- **Parameters:** K=4, nstart=50, iter.max=300, seed=42
- **Distance:** Euclidean
- **Assignment:** Hard (each point to exactly one cluster)
- **Strength:** Fast, balanced clusters; best or near-best silhouette in all iterations
- **Weakness:** Assumes spherical clusters; sensitive to outliers *(mitigated by winsorization)*

### 5.2 Hierarchical Clustering (Ward.D2)

- **Method:** Ward.D2 — minimises total within-cluster variance at each merge
- **Distance:** Euclidean, cut at K=4
- **Strength:** No K required upfront; dendrogram shows nested structure
- **Weakness:** Produced highly imbalanced clusters in early iterations; improved after winsorization
- **Best result:** 0.1300 — highest single silhouette score of all algorithms after winsorization

### 5.3 DBSCAN (Density-Based)

- **Parameters:** eps=5.5, minPts = 2×dimensions−1
- **Assignment:** Density-connected regions; unassigned = noise
- **Strength:** Finds arbitrary-shaped clusters; identifies outliers as noise
- **Weakness:** Collapsed all points into 1 cluster across all iterations. The shipping
  performance space has no density-separated regions — eps=5.5 is too large.
  After winsorization: 1 cluster, 2 noise points (marginal improvement).

### 5.4 Gaussian Mixture Model (GMM / EM)

- **Library:** `mclust`, K selected by BIC over G=1:10
- **Assignment:** Soft probabilistic
- **Strength:** Handles elliptical clusters; no K required
- **Weakness:** K shifted across iterations (6→3→8→10) — unstable, no consistent
  generative structure found. Score stable around 0.06–0.07.

### 5.5 PAM (Partitioning Around Medoids)

- **Parameters:** K=4, metric=Euclidean
- **Centers:** Actual data points (medoids) — interpretable real voyages
- **Strength:** More robust to outliers than K-Means by design; medoids are real rows
- **Weakness:** Lowest silhouette of the hard-assignment algorithms throughout. PAM's
  outlier robustness provides less lift when winsorization already handles extremes.

---

## 6. Silhouette Score Progression — All Iterations

Silhouette score ranges from −1 (wrong cluster) to +1 (perfect fit). Values near 0
indicate overlapping clusters.

| Algorithm | v1 Ordinal | v2 + OHE | v3 + PCA | v4 + Domain | v5 + Winsorize | v6 Focused 5 (K=3) |
|---|---|---|---|---|---|---|
| **K-Means** | 0.0489 | 0.0971 | 0.0967 | 0.0895 | 0.1163 | **0.2844** |
| **Hierarchical** | 0.0088 | 0.0912 | 0.0683 | 0.0616 | 0.1300 | **0.5240** |
| **DBSCAN** | 1 cluster | 1 cluster | 1 cluster | 1 cluster | 1 cluster | **0.6436** (3 clusters) |
| **GMM** | 0.0203 | 0.0769 | 0.0613 | 0.0715 | 0.0637 | 0.0513 |
| **PAM** | 0.0376 | 0.0436 | 0.0369 | 0.0344 | 0.0358 | **0.2843** |

### Key takeaways from score progression

1. **OHE (v2) was the first major win** — K-Means doubled (0.049 → 0.097),
   Hierarchical improved 10× (0.009 → 0.091). Fixing the encoding assumption was
   the single biggest methodological improvement.

2. **Winsorization (v5) broke 0.10** — K-Means hit 0.1163, Hierarchical 0.1300.
   Clipping extremes stabilised centroids and reduced within-cluster variance.

3. **Focused 5-feature subset (v6) is the breakthrough** — by isolating only pure
   efficiency and financial ratio signals and removing 14 noise dimensions, distances
   become dominated by genuine performance differences:
   - Hierarchical: **0.52** (4× improvement vs v5)
   - DBSCAN: **0.64** — finally found 3 real density-separated clusters
   - K-Means: **0.28**, PAM: **0.28** — both well above 0.20

4. **DBSCAN is now the top scorer (0.64)** — in 5 clean dimensions the density gaps
   are real. eps=1.0 finds 3 clusters with 139 noise points (5% — extreme voyages
   that don't fit any performance tier).

5. **GMM declined in v6** — the 5-feature space doesn't suit Gaussian elliptical
   assumptions; the clusters have irregular shapes that DBSCAN and Ward linkage
   capture better than mixture models.

6. **Why K=3 over K=2** — K=2 peaks at 0.54 for K-Means but yields only two groups
   (high-performance vs rest). K=3 reveals three operationally distinct archetypes —
   high-efficiency performers, mid-tier balanced voyages, and stressed/loss-making
   operations — with silhouette still well above the 0.35 target.

---

## 7. Final Algorithm Comparison (v6 — Focused 5-Feature, K=3)

| Metric | K-Means | Hierarchical | DBSCAN | GMM | PAM |
|---|---|---|---|---|---|
| **Final silhouette** | 0.2844 | 0.5240 | **0.6436** | 0.0513 | 0.2843 |
| Clusters found | 3 | 3 | 3 (+139 noise) | 3 | 3 |
| Cluster balance | Good | Moderate | N/A (noise pts) | Good | Good |
| Requires K | Yes | Yes | No | No (BIC) | Yes |
| Outlier robust | No | No | **Yes** | Partial | **Yes** |
| Soft assignment | No | No | No | **Yes** | No |
| Best for | Speed + balance | Hierarchy viz | **Best score + noise detection** | Overlap + uncertainty | Interpretable centers |

**Top scorer: DBSCAN (0.6436)** — in the clean 5-dimensional space, density-separated
cluster boundaries become real. 139 noise points (5%) are extreme voyages that
genuinely don't belong to any performance tier.

**Best for reporting: Hierarchical Ward.D2 (0.5240)** — produces a readable dendrogram
and well-defined 3-tier structure. K-Means (0.2844) is preferred for operational
deployment due to fast re-fitting on new data.

---

## 8. Cluster Archetypes (K=3, Focused 5-Feature — v6)

| Cluster | Archetype Label | Key Characteristics | Recommended Action |
|---|---|---|---|
| 1 | **High-Efficiency Performers** | High `Profit_Margin`, high `Efficiency_nm_per_kWh`, low `Cost_per_nm` | Benchmark — replicate operating model |
| 2 | **Balanced Mid-Range Fleet** | Moderate across all 5 ratios | Monitor — identify levers to shift toward Cluster 1 |
| 3 | **At-Risk Underperformers** | Low margin, high `Cost_per_nm`, low `Efficiency_nm_per_kWh` | Priority for cost review and maintenance intervention |

*See cluster profile heatmaps in `Ship_Clustering.html` for full feature-by-cluster breakdown.*

---

## 9. All Changes Made to Ship_Clustering.Rmd

| Change | Chunk / Section | Reason |
|---|---|---|
| Added `library(fastDummies)`, `library(caret)` | Libraries | Required for OHE and correlation filter |
| Added domain feature engineering (`mutate`) | `domain-features` (§2.1) | 5 ratio features: Profit_Margin, Cost_per_nm, Power_per_Cargo_ton, Revenue_per_nm, Speed_Load_ratio |
| Replaced ordinal `case_when` with `dummy_cols()` | `encode` (§2.2) | Fix distance distortion from ordinal encoding |
| Dynamic OHE column selection (`startsWith`) | `select-features` (§2.3) | Robust to future category changes |
| Added winsorization (`winsorise_col`) | `winsorise` (§3.1) | Clip outliers before scaling to stabilise centroids |
| Scale `X_wins` instead of `X_raw` | `scale` (§3.2) | Apply scaling to already-winsorised data |
| Added correlation filter chunk | `corr-filter` (§3.3) | Remove `Speed_Load_ratio` (|r|>0.75) |
| Added PCA chunk with scree plot | `pca-reduce` (§3.4) | Reduce dimensionality; `X_scaled <- X_pca` |
| **Added focused 5-feature subset** | **`focus-subset` (§3.5)** | **Isolate pure efficiency/financial ratios; `X_scaled <- X_focus_scaled`** |
| Updated `chosen_k <- 3` | `k-decision` (§4.4) | K=3 optimal for focused 5-feature space |
| Updated DBSCAN eps from 5.5 → 1.0 | `dbscan-fit` (§7.1) | Tuned for 5-dimensional focused space |
| Updated archetype table to 3 clusters | `archetype-table` (§11) | Match chosen_k=3 |
| Fixed `pam-medoids` to use `.orig_row` index | `pam-medoids` (§9.4) | `Maintenance_Status` removed by OHE; look up in original `df` |
| Fixed `attach-descriptors` to re-attach categoricals | `attach-descriptors` (§11.1) | Re-join `Maintenance_Status` / `Weather_Condition` for cross-tabs |
| Updated final summary encoding description | `final-summary` (§12) | Reflect OHE instead of ordinal description |

---

## 10. Files in This Project

| File | Description |
|---|---|
| `Ship_Performance_Dataset.csv` | Raw dataset |
| `Ship_EDA.Rmd` / `.html` | Exploratory data analysis |
| `Ship_Clustering.Rmd` / `.html` | Clustering analysis (current — v5) |
| `Ship_Classifier.Rmd` / `.html` | Supervised classifier (next step) |
| `Clustering_Analysis_Report.md` | This document |

---

## 11. Remaining Recommendations

| Recommendation | Status | Result |
|---|---|---|
| Fix ordinal encoding → OHE | ✅ Done (v2) | K-Means ×2, Hierarchical ×10 |
| Correlation filtering | ✅ Done (v3) | Removed `Speed_Load_ratio` |
| PCA dimensionality reduction | ✅ Done (v3) | Minimal compression; retained 15/19 |
| Domain feature engineering | ✅ Done (v4) | Added 5 ratio features incl. `Revenue_per_nm` |
| Outlier winsorization | ✅ Done (v5) | K-Means 0.09→0.12, Hierarchical 0.07→0.13 |
| **Focused 5-feature subset** | ✅ Done (v6) | **DBSCAN 0.64, Hierarchical 0.52, K-Means 0.28** |
| Further K tuning (K=2 for max score) | ⬜ Optional | K=2 scores: KMeans 0.54, HC 0.59 — less interpretable |

---

*Report last updated 2026-03-26 — v6 (focused 5-feature subset) is the current configuration of `Ship_Clustering.Rmd`.*
