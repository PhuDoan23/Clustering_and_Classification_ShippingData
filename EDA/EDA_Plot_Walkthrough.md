# EDA Plot Walkthrough — Ship Performance Dataset
### Clustering Analysis · Visualization Guide

This document explains every plot produced in `Ship_EDA.Rmd` — why each was chosen,
what it encodes, and what to look for when reading it.

---

## Table of Contents

1. [Univariate — Categorical Bar Charts](#1-univariate--categorical-bar-charts)
2. [Univariate — Numerical Histograms + Density](#2-univariate--numerical-histograms--density)
3. [Univariate — Boxplots (Outlier Detection)](#3-univariate--boxplots-outlier-detection)
4. [Bivariate — Speed vs Efficiency Scatter](#4-bivariate--speed-vs-efficiency-scatter)
5. [Bivariate — Operational Cost vs Revenue Scatter](#5-bivariate--operational-cost-vs-revenue-scatter)
6. [Bivariate — Engine Power vs Distance Scatter](#6-bivariate--engine-power-vs-distance-scatter)
7. [Bivariate — Cargo Weight vs Draft Scatter](#7-bivariate--cargo-weight-vs-draft-scatter)
8. [Bivariate — Efficiency by Ship Type Boxplot](#8-bivariate--efficiency-by-ship-type-boxplot)
9. [Bivariate — Operational Cost by Engine Type Boxplot](#9-bivariate--operational-cost-by-engine-type-boxplot)
10. [Bivariate — Revenue by Maintenance Status Boxplot](#10-bivariate--revenue-by-maintenance-status-boxplot)
11. [Bivariate — Turnaround Time by Route Type Boxplot](#11-bivariate--turnaround-time-by-route-type-boxplot)
12. [Bivariate — Speed by Weather Condition Ridge Plot](#12-bivariate--speed-by-weather-condition-ridge-plot)
13. [Bivariate — Correlation Matrix](#13-bivariate--correlation-matrix)
14. [Multivariate — Pairs Plot (GGally)](#14-multivariate--pairs-plot-ggally)
15. [Multivariate — Heatmap: Ship Type × Maintenance](#15-multivariate--heatmap-ship-type--maintenance)
16. [Multivariate — Bubble Chart: Cost vs Revenue vs Cargo](#16-multivariate--bubble-chart-cost-vs-revenue-vs-cargo)
17. [Multivariate — Faceted Ridge: Efficiency by Engine × Route](#17-multivariate--faceted-ridge-efficiency-by-engine--route)
18. [Multivariate — Faceted Scatter: Speed vs Efficiency × Maintenance](#18-multivariate--faceted-scatter-speed-vs-efficiency--maintenance)
19. [Multivariate — Parallel Coordinates Plot](#19-multivariate--parallel-coordinates-plot)
20. [Multivariate — PCA Scree Plot](#20-multivariate--pca-scree-plot)
21. [Multivariate — PCA Scatter (PC1 vs PC2)](#21-multivariate--pca-scatter-pc1-vs-pc2)
22. [Multivariate — PCA Loadings Biplot](#22-multivariate--pca-loadings-biplot)

---

## Section Overview

Before diving into individual plots, there are three analysis levels in this report:

| Level | Goal | Tools Used |
|-------|------|-----------|
| **Univariate** | Understand each feature in isolation — shape, spread, outliers | Bar charts, histograms, density curves, boxplots |
| **Bivariate** | Detect pairwise relationships between two features at a time | Scatter plots, grouped boxplots, ridge plots, correlation matrix |
| **Multivariate** | Reveal structure involving 3+ features simultaneously | Pairs plot, heatmap, bubble chart, parallel coordinates, PCA |

---

## 1. Univariate — Categorical Bar Charts

**Plot type:** Bar chart (count), arranged in a 2-column grid
**Features covered:** `Ship_Type`, `Route_Type`, `Engine_Type`, `Maintenance_Status`, `Weather_Condition`

### Why this plot?

Categorical features do not have a numeric distribution, so histograms do not apply.
A bar chart directly answers the most fundamental question: **how many observations fall
into each category?** This reveals whether categories are balanced or dominated by one class,
which has a direct impact on clustering — a severely imbalanced category will produce
clusters that merely reflect the dominant class rather than meaningful performance groups.

### What to look for

- **Imbalance**: If one bar is dramatically taller than others, that category will dominate
  any cluster shaped by that feature. For example, if 80% of ships are Container Ships,
  clusters may just separate Container Ships from everyone else.
- **"None" categories**: `Ship_Type` and `Route_Type` both contain a `None` value.
  This may represent missing data mislabelled as a category. Before clustering,
  decide whether to impute, drop, or keep these as a valid "unknown" group.
- **Maintenance balance**: `Maintenance_Status` uses a traffic-light colour scheme
  (green = Good, orange = Fair, red = Critical). If Critical is rare, it may form
  a small but important cluster on its own.
- **Weather spread**: Knowing whether Rough conditions are frequent or rare tells
  you how much weight the weather feature will carry during clustering.

---

## 2. Univariate — Numerical Histograms + Density

**Plot type:** Histogram overlaid with a density curve, 3-column grid (12 plots)
**Features covered:** All 12 core numerical features

### Why this plot?

A histogram divides the value range into equal-width bins and counts how many
observations fall in each. Overlaying a smooth density curve removes the binning
artefact and shows the true underlying shape of the distribution.

For clustering this is critical because:

- **K-Means assumes roughly spherical, similarly-scaled clusters.** Heavily skewed
  features violate this assumption and must be log-transformed or normalised before fitting.
- **Bimodal or multimodal distributions** (two or more humps) are the clearest visual
  signal that a single feature already separates the data into natural groups — strong
  evidence for a meaningful cluster boundary.

### What to look for

| Pattern | What it means for clustering |
|---------|------------------------------|
| **Symmetric / bell-shaped** | Feature is approximately normal — safe to use as-is after Z-score scaling |
| **Right-skewed (long tail to the right)** | Many low values, few very high values — consider log transformation (e.g., `Operational_Cost_USD`, `Revenue_per_Voyage_USD`) |
| **Bimodal (two humps)** | Two natural subgroups exist within this single feature — a strong clustering signal |
| **Uniform (flat)** | Values spread evenly — feature carries less discriminating power for clustering |
| **Spike at one value** | May indicate a default/coded value — investigate before using |

> **Key features to watch:** `Efficiency_nm_per_kWh` and `Cargo_Weight_tons` —
> check whether their distributions hint at distinct ship operating modes.

---

## 3. Univariate — Boxplots (Outlier Detection)

**Plot type:** Boxplot, 3-column grid (12 plots)
**Features covered:** All 12 core numerical features

### Why this plot?

A boxplot summarises a distribution in 5 numbers: minimum whisker, Q1 (25th percentile),
median, Q3 (75th percentile), maximum whisker. Points beyond the whiskers
(1.5 × IQR from Q1/Q3) are flagged as outliers in red.

Where histograms show shape, boxplots show **spread and extremes** more precisely.
For clustering this matters because:

- Outliers pull cluster centroids toward them in K-Means, distorting cluster shapes.
- The IQR (height of the box) tells you which features have high internal variability
  — these are typically the most informative for separating clusters.

### What to look for

- **Red dots far from the whiskers**: These are outlier voyages. Decide whether they
  represent genuine extreme events (storms, overloaded ships) worth keeping, or
  data entry errors worth removing.
- **Long boxes (large IQR)**: Wide variability — these features will drive cluster
  separation the most.
- **Short boxes with many outliers**: Distribution is tight but with a heavy tail.
  Log-scaling often helps here.
- **Median position within box**: If the median is near the bottom of the box,
  the distribution is right-skewed; if near the top, it is left-skewed.

---

## 4. Bivariate — Speed vs Efficiency Scatter

**Plot type:** Scatter plot with linear regression lines per group, coloured by `Engine_Type`
**Features:** `Speed_Over_Ground_knots` (x) · `Efficiency_nm_per_kWh` (y) · `Engine_Type` (colour)

### Why this plot?

Speed and fuel efficiency together define the **operational performance envelope** of a ship.
Physically, a ship moving faster burns more fuel per unit distance — so we expect a
negative relationship. Plotting by engine type reveals whether different propulsion
technologies follow the same speed-efficiency trade-off or occupy distinct performance zones.

If engine types separate into distinct bands on this plot, `Engine_Type` is a good
post-cluster label — we can later check whether our clusters align with these bands
even without telling the algorithm what engine type each ship has.

### What to look for

- **Slope direction**: Negative slope confirms the physics (faster = less efficient).
  Flat or positive slope is a data quality flag.
- **Colour separation**: If Diesel, HFO, and Steam Turbine points occupy visually
  distinct regions, engine type is a strong natural segmentation variable.
- **Fan shape (heteroscedasticity)**: If scatter increases at higher speeds, it means
  high-speed ships are more variable in their efficiency — possibly due to weather
  or cargo differences at those speeds.

---

## 5. Bivariate — Operational Cost vs Revenue Scatter

**Plot type:** Scatter plot with a 45-degree break-even reference line, coloured by `Ship_Type`
**Features:** `Operational_Cost_USD` (x) · `Revenue_per_Voyage_USD` (y) · `Ship_Type` (colour)

### Why this plot?

This is the **profitability diagnostic** plot. The dashed diagonal line is the break-even
line (Revenue = Cost). Points **above** the line are profitable voyages; points **below**
are loss-making. Clustering on these two features alone would naturally produce groups like:
"high-cost high-revenue", "low-cost low-revenue", "loss-making", "highly profitable".

Including both features in the clustering model (rather than using a single derived Profit
column) preserves the information about whether a voyage is expensive-but-successful
versus cheap-and-low-return.

### What to look for

- **How many points fall below the break-even line**: These are unprofitable voyages.
  If they form a dense cluster, that cluster represents a fleet problem worth flagging.
- **Colour separation by Ship Type**: If Container Ships consistently sit above the line
  while Bulk Carriers sit below, ship type is confounding economic performance.
- **Outliers at extreme right**: Very high-cost voyages. Are they also high-revenue
  (top-right corner) or loss-making (bottom-right corner)?
- **Vertical spread at fixed cost**: For a given operating cost level, do revenues vary
  widely? This indicates revenue predictability risk.

---

## 6. Bivariate — Engine Power vs Distance Scatter

**Plot type:** Scatter plot with regression lines per group, coloured by `Route_Type`
**Features:** `Engine_Power_kW` (x) · `Distance_Traveled_nm` (y) · `Route_Type` (colour)

### Why this plot?

Engine power and voyage distance represent the **scale of operations**. A transoceanic
route should require either higher engine power or more time. Plotting these together
reveals whether route type determines what combination of power and distance ships operate in.

This matters for clustering because if route type completely determines the
power-distance relationship, then those two features may be redundant with route type —
no need to encode the same information twice.

### What to look for

- **Colour-separated bands**: If Transoceanic routes cluster in the top-right (high power,
  high distance) and Coastal routes in the bottom-left, the features carry route information.
- **Slope per group**: A positive slope means more powerful engines travel further —
  intuitive for cargo ships. A near-zero slope means power and distance are unrelated
  within a route category.
- **Overlap between route types**: Heavy overlap suggests these two features are
  **not** capturing route differences — they are measuring something else and remain
  useful independent cluster signals.

---

## 7. Bivariate — Cargo Weight vs Draft Scatter

**Plot type:** Scatter plot with overall regression line, coloured by `Ship_Type`
**Features:** `Cargo_Weight_tons` (x) · `Draft_meters` (y) · `Ship_Type` (colour)

### Why this plot?

Draft (how deep the ship sits in the water) is physically determined by the total
weight the ship is carrying. A heavier cargo load should push the ship deeper, producing
a positive correlation between these two features. This plot is a **data sanity check**
as well as an exploratory one.

If the correlation is strong, `Draft_meters` and `Cargo_Weight_tons` may be redundant
for clustering. If the correlation is weak or absent, draft is measuring something
else (vessel type, ballast water, design differences) and both features carry
independent clustering value.

### What to look for

- **Positive slope**: Confirms the physical relationship — heavier cargo, deeper draft.
- **Scatter around the line**: Wide scatter means draft varies a lot even at the same
  cargo weight — ship size/type differences dominate.
- **Colour separation**: If Fish Carriers and Bulk Carriers sit in different draft ranges
  at the same cargo weight, vessel design is a confounding variable.
- **Correlation strength**: If R² from the regression line is high (points cluster tightly
  around the line), consider dropping one of the two features to reduce redundancy
  before clustering.

---

## 8. Bivariate — Efficiency by Ship Type Boxplot

**Plot type:** Horizontal boxplot with jittered points, ordered by median, coloured by `Ship_Type`
**Features:** `Efficiency_nm_per_kWh` (x) · `Ship_Type` (y)

### Why this plot?

While the scatter plot (Plot 4) shows the relationship between speed and efficiency,
this plot isolates efficiency **by ship type** without any confounding speed axis.
It answers: does the vessel type itself determine how fuel-efficient a voyage tends to be?

If ship types show clearly separated boxes, it confirms that `Ship_Type` is a natural
cluster descriptor — after we run clustering, we should expect different clusters to
correspond to different ship types.

### What to look for

- **Gap between boxes**: A clear vertical gap between ship types indicates efficiency
  is strongly determined by vessel type. Small gaps mean ship type alone does not
  explain efficiency variation.
- **Box width (IQR)**: Wide boxes mean high internal variability within a ship type —
  other factors (weather, route, maintenance) are also driving efficiency.
- **Jittered points**: Show the actual data density. If points pile up at extreme values,
  those are the voyages most worth investigating as candidate cluster anchors.
- **Ordering**: Boxes are sorted by median. The ordering tells you the efficiency ranking
  across ship types.

---

## 9. Bivariate — Operational Cost by Engine Type Boxplot

**Plot type:** Vertical boxplot with jittered points, coloured by `Engine_Type`
**Features:** `Operational_Cost_USD` (y) · `Engine_Type` (x)

### Why this plot?

Engine type is one of the most controllable design decisions for a ship. If engine
technology has a measurable effect on operating cost, it will be a key differentiator
in performance clusters. This plot tests whether Diesel, Heavy Fuel Oil, and Steam
Turbine engines lead to systematically different cost profiles.

### What to look for

- **Median differences**: Is there a clear cost ordering (e.g., Steam Turbine costs most,
  Diesel least)? A consistent ranking would make `Engine_Type` a meaningful cluster label.
- **Overlap in IQRs**: Extensive overlap between engine types means cost is driven more
  by voyage characteristics (distance, cargo) than by engine choice.
- **Outliers per engine type**: Are extreme cost outliers concentrated in one engine type?
  This could indicate maintenance issues specific to that technology.

---

## 10. Bivariate — Revenue by Maintenance Status Boxplot

**Plot type:** Vertical boxplot with jittered points, traffic-light colour scheme
**Features:** `Revenue_per_Voyage_USD` (y) · `Maintenance_Status` (x)

### Why this plot?

Maintenance status is an ordinal risk variable — ships in Critical condition are more
likely to break down, lose cargo, or be delayed. This plot tests the economic consequence
of poor maintenance: do vessels in worse condition earn less revenue?

If the answer is yes, maintenance status becomes not just a risk indicator but an
economic performance separator — critical information for defining cluster profiles.

### Why traffic-light colours?

Green (Good), orange (Fair), and red (Critical) map directly to the risk intuition.
This makes the ordinal relationship immediately legible without reading axis labels.

### What to look for

- **Downward trend from Good → Critical**: If median revenue falls as maintenance
  worsens, the ordinal encoding (0=Good, 1=Fair, 2=Critical) is valid and the feature
  is a strong cluster signal.
- **Overlap between statuses**: If Good and Critical boxes overlap substantially,
  maintenance does not drive revenue — other features dominate.
- **Variance by status**: If Critical ships show much higher variance than Good ships,
  it suggests that critical maintenance creates unpredictable outcomes — some voyages
  still succeed, others fail badly.

---

## 11. Bivariate — Turnaround Time by Route Type Boxplot

**Plot type:** Horizontal boxplot ordered by median, coloured by `Route_Type`
**Features:** `Turnaround_Time_hours` (x) · `Route_Type` (y)

### Why this plot?

Turnaround time (time spent in port between voyages) is a throughput efficiency metric.
A ship that spends many hours in port is less productive than one with a quick turnaround,
even if both voyages cover the same distance. Route type might drive turnaround time
because transoceanic routes involve larger ports with more complex loading operations.

### What to look for

- **Median ordering**: Do longer routes (Transoceanic, Long-haul) also have longer
  turnaround times? Or is turnaround independent of route scope?
- **Wide boxes on specific routes**: High variability in turnaround time for a route
  type indicates inconsistent port operations — an operational risk worth flagging.
- **"None" route type position**: Where does the undefined route category sit relative
  to the known ones? This helps decide whether to drop or encode those records.

---

## 12. Bivariate — Speed by Weather Condition Ridge Plot

**Plot type:** Ridge density plot (overlapping density curves per group)
**Features:** `Speed_Over_Ground_knots` (x) · `Weather_Condition` (y/group)

### Why a ridge plot instead of a boxplot?

A boxplot would show the median and spread, but it compresses the full shape of
the distribution into five numbers. A **ridge plot** (also called a joy plot) shows
the entire density curve for each category stacked vertically. This is ideal when
you want to compare full distributions — not just medians — across a small number
of groups.

### Why this pairing?

Ships sailing in rough weather are expected to slow down for safety. If the speed
distribution shifts leftward in Rough conditions compared to Moderate or Calm, it
confirms that `Weather_Condition` has a real physical effect on speed — making it
a valid and informative feature to include in the clustering model.

### What to look for

- **Left shift in Rough weather**: The Rough weather curve should peak at lower
  speeds than Moderate. If the curves are identical, weather does not affect speed
  in this dataset.
- **Curve overlap**: Heavy overlap means weather condition alone does not separate
  operational regimes — other features will need to carry the clustering signal.
- **Bimodal curves**: A double-humped speed distribution within a single weather
  condition could mean two distinct ship types are mixed together in that group.

---

## 13. Bivariate — Correlation Matrix

**Plot type:** Colour-coded upper-triangular matrix with correlation coefficients
**Features:** All 12 core numerical features

### Why this plot?

A correlation matrix computes the Pearson correlation coefficient (r) between every
pair of numerical features at once. It is the most efficient way to scan for:

1. **Redundant features** — pairs with |r| > 0.8 carry nearly identical information.
   Including both inflates the influence of that dimension during clustering.
2. **Feature independence** — pairs with |r| < 0.2 are uncorrelated and carry
   genuinely separate information. Including both is valuable for clustering.
3. **Unexpected relationships** — correlations that should not exist according to
   domain knowledge may signal data quality issues.

### Why hierarchical clustering order?

The features are reordered using hierarchical clustering of the correlation values.
This groups related features together visually, making blocks of correlated features
easier to spot than a random ordering.

### What to look for

- **Dark blue squares (r close to +1)**: Strong positive correlation — the two features
  rise and fall together. Candidate for feature reduction.
- **Dark red squares (r close to -1)**: Strong negative correlation — one rises as the
  other falls. Also a candidate for feature reduction.
- **White squares (r near 0)**: No linear relationship — both features are informative
  for clustering.
- **Blocks of correlated features**: A cluster of features in the matrix that are all
  positively correlated with each other suggests they measure the same underlying concept.

---

## 14. Multivariate — Pairs Plot (GGally)

**Plot type:** Scatterplot matrix (lower triangle), correlation coefficients (upper triangle), density curves (diagonal)
**Features:** `Speed`, `Efficiency`, `Engine_Power`, `Operational_Cost`, `Revenue`, `Load_Percentage` · coloured by `Ship_Type`

### Why this plot?

The pairs plot is the multivariate extension of the bivariate scatter plot. Instead of
one x-y pair, it shows **every combination** of the 6 selected features simultaneously,
all in one view. It is the single richest exploratory visualisation in this report.

The three panels serve different purposes:

| Panel | What it shows |
|-------|--------------|
| **Lower triangle (scatter)** | Raw point clouds for each feature pair — reveals shape of relationships and outliers |
| **Upper triangle (correlation)** | Pearson r per pair, coloured by ship type — quick numerical summary |
| **Diagonal (density)** | Distribution of each feature per ship type — combines univariate and categorical in one |

### Why these 6 features?

These 6 were chosen because they span the five key clustering dimensions:
operational performance (Speed, Efficiency), economic performance (Cost, Revenue),
capacity (Load %), and engine power. Together they capture the most important aspects
of a voyage's profile.

### What to look for

- **Colour separation in scatter panels**: If the colour (Ship Type) points form
  distinct clouds in several panels, ship type is a natural clustering variable.
- **Diagonal density differences**: If the density curves for different ship types
  are in completely different positions, that feature cleanly separates ship types.
- **Strong correlations that vary by colour**: If two features are correlated for one
  ship type but not another, the relationship is ship-type-dependent — important context
  for interpreting cluster results.

---

## 15. Multivariate — Heatmap: Ship Type × Maintenance

**Plot type:** Faceted tile heatmap (4 metrics × Ship Type × Maintenance Status grid)
**Features:** `Ship_Type` (y) · `Maintenance_Status` (x) · cell values = mean of `Efficiency`, `Cost`, `Revenue`, `Speed`

### Why this plot?

This heatmap answers a three-way question at once: **Does the interaction between
ship type and maintenance status produce systematically different performance profiles?**

A simple boxplot can show how maintenance status affects revenue (Plot 10), but it
cannot tell you whether the maintenance effect is the same for Container Ships and
Fish Carriers. This heatmap reveals **interaction effects** — situations where the
combination of two categorical variables produces an outcome that neither variable
predicts alone.

### Why use mean value as the cell colour?

The colour encodes the average performance metric for every Ship Type × Maintenance
combination. Darker cells (higher values on the plasma scale) represent better-performing
combinations; lighter cells represent worse ones.

### What to look for

- **Row patterns**: Do all ship types degrade equally from Good → Critical maintenance,
  or does one type (e.g., Bulk Carriers) degrade more sharply?
- **Column patterns**: Is one maintenance status consistently the best/worst across
  all ship types, or does the "best" maintenance status vary by vessel?
- **Bright cells in unexpected places**: A Critical-maintenance ship type outperforming
  a Good-maintenance ship type on revenue would be a surprising finding worth investigating.
- **Consistent gradients**: If the heatmap shows a smooth colour gradient from Good to
  Critical in every row, the ordinal encoding of maintenance is valid and the relationship
  is monotonic.

---

## 16. Multivariate — Bubble Chart: Cost vs Revenue vs Cargo

**Plot type:** Scatter plot where point size encodes a third variable
**Features:** `Operational_Cost_USD` (x) · `Revenue_per_Voyage_USD` (y) · `Cargo_Weight_tons` (size) · `Route_Type` (colour)

### Why this plot?

This extends Plot 5 (Cost vs Revenue) by adding two more dimensions: cargo weight
as bubble size and route type as colour. It encodes **four variables** in a single
two-dimensional view. The break-even dashed line is retained so profitability
remains readable.

The question this plot answers: **Is profitability driven by cargo load, or by
route type, or both?** If large bubbles (heavy cargo) consistently appear above
the break-even line, cargo weight is the primary profit driver. If colours cluster
above/below the line, route type matters more.

### What to look for

- **Large bubbles above the line**: Heavy-cargo voyages are profitable — cargo weight
  is a revenue driver.
- **Small bubbles below the line**: Light-cargo voyages lose money — unprofitable when
  not fully loaded.
- **Colour clustering above/below the line**: If Transoceanic routes consistently sit
  below the break-even line, those routes may not be economically viable regardless
  of cargo.
- **Bubble size independent of position**: If bubble size (cargo) is randomly scattered
  across profitable and unprofitable positions, cargo weight alone does not predict profit.

---

## 17. Multivariate — Faceted Ridge: Efficiency by Engine × Route

**Plot type:** Ridge density plots, faceted by Route Type (6 panels), grouped by Engine Type within each panel
**Features:** `Efficiency_nm_per_kWh` (x) · `Engine_Type` (y/group) · `Route_Type` (facet panel)

### Why this plot?

This extends Plot 12 (ridge plot) to a three-variable context. Route type and engine type
are both expected to influence efficiency, but their **interaction** is unknown. Perhaps
Diesel engines are most efficient on Short-haul routes but Steam Turbines dominate on
Transoceanic routes. This faceted ridge plot can reveal that.

### Why facet by Route rather than Engine?

Engine Type has only 3 levels and is the within-panel grouping variable (colour + y-axis).
Route Type has 5 levels and creates more visual panels — faceting by the variable with more
levels makes each panel cleaner.

### What to look for

- **Consistent engine ranking across facets**: If Diesel always has the highest efficiency
  regardless of route type, engine technology dominates. If the ranking flips between facets,
  there is a true interaction effect.
- **Panel-to-panel shape changes**: If the overall efficiency distribution shifts right on
  Transoceanic routes (more efficient), distance voyages are more fuel-optimal.
- **Narrowing of distributions in specific panels**: A tight distribution in one route–engine
  combination suggests very consistent operating conditions — potentially a stable cluster centre.

---

## 18. Multivariate — Faceted Scatter: Speed vs Efficiency × Maintenance

**Plot type:** Scatter with regression lines per ship type, faceted by Maintenance Status
**Features:** `Speed_Over_Ground_knots` (x) · `Efficiency_nm_per_kWh` (y) · `Ship_Type` (colour/regression) · `Maintenance_Status` (facet)

### Why this plot?

This revisits the core performance trade-off (speed vs efficiency from Plot 4) but splits
the view by maintenance status. The question: **Does poor maintenance change the
speed-efficiency relationship?** A ship in Critical condition may be slower at the same
engine output, or may consume more fuel for the same speed.

### What to look for

- **Regression line slope change across panels**: If the negative speed-efficiency slope
  is steeper in the Critical panel than the Good panel, poor maintenance amplifies the
  speed-efficiency trade-off.
- **Point cloud position shifting**: If the Critical panel's points sit lower (less
  efficient) at every speed, maintenance is a pure efficiency penalty.
- **Ship type colour separation consistency**: If Container Ships and Bulk Carriers are
  well-separated in the Good panel but mixed in the Critical panel, maintenance homogenises
  ship-type differences.

---

## 19. Multivariate — Parallel Coordinates Plot

**Plot type:** Multi-axis line chart where each axis is a normalised feature (0–1 scale)
**Features:** `Speed`, `Engine_Power`, `Distance`, `Cargo_Weight`, `Operational_Cost`, `Revenue`, `Efficiency`, `Load_Percentage` · coloured by `Ship_Type`

### Why this plot?

A parallel coordinates plot is specifically designed for high-dimensional data exploration.
Each vertical axis represents one feature. Each observation in the dataset is drawn as a
single line connecting its normalised value on each axis. Lines that follow similar paths
across all axes represent observations that are **similar in all dimensions simultaneously**
— which is exactly the clustering objective.

All features are normalised to 0–1 before plotting so that features with different units
(knots, USD, tonnes) can be compared on the same vertical scale.

### Why these 8 features?

These 8 represent the core numerical profile of a voyage: how fast, how powerful, how far,
how heavy, how costly, how profitable, how efficient, and how loaded. Together they capture
the full economic and operational character of each observation.

### What to look for

- **Line bundles (groups of parallel lines that follow the same path)**: These are natural
  clusters. If a group of lines all goes low-Speed → high-Efficiency → low-Cost, those
  voyages share a common operating mode.
- **Colour separation**: If all green lines (Container Ships) take one path and all blue
  lines (Fish Carriers) take another, ship type strongly determines the voyage profile.
- **Crossing lines**: Lines that cross between two adjacent axes mean the two features
  have a negative relationship for those observations — consistent crossings confirm
  negative correlation.
- **Mixed colour bundles**: If a colour bundle contains multiple ship types, those vessel
  types are operationally indistinguishable on these dimensions — clustering may merge them.

---

## 20. Multivariate — PCA Scree Plot

**Plot type:** Bar chart (per-component variance) + line chart (cumulative variance), with 80% threshold
**Features:** All 12 core numerical features (after Z-score standardisation)

### Why this plot?

Principal Component Analysis (PCA) is a mathematical technique that transforms the
original 12 correlated features into 12 uncorrelated components, ordered by how much
variance they explain. The scree plot shows:

- **Blue bars**: How much of the total variance each component captures.
- **Red line**: Cumulative variance — how much the first N components explain together.
- **Green dashed line at 80%**: The conventional threshold — the number of components
  needed to reach it is the recommended dimensionality for clustering.

### Why run PCA before clustering?

1. **Dimensionality check**: If 2 components explain 80% of variance, the data is
   essentially 2-dimensional despite having 12 features — clustering can be visualised.
2. **Redundancy detection**: Features that load heavily onto the same component are
   partially redundant. PCA reveals whether the 12 features carry 12 independent
   signals or just 4-5.
3. **Scaling validation**: Running PCA after Z-score standardisation confirms all
   features contribute equally, rather than high-magnitude features (USD costs)
   dominating.

### What to look for

- **Steep initial drop (elbow in bars)**: A sharp drop from PC1 to PC2 means one
  dimension dominates. A gradual decline means the data is genuinely high-dimensional.
- **Where the red line crosses 80%**: If it crosses at PC3 or PC4, the data can be
  reduced to 3-4 dimensions for clustering without major information loss.
- **Flat bars near the end**: Components adding less than 5% variance each are mostly
  noise — safe to discard.

---

## 21. Multivariate — PCA Scatter (PC1 vs PC2)

**Plot type:** 2D scatter plot of the first two principal components, coloured by Ship Type, shaped by Maintenance Status
**Features:** PC1 (x) · PC2 (y) · `Ship_Type` (colour) · `Maintenance_Status` (point shape)

### Why this plot?

The PCA scatter is the closest thing to a direct "pre-view" of what clustering will
find. It projects the 12-dimensional feature space onto the 2 dimensions that explain
the most variance. If natural clusters exist in the data, they often become visible as
separate point clouds in this 2D projection.

### Why colour by Ship Type and shape by Maintenance?

Both are categorical variables we expect to relate to performance. By mapping both
onto the PCA scatter simultaneously, we can see whether the natural groupings in the
data (the point clouds) align with ship type alone, maintenance alone, their combination,
or neither.

### What to look for

- **Distinct point clouds**: Separated groups visible in the scatter are the natural
  clusters the algorithm will discover.
- **Colour-separated clouds**: If blue points (e.g., Container Ships) form one cloud
  and green (Fish Carriers) form another, ship type aligns with the dominant variance
  dimensions — expected clusters will correspond to vessel types.
- **Shape patterns within a colour**: If triangles (Critical maintenance) are consistently
  in one part of a colour cloud, maintenance status adds a sub-cluster within vessel type.
- **Fully mixed scatter**: If all colours and shapes are uniformly distributed, the natural
  groupings are not along ship-type or maintenance lines — clustering will discover
  cross-cutting performance groups.

---

## 22. Multivariate — PCA Loadings Biplot

**Plot type:** Arrow diagram where each arrow represents a feature's contribution to PC1 and PC2
**Features:** All 12 core numerical features as arrows, coloured by arrow length (magnitude)

### Why this plot?

The loadings biplot shows **which original features drive each principal component**.
Each arrow points in the direction that the feature "pulls" observations in the PC1-PC2
space. Longer arrows (brighter colour on the plasma scale) contribute more to the
first two components.

This is the interpretation key for the PCA scatter (Plot 21): once you see clusters
in the scatter, the loadings biplot tells you **which features define those clusters**.

### Reading the arrows

| Arrow direction | Interpretation |
|-----------------|----------------|
| **Points right (+PC1)** | High values of this feature → observations pushed right on PC1 |
| **Points left (−PC1)** | High values → observations pushed left on PC1 |
| **Points up (+PC2)** | High values → observations pushed up on PC2 |
| **Arrows pointing same direction** | Features are positively correlated |
| **Arrows pointing opposite directions** | Features are negatively correlated |
| **Arrows at 90 degrees** | Features are uncorrelated |

### What to look for

- **Long arrows (bright yellow/orange)**: These features dominate the variance —
  they are the most important for defining clusters.
- **Bundles of arrows in the same direction**: These features are correlated and
  may represent a single underlying concept (e.g., all "economic performance" arrows
  pointing the same way).
- **Short arrows (dark)**: These features contribute little to the first two PCs.
  They may still matter for later PCs — but if most short-arrow features are in the same
  cluster, consider dropping the weakest.
- **Arrow directions vs. cluster positions in Plot 21**: If a cluster in Plot 21 sits in
  the top-right corner, look at which arrows point top-right in the loadings biplot — those
  are the features that characterise that cluster.

---

## Summary: Plot Selection Logic

| Analysis goal | Plot chosen | Why that plot type |
|---------------|-------------|-------------------|
| Count per category | Bar chart | Direct frequency encoding |
| Distribution shape | Histogram + density | Shows both binning and smoothed shape |
| Outlier detection | Boxplot | IQR and whisker logic is explicit |
| Two continuous features | Scatter + regression | Reveals direction and strength of relationship |
| Continuous vs categorical | Boxplot + jitter | Shows spread within each category; jitter prevents hiding of density |
| Full distribution vs categorical | Ridge plot | Preserves shape information lost in boxplot |
| All-pairs numeric | Correlation matrix | O(n²) pairings in compact colour grid |
| 6 features simultaneously | Pairs plot (GGally) | Every pair in one view; diagonal adds univariate |
| Two categoricals × metric | Heatmap | Interaction effects visible as colour patterns |
| 4 variables in 2D | Bubble chart | Third dimension as size; fourth as colour |
| Engine × Route interaction | Faceted ridge | Three-way comparison without 3D axis |
| Maintenance moderation | Faceted scatter | Separate panels for each moderating level |
| High-dimensional profile | Parallel coordinates | Each feature as its own axis; lines show profiles |
| Dimensionality structure | PCA scree | Explains variance by component — dimension selection |
| Pre-cluster structure | PCA scatter | 2D projection reveals natural groupings |
| Feature importance | PCA loadings | Arrow length/direction shows what drives variance |

---

*This walkthrough accompanies `Ship_EDA.Rmd`. Render that file to HTML to see all plots.*
