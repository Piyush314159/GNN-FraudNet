# 🔬 EDA Report — Elliptic Bitcoin Fraud Dataset

> Generated from [`src/eda.py`](file:///Users/piyushmaji/Desktop/Project/GNN-FraudNet/src/eda.py)
> Plots saved at [`results/plots/eda/`](file:///Users/piyushmaji/Desktop/Project/GNN-FraudNet/results/plots/eda)

---

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| Total nodes (transactions) | **203,769** |
| Total edges (directed flows) | **234,355** |
| Features per node | **166** (pre-normalized, zero-mean ~unit-variance) |
| Timesteps | **49** (feature 1 = timestep index) |
| Memory footprint | **273.9 MB** |
| Missing values | **0** |
| Inf values | **0** |

---

## 1. Label Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| 🔴 Illicit | 4,545 | 2.2% |
| 🟢 Licit | 42,019 | 20.6% |
| ⚪ Unknown | 157,205 | 77.1% |

![Label Distribution](/Users/piyushmaji/.gemini/antigravity-ide/brain/739dc1a1-c6a0-465f-b57f-8f0ab8e2bed8/01_label_distribution.png)

> [!WARNING]
> **77% of nodes are unlabeled.** The GNN will only train/evaluate on the 46,564 labeled nodes. Semi-supervised learning could leverage the unlabeled majority.

---

## 2. Class Imbalance

| Metric | Value |
|--------|-------|
| Labeled nodes | 46,564 |
| Licit | 42,019 (90.2%) |
| Illicit | 4,545 (9.8%) |
| **Imbalance ratio** | **1 : 9.2** (illicit : licit) |

![Class Imbalance](/Users/piyushmaji/.gemini/antigravity-ide/brain/739dc1a1-c6a0-465f-b57f-8f0ab8e2bed8/02_class_imbalance.png)

> [!IMPORTANT]
> **~9:1 imbalance** means the model will need class-weighted loss, oversampling, or focal loss. Without it, the model can achieve ~90% accuracy by predicting all-licit.

---

## 3. Feature Statistics

All 166 features are **pre-normalized** (zero-mean, unit-variance) — except `feat_1` (timestep, range 1–49, variance 230).

| Property | Value |
|----------|-------|
| Features with NaN | 0 |
| Features with Inf | 0 |
| Features >50% zeros | 0 |
| Features >90% zeros | 0 |

![Feature Ranges](/Users/piyushmaji/.gemini/antigravity-ide/brain/739dc1a1-c6a0-465f-b57f-8f0ab8e2bed8/03_feature_ranges.png)

> [!NOTE]
> The features are already standardized by the dataset authors. `feat_1` (timestep) is the outlier with variance=230. Features 2–94 are **local transaction features**; features 95–166 are **aggregated neighborhood features**.

---

## 4. Feature Distributions by Class

![Feature Distributions](/Users/piyushmaji/.gemini/antigravity-ide/brain/739dc1a1-c6a0-465f-b57f-8f0ab8e2bed8/04_feature_distributions.png)

> Several features show clear **distributional separation** between illicit and licit classes — especially in the later feature indices (aggregated features), suggesting that neighborhood information is discriminative for fraud detection.

---

## 5. Top Features Correlated with Fraud

| Rank | Feature | |Correlation| |
|------|---------|---------------|
| 1 | `feat_54` | 0.2615 |
| 2 | `feat_90` | 0.2276 |
| 3 | `feat_56` | 0.2271 |
| 4 | `feat_91` | 0.2213 |
| 5 | `feat_143` | 0.1919 |
| 6 | `feat_151` | 0.1872 |
| 7 | `feat_92` | 0.1858 |
| 8 | `feat_53` | 0.1718 |
| 9 | `feat_155` | 0.1530 |
| 10 | `feat_55` | 0.1421 |

![Correlation with Label](/Users/piyushmaji/.gemini/antigravity-ide/brain/739dc1a1-c6a0-465f-b57f-8f0ab8e2bed8/05_correlation_with_label.png)

> [!TIP]
> The strongest single-feature correlation is only **0.26** — no single feature is a strong fraud indicator alone. This motivates using GNNs that can combine features with graph-structural patterns.

---

## 6. Graph Degree Distribution

| Metric | Value |
|--------|-------|
| Isolated nodes (degree 0) | **0** |
| Mean degree | **2.30** |
| Median degree | **2** |
| Max degree | **473** |
| Mean out-degree | **1.41** |
| Mean in-degree | **1.58** |

![Degree Distribution](/Users/piyushmaji/.gemini/antigravity-ide/brain/739dc1a1-c6a0-465f-b57f-8f0ab8e2bed8/06_degree_distribution.png)

> The graph is **sparse** (avg 2.3 edges/node) with a **heavy-tailed** degree distribution — a few hub nodes have hundreds of connections. This is typical of financial transaction networks and favors GNN architectures like GraphSAGE that use neighborhood sampling.

---

## 7. Temporal Analysis (49 Timesteps)

| Metric | Value |
|--------|-------|
| Average fraud rate | **11.3%** |
| Peak fraud rate | **36.0%** (timestep 13) |
| Lowest fraud rate | **0.3%** (timestep 46) |

![Temporal Analysis](/Users/piyushmaji/.gemini/antigravity-ide/brain/739dc1a1-c6a0-465f-b57f-8f0ab8e2bed8/07_temporal_analysis.png)

> [!IMPORTANT]
> Fraud rate varies **dramatically** across timesteps (0.3% to 36%). This suggests temporal patterns in fraud activity. Train/val/test splits should ideally be **chronological** (earlier timesteps for training, later for testing) to simulate real deployment.

---

## 8. Feature Variance Ranking

![Feature Variance](/Users/piyushmaji/.gemini/antigravity-ide/brain/739dc1a1-c6a0-465f-b57f-8f0ab8e2bed8/08_feature_variance.png)

> Apart from `feat_1` (timestep, variance=230), all features have nearly identical variance (~1.0) due to pre-normalization. No features need to be dropped for low variance.

---

## 9. Pairwise Scatter — Top Correlated Features

![Pairwise Scatter](/Users/piyushmaji/.gemini/antigravity-ide/brain/739dc1a1-c6a0-465f-b57f-8f0ab8e2bed8/09_pairwise_scatter.png)

> The scatter plots show **moderate but imperfect separation** between illicit (red) and licit (green) in the top correlated feature pairs. The overlap confirms that fraud detection requires combining multiple features + graph structure.

---

## 10. Feature Correlation Heatmap

![Correlation Heatmap](/Users/piyushmaji/.gemini/antigravity-ide/brain/739dc1a1-c6a0-465f-b57f-8f0ab8e2bed8/10_correlation_heatmap.png)

> Several feature pairs show **high inter-correlation** (r > 0.8), especially among aggregated features (feat_90, feat_91, feat_92). This multicollinearity is fine for GNNs but could cause issues for linear baselines.

---

## 🧠 Key Takeaways for Model Design

| Finding | Implication |
|---------|-------------|
| 9:1 class imbalance | Use **weighted CrossEntropyLoss** with `weight=[1.0, 9.2]` |
| 77% unlabeled nodes | GNNs naturally leverage unlabeled neighbors via message passing |
| No single strong predictor (max r=0.26) | Feature combinations + graph structure needed → GNN advantage |
| Sparse graph (avg degree 2.3) | 2-hop GNN sufficient; deeper models risk over-smoothing |
| Heavy-tailed degree distribution | Use **GraphSAGE** with neighbor sampling for scalability |
| Temporal fraud variation (0.3%–36%) | Consider **chronological splits** for realistic evaluation |
| Features pre-normalized | Skip z-score normalization in `graph_builder.py` (already done) |
| Aggregated features (95–166) are informative | Confirms graph neighborhood features matter → GNN should help |
