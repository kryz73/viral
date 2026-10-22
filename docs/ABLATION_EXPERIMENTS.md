# Research Ablation Experiments Suite

This document defines the mathematical formulation, experimental design, and evaluation protocols for the **7 Core Ablation Experiments** in the YouTube Virality and Trending Prediction System.

---

## 📑 Table of Contents
1. [Overview & Research Methodology](#1-overview--research-methodology)
2. [Ablation 1: Temporal Horizon (The Lead-Time Frontier)](#2-ablation-1-temporal-horizon-the-lead-time-frontier)
3. [Ablation 2: Feature Modality Contributions (The Multi-Modal Stack)](#3-ablation-2-feature-modality-contributions-the-multi-modal-stack)
4. [Ablation 3: Kinematic Representations (Engagement Physics)](#4-ablation-3-kinematic-representations-engagement-physics)
5. [Ablation 4: Dual-Stage Comment Dynamics & Sentiment Drift](#5-ablation-4-dual-stage-comment-dynamics--sentiment-drift)
6. [Ablation 5: Circadian & Multi-Region Contagion](#6-ablation-5-circadian--multi-region-contagion)
7. [Ablation 6: Model Family & Inductive Biases](#7-ablation-6-model-family--inductive-biases)
8. [Ablation 7: Content Format Stratification (Shorts vs. Long-Form)](#8-ablation-7-content-format-stratification-shorts-vs-long-form)
9. [Anti-Leakage Protocol & Chronological Splits](#9-anti-leakage-protocol--chronological-splits)

---

## 1. Overview & Research Methodology

Predicting video virality from lifetime metrics suffers from hindsight bias. To simulate true production deployment, every experiment adheres to three non-negotiable principles:

1. **Strict Temporal Horizon Cutoff ($t \le H$)**: For any evaluation at horizon $H \in \{1\text{h}, 2\text{h}, 4\text{h}, 6\text{h}\}$, models have access *only* to observations and comments recorded at or before $t = H$.
2. **Chronological Splitting (Out-of-Time Testing)**:
   - **Training Set (Oct 11 – 15)**: Learns model parameters and rolling creator baselines.
   - **Validation Set (Oct 16)**: Used exclusively for early stopping and decision threshold tuning ($F_1$-maximization).
   - **Test Set (Oct 17)**: Completely held-out future cohort for reporting metrics.
3. **Controlled Cohort Matching**: To prevent sample selection bias from distorting comparative results, experiments are evaluated on identical matched subsets of videos with valid observations across all compared horizons.

---

## 2. Ablation 1: Temporal Horizon (The Lead-Time Frontier)

### Core Question
> **How early can viral breakout be reliably predicted, and what is the trade-off between lead time and predictive accuracy?**

### Mathematical Formulation
Let $\mathcal{D}_H$ be the feature representation extracted using observations up to time $H$:
$$\hat{y} = f_{\theta}(\mathcal{D}_H), \quad H \in \{1.0\text{h}, 2.0\text{h}, 4.0\text{h}, 6.0\text{h}\}$$
The target label $y \in \{0, 1\}$ denotes whether the video appears on the YouTube Trending leaderboard ($US, CA, GB, AU$) within its 60-hour lifecycle.

### Benchmark Results (Completed on 20,234 Matched Videos)

| Horizon ($H$) | Features | PR-AUC | ROC-AUC | Optimal $F_1$ | Precision | Recall | Median Lead Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$t \le 1.0\text{h}$** | 26 | 0.3299 | 0.9574 | 0.4719 | 0.3750 | 0.6364 | **7.9 hours** |
| **$t \le 2.0\text{h}$** | 34 | 0.3467 | 0.9588 | 0.4485 | 0.3737 | 0.5606 | **7.9 hours** |
| **$t \le 4.0\text{h}$** | 41 | 0.3321 | 0.9604 | **0.5368** | **0.4113** | **0.7727** | **7.9 hours** |
| **$t \le 6.0\text{h}$** | 49 | **0.4015** | **0.9664** | 0.5051 | 0.3788 | 0.7576 | **7.9 hours** |

### Key Takeaways
- **1-Hour Viability**: ROC-AUC is already **0.9574** at just 60 minutes after publish, capturing 63.6% of viral videos.
- **PR-AUC Progression**: PR-AUC improves by **+21.7% relative** from 1h to 6h (0.33 $\to$ 0.40), with the largest jump occurring between 4h and 6h due to discussion maturation.
- **Optimal Operating Horizon**: $t = 4.0\text{h}$ achieves the highest $F_1$ score (**0.5368**) and recall (**77.3%**), providing an actionable ~4-hour advance warning before trending onset.

---

## 3. Ablation 2: Feature Modality Contributions (The Multi-Modal Stack)

### Core Question
> **Which data modality drives viral prediction the most: static packaging, creator authority, engagement kinematics, or audience comments?**

### Stacks Tested (Fixed at $t \le 4.0\text{h}$)
1. **Stack $M_1$ (Metadata Only — 18 features)**:
   - Video duration, `is_short` indicator, category ID.
   - Title length, word count, uppercase ratio, punctuation markers (`!`, `?`), tag count.
   - Definition (`hd`), captions, licensed content.
   - Circadian periodic encodings ($\sin/\cos$ of local publish hour and day of week).
2. **Stack $M_2$ (Metadata + Creator Authority Prior — 20 features)**:
   - All $M_1$ features.
   - Creator Tier: `tier_1_high_potential`, `tier_2_battleground`, `tier_3_breakout`, `trending_discovered`.
   - Historical upload cadence (rolling count of uploads prior to publish timestamp).
3. **Stack $M_3$ (Metadata + Creator + Engagement Kinematics — 35 features)**:
   - All $M_2$ features.
   - Log-transformed views, likes, and comments at $1\text{h}, 2\text{h}, 4\text{h}$.
   - Numerical view velocities: $v_1 = \frac{V_1}{\Delta t_1}, v_2 = \frac{V_2 - V_1}{\Delta t_{1\to 2}}, v_4 = \frac{V_4 - V_2}{\Delta t_{2\to 4}}$.
   - View acceleration: $a_{1\to 2} = \frac{v_2 - v_1}{\Delta t}, a_{2\to 4} = \frac{v_4 - v_2}{\Delta t}$.
   - Engagement ratios: $\frac{\text{likes}}{\text{views} + 1}, \frac{\text{comments}}{\text{views} + 1}$.
   - Growth multipliers: $\log_{10}\left(\frac{V_2 + 1}{V_1 + 1}\right), \log_{10}\left(\frac{V_4 + 1}{V_2 + 1}\right)$.
4. **Stack $M_4$ (Full Multi-Modal Stack — 41 features)**:
   - All $M_3$ features.
   - Cumulative comment arrivals at $1\text{h}, 2\text{h}, 4\text{h}$.
   - Comment arrival rate ($\text{comments} / \text{hour}$).

### Benchmark Results (Completed on 20,234 Matched Videos at $t \le 4.0\text{h}$)

| Stack | Modality Included | Features | PR-AUC | $\Delta$ PR-AUC | ROC-AUC | Optimal $F_1$ | Precision | Recall |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$M_1$** | **Metadata Only** | 16 | **0.2847** | *Baseline* | 0.9088 | 0.4352 | 33.1% | 63.6% |
| **$M_2$** | **+ Creator Authority** | 18 | **0.2810** | -0.0037 | 0.9117 | 0.4023 | 32.4% | 53.0% |
| **$M_3$** | **+ Engagement Kinematics** | 35 | **0.3941** | **+0.1131** | **0.9644** | **0.4859** | **38.7%** | **65.2%** |
| **$M_4$** | **+ Comments Arrival Dynamics** | 41 | **0.3315** | -0.0626 | 0.9626 | 0.4706 | 38.5% | 60.6% |

### Key Takeaways
1. **Packaging Prior ($M_1$)**: Static video metadata alone at $t=0$ achieves **ROC-AUC = 0.9088** and **PR-AUC = 0.2847** (driven by video duration, title capitalization, and publication circadian hour).
2. **Creator Prior ($M_2$)**: Creator tier and upload cadence provide minimal marginal lift ($\Delta \text{PR-AUC} = -0.0037$), showing that creator tier sets a general baseline but cannot predict *which specific upload* will go viral.
3. **Kinematics Dominance ($M_3$)**: Engagement kinematics ($v, a$, growth log-ratios) produces a massive **+40.2% relative lift in PR-AUC** (0.2810 $\to$ 0.3941), proving that early physical reaction velocity is the primary determinant of virality.
4. **Comment Count Redundancy ($M_4$)**: Adding raw comment volume on top of kinematic statistics dilutes model precision ($\text{PR-AUC} = 0.3315$). Audience comments require **NLP sentiment drift features** ($\Delta S$) rather than raw counts to add incremental value (tested in Ablation 4).

---

## 4. Ablation 3: Kinematic Representations (Engagement Physics)

### Core Question
> **Does modeling engagement as continuous derivatives ($v = \frac{dV}{dt}, a = \frac{d^2V}{dt^2}$) outperform raw cumulative counts?**

### Formulations Compared
1. **Raw Cumulative Counts**: Using raw $\text{views}_t, \text{likes}_t, \text{comments}_t$ directly.
2. **First Derivative (Velocity $v$)**:
   $$v(t_k) = \frac{V(t_k) - V(t_{k-1})}{t_k - t_{k-1}}$$
3. **Second Derivative (Acceleration $a$)**:
   $$a(t_k) = \frac{v(t_k) - v(t_{k-1})}{t_k - t_{k-1}}$$
4. **Relative Multiplicative Expansion (Log-Ratios)**:
   $$\rho(t_k) = \log_{10}\left(\frac{V(t_k) + 1}{V(t_{k-1}) + 1}\right)$$

### Evaluation Goal
Quantify whether physics-informed acceleration features help distinguish between a video with high views that has already peaked ($v > 0, a \ll 0$) versus an exponential viral ignition ($v > 0, a > 0$).

### Benchmark Results (Completed on 20,234 Matched Videos at $t \le 4.0\text{h}$)

| Formulation | Physical Components | Features | PR-AUC | $\Delta$ PR-AUC vs $K_1$ | ROC-AUC | Optimal $F_1$ | Precision | Recall | Brier Loss |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K_1$** | **Cumulative Counts (0th Order)** | 27 | 0.3641 | *Baseline* | 0.9628 | 0.4720 | **40.0%** | 57.6% | 0.0264 |
| **$K_2$** | **+ 1st Derivative (Velocity $v$)** | 33 | 0.3744 | **+0.0103** | **0.9664** | 0.4757 | 35.0% | **74.2%** | 0.0256 |
| **$K_3$** | **+ 2nd Derivative (Acceleration $a$)** | 37 | 0.3562 | -0.0079 | 0.9661 | 0.4713 | 38.0% | 62.1% | 0.0258 |
| **$K_4$** | **+ Growth Log-Ratios ($\rho$)** | 41 | 0.3564 | -0.0077 | 0.9623 | **0.5079** | 39.0% | 72.7% | 0.0255 |
| **$K_5$** | **Full Integrated Physics** | 46 | **0.3975** | **+0.0334** | 0.9658 | 0.4898 | 36.9% | 72.7% | **0.0249** |
| **$K_0$** | **Scale-Free Kinematics (No Counts)** | 37 | 0.3664 | **+0.0023** | 0.9644 | **0.5052** | 38.3% | **74.2%** | 0.0253 |

### Key Takeaways
1. **Velocity ($v$) Drives Early Recall (+16.6% Surge)**: Cumulative counts alone ($K_1$) miss 42.4% of breakout hits (Recall = 57.6%). Adding instantaneous velocities ($K_2$) propels recall to **74.2%**, proving that arrival speed catches viral breakout before view volume piles up.
2. **Multiplicative Log-Ratios ($\rho$) Stabilize Scale**: Relative growth log-ratios ($\rho = \Delta \log_{10} V$) stabilize growth across heterogeneous channel sizes, producing the highest balanced score (**$F_1 = 0.5079$** in $K_4$).
3. **Full Integrated Physics ($K_5$) Peaks at PR-AUC 0.3975**: Combining scale, derivatives, log-ratios, and engagement interaction density (`like_ratio`, `comment_ratio`) achieves the highest PR-AUC and lowest calibration loss.
4. **Scale-Free Discovery ($K_0 > K_1$)**: Pure kinematics with **zero raw cumulative counts** outperforms the cumulative counts model ($K_0$ $F_1 = 0.5052$ vs $K_1$ $F_1 = 0.4720$; Recall 74.2% vs 57.6%), proving that virality is a dynamic process of physical momentum rather than a static accumulation of views.

---

## 5. Ablation 4: Dual-Stage Comment Dynamics & Sentiment Drift

### Core Question
> **Does temporal audience sentiment drift ($\Delta S = S_{6\text{h}} - S_{1\text{h}}$) predict breakout virality better than a single static comment snapshot?**

### Background & Signal
- **Stage 1 ($t \le 1.0\text{h}$)**: Immediate audience excitement from loyal subscribers.
- **Stage 2 ($1.0\text{h} < t \le 6.0\text{h}$)**: Broader algorithmic discovery and general audience reaction.
- **Sentiment Drift**:
  $$\Delta S = S_{\text{Stage 2}} - S_{\text{Stage 1}}$$
  $$\Delta \text{Volume Ratio} = \frac{N_{\text{Stage 2}}}{N_{\text{Stage 1}} + 1}$$

### Experimental Tiers
1. **No Comment Text**: Pure tabular engagement metrics.
2. **Stage 1 Comments Only ($t \le 1\text{h}$)**: VADER/DistilBERT compound sentiment of early reactions.
3. **Stage 2 Comments Only ($t \le 6\text{h}$)**: Cumulative sentiment of matured reactions.
4. **Dual-Stage Drift Ensemble**: Stage 1 + Stage 2 + Sentiment Drift ($\Delta S$) + Volume Acceleration.

---

## 6. Ablation 5: Circadian & Multi-Region Contagion

### Core Question
> **Do publisher local circadian time and early international charting (GB/AU) significantly improve North American virality prediction?**

### Background
Comment arrival is heavily dominated by recency ($t - t_{\text{pub}}$), not audience location. However, publisher local time determines when notifications hit subscriber phones. Furthermore, videos from global creators often chart in Great Britain (`GB`) or Australia (`AU`) hours before entering the United States (`US`) chart due to timezone offsets.

### Features Evaluated
1. **Baseline**: Pure UTC timestamps (no circadian adjustments).
2. **+ Cyclical Publisher Local Hour**:
   $$h_{\sin} = \sin\left(\frac{2\pi \cdot h_{\text{local}}}{24}\right), \quad h_{\cos} = \cos\left(\frac{2\pi \cdot h_{\text{local}}}{24}\right)$$
3. **+ Cross-Region Contagion Indicators**:
   $$\text{charted\_overseas} = \mathbb{I}(\text{trended in GB or AU at } t \le 6\text{h})$$

---

## 7. Ablation 6: Model Family & Inductive Biases

### Core Question
> **Which inductive bias best models non-uniform temporal virality: linear baselines, decision trees, recurrent neural networks, or multi-modal transformers?**

### Architectures Evaluated
1. **Linear Baseline**: ElasticNet Regularized Logistic Regression.
2. **Tabular GBDT**: LightGBM and XGBoost with histogram-based splitting and class weighting.
3. **Temporal Recurrent Network**: Bi-directional LSTM / GRU processing variable-length checkpoint sequences $[(V_1, L_1, C_1), (V_2, L_2, C_2), \dots]$.
4. **Late Fusion Transformer**: Multi-modal network concatenating DistilBERT title/comment embeddings with a tabular MLP backbone.

---

## 8. Ablation 7: Content Format Stratification (Shorts vs. Long-Form)

### Core Question
> **Do YouTube Shorts ($\le 60\text{s}$) and Long-Form videos share the same viral dynamics, or must they be modeled separately?**

### Empirical Data Discovery
In our post-fix dataset of 30,152 clean videos:
- **Long-Form ($> 60\text{s}$)**: 18,258 videos $\to$ **756 trending (4.14% trending rate)**
- **YouTube Shorts ($\le 60\text{s}$)**: 11,894 videos $\to$ **63 trending (0.53% trending rate)**

Long-form videos have an **8x higher likelihood** of appearing on YouTube's classic trending page than Shorts because Shorts are predominantly distributed through the dedicated swipe feed.

### Experimental Setups
1. **Pooled Model**: Single unified model containing an `is_short` indicator feature.
2. **Stratified Long-Form Model**: Exclusively trained and tested on videos with $\text{duration} > 60\text{s}$.
3. **Stratified Shorts Model**: Exclusively trained and tested on videos with $\text{duration} \le 60\text{s}$.

---

## 9. Anti-Leakage Protocol & Chronological Splits

To guarantee that reported performance matches real-world utility, all experiments strictly enforce the following protocol:

1. **Category 25 Filter**: All News & Politics broadcast syndicate clips are removed to prevent non-viral spam from skewing class priors.
2. **Tracked-From-Birth Only**: Only videos discovered within 90 minutes of upload via 0-quota RSS monitoring are retained. Untracked videos added after already trending (`tracked_from_birth == False`) are strictly excluded.
3. **Maturity / Censoring Cutoff**: Videos must have been published at least 60 hours prior to the evaluation snapshot date to ensure their true trending status is fully observed.
4. **No Random Splits**: Random K-fold cross-validation leaks creator baseline popularity across time. Chronological splitting (Train $\to$ Validation $\to$ Test) is mandatory across all 7 experiments.
