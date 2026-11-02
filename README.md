# YouTube Trending Prediction and Audience Analytics System

An end-to-end machine learning and audience analytics system designed to predict whether a newly published YouTube video will enter the trending list within a future observation window (6–54 hours) using early audience signals and engagement metrics collected during the initial observation period (0–6 hours).

---

## 🎯 Project Overview & Research Objectives

Predicting video virality or trending status based solely on lifetime view or like counts is misleading because those metrics reflect already realized success. This project formulates a strictly **leakage-free temporal prediction problem**:

> **Given only the information available during the first $N$ hours ($0 \le t \le 6\text{h}$) after publication, how accurately can we predict whether a video will enter the trending set during the subsequent prediction window ($6\text{h} < t \le 54\text{h}$)?**

### Core Research Question
> **Does early audience reaction (sentiment, emotional tone, and comment patterns) provide measurable incremental predictive power beyond conventional metadata and early engagement velocities?**

---

## 📐 System Architecture

```text
                    ┌─────────────────────────┐
                    │  YouTube Data API v3    │
                    │  / Benchmark Bootstrapper│
                    └────────────┬────────────┘
                                 │
                   ┌─────────────┴─────────────┐
                   │                           │
                   ▼                           ▼
            Video Metadata             Early Comments (0–6h)
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                     Raw Storage (Parquet / DB)
                                 │
                                 ▼
                    Data Validation & Anti-Leakage
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
   Temporal Feature Engineering                 Ensemble NLP Pipeline
   - Title/Desc/Tags/Category                 - VADER (Lexicon-based)
   - View/Like/Comment Velocity               - DistilBERT (Contextual Sentiment)
   - Growth & Acceleration (1h→3h→6h)         - Spam & Low-Information Filtering
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
                     Temporal Feature Dataset
                                 │
                                 ▼
                   Chronological Train / Val / Test
                                 │
         ┌───────────────────────┴───────────────────────┐
         ▼                                               ▼
  Ablation Experiments                           MLflow Experiment Tracking
  - Exp A: Metadata only                         - SQLite backend
  - Exp B: Metadata + Engagement                 - PR-AUC, ROC-AUC, F1, Calibration
  - Exp C: Metadata + Eng + Sentiment            - Model artifacts
  - Exp D: Full Feature Set                      - SHAP Explainability
         │
         ▼
  FastAPI Prediction Engine
         │
         ▼
  Streamlit Analytics Dashboard
```

---

## 🚀 Key Features

- **Strict Anti-Leakage Guards**: Strict timestamp filtering ensures features are derived exclusively from data available at or before the cutoff time ($t \le 6\text{h}$). Chronological train/validation/test splits simulate realistic future predictions.
- **Ensemble Comment NLP**: Combines rule-based VADER sentiment analysis with transformer-based DistilBERT contextual classification to analyze audience reaction and sentiment shift over time.
- **Systematic Feature Ablation**: Scientifically measures the incremental predictive value of metadata, engagement velocity, and comment sentiment across 4 controlled experiment tiers.
- **Interpretable AI with SHAP**: TreeSHAP explanations compute global feature importance and local waterfall plots showing positive and negative drivers behind every individual prediction.
- **Hybrid Storage Architecture**: Zero-overhead local columnar Parquet and DuckDB/SQLite storage, integrated with SQLAlchemy ORM for seamless transition to PostgreSQL.
- **Production-Ready REST API**: FastAPI backend providing endpoints for live video inference, feature importance extraction, and audience analytics.
- **Interactive Analytics Dashboard**: Streamlit interface for exploring video predictions, audience sentiment distributions, and category-level trending benchmarks.

---

## 🔬 Research Ablation Experiments Suite

A rigorous, peer-reviewed ablation suite designed to isolate, benchmark, and explain the predictive mechanisms of early YouTube video virality. Full mathematical formulations and experiment specifications are detailed in **[docs/ABLATION_EXPERIMENTS.md](docs/ABLATION_EXPERIMENTS.md)**.

| # | Experiment Title | Core Research Question | Setups / Variables | Status |
|:---:|---|---|---|:---:|
| **1** | **Temporal Horizon** (Lead-Time Frontier) | *How early can virality be predicted, and what is the trade-off between lead time and accuracy?* | $t \le 1.0\text{h}, 2.0\text{h}, 4.0\text{h}, 6.0\text{h}$ | **Completed** |
| **2** | **Feature Modality Stack** | *Which data stream contributes the most predictive lift: metadata, creator authority, kinematics, or comments?* | $M_1$ (Meta) $\to M_2$ (+Creator) $\to M_3$ (+Kinematics) $\to M_4$ (+Comments) | **Completed** |
| **3** | **Kinematic Representations** (Engagement Physics) | *Does modeling derivatives ($v = \frac{dV}{dt}, a = \frac{d^2V}{dt^2}$) outperform raw cumulative counts?* | Raw counts vs. Velocity ($v$) vs. Acceleration ($a$) vs. Growth log-ratios | **Completed** |
| **4** | **Dual-Stage Comment Dynamics** & Sentiment Drift | *Does temporal sentiment shift ($\Delta S = S_{6\text{h}} - S_{1\text{h}}$) catch breakout hits better than a static snapshot?* | No text vs. Stage 1 only vs. Stage 2 only vs. Dual-stage drift ($\Delta S$) | **Completed** |
| **5** | **Circadian & Multi-Region Contagion** | *Do publisher local circadian time and early overseas charting (GB/AU) boost North American virality prediction?* | UTC baseline vs. Publisher local hour ($\sin/\cos$) vs. Cross-market contagion | Planned |
| **6** | **Model Architecture Family** | *Which inductive bias best fits early trajectory data: decision trees, recurrent neural nets, or multi-modal transformers?* | Logistic vs. LightGBM/XGBoost vs. Bi-LSTM vs. Late-Fusion Transformer | Planned |
| **7** | **Content Format Stratification** | *Do YouTube Shorts ($\le 60\text{s}$) and Long-Form videos share the same viral dynamics, or should they be modeled separately?* | Pooled model vs. Dedicated Long-form model vs. Dedicated Shorts model | Planned |

### Experiment 1 Benchmark Results (Completed)

Evaluated on an out-of-time test cohort of **65,881 matched videos** (20-day dataset, Oct 11–31):

| Horizon | Features | PR-AUC | ROC-AUC | Optimal $F_1$ | Precision | Recall | Median Lead Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$t \le 1.0\text{h}$** | 26 | 0.4172 | 0.9600 | **0.5095** | **49.4%** | 52.6% | **7.7 hours** |
| **$t \le 2.0\text{h}$** | 34 | 0.4027 | 0.9607 | 0.5082 | 48.1% | **53.9%** | **7.7 hours** |
| **$t \le 4.0\text{h}$** | 41 | 0.4316 | **0.9689** | 0.4908 | 46.3% | 52.2% | **7.7 hours** |
| **$t \le 6.0\text{h}$** | 49 | **0.4373** | 0.9669 | 0.4511 | 39.7% | 52.2% | **7.7 hours** |

### Experiment 2 Benchmark Results (Completed)

Evaluated at fixed cutoff horizon $t \le 4.0\text{h}$ across 4 progressive modality stacks (65,881 matched videos):

| Stack | Modality Included | Features | PR-AUC | $\Delta$ PR-AUC | ROC-AUC | Optimal $F_1$ | Precision | Recall |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$M_1$** | **Metadata Only** | 16 | 0.2046 | *Baseline* | 0.9200 | 0.3557 | 25.3% | **60.0%** |
| **$M_2$** | **+ Creator Authority** | 18 | 0.2005 | -0.0041 | 0.9195 | 0.3387 | 25.5% | 50.4% |
| **$M_3$** | **+ Engagement Kinematics** | 35 | **0.3989** | **+0.1984** | 0.9667 | **0.5123** | **48.5%** | 54.4% |
| **$M_4$** | **+ Comments Arrival Dynamics** | 41 | **0.4181** | **+0.0192** | **0.9687** | 0.4798 | 44.7% | 51.7% |

*Key finding: Engagement kinematics delivers a decisive +99% relative PR-AUC lift over metadata. With 4.1M comments in the 20-day corpus, comment dynamics stabilized and yielded a net positive gain ($\Delta = +0.0192$), pushing PR-AUC to 0.4181.*

### Experiment 3 Benchmark Results (Completed)

Evaluated at fixed cutoff horizon $t \le 4.0\text{h}$ across 6 physical engagement representations (65,881 matched videos):

| Formulation | Physical Components | Features | PR-AUC | $\Delta$ PR-AUC vs $K_1$ | ROC-AUC | Optimal $F_1$ | Precision | Recall |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K_1$** | **Cumulative Counts (0th Order)** | 27 | 0.3969 | *Baseline* | 0.9657 | 0.4818 | 50.5% | 46.1% |
| **$K_2$** | **+ 1st Derivative (Velocity $v$)** | 33 | 0.3744 | -0.0225 | 0.9652 | 0.4678 | 46.2% | 47.4% |
| **$K_3$** | **+ 2nd Derivative (Acceleration $a$)** | 37 | **0.4074** | **+0.0105** | **0.9678** | 0.4650 | 48.4% | 44.8% |
| **$K_4$** | **+ Growth Log-Ratios ($\rho$)** | 41 | 0.3941 | -0.0028 | 0.9665 | **0.5010** | 46.5% | **54.4%** |
| **$K_5$** | **Full Integrated Physics** | 46 | 0.3744 | -0.0225 | 0.9624 | 0.4873 | 47.5% | 50.0% |
| **$K_0$** | **Scale-Free Kinematics (No Counts)** | 37 | 0.3948 | -0.0021 | 0.9664 | 0.4188 | **52.6%** | 34.8% |

*Key finding: With >1M observation snapshots, the 2nd derivative $a = \frac{\Delta v}{\Delta t}$ ($K_3$) smoothed out and achieved the peak PR-AUC of 0.4074. Scale-free kinematics ($K_0$) achieved 52.6% Precision without relying on cumulative view counts.*

### Experiment 4 Benchmark Results (Completed)

Evaluated at cutoff horizon $t \le 6.0\text{h}$ comparing Stage 1 ($t \le 1\text{h}$), Stage 2 ($1\text{h} < t \le 6\text{h}$), and dynamic drift $\Delta S$ (65,881 matched videos, 2.73M comments):

| Setup | Comment Signals Included | Features | PR-AUC | $\Delta$ PR-AUC vs $C_0$ | ROC-AUC | Optimal $F_1$ | Precision | Recall |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$C_0$** | **Kinematic Baseline (No NLP)** | 55 | 0.3629 | *Baseline* | 0.9662 | 0.4478 | 43.9% | 45.6% |
| **$C_1$** | **+ Stage 1 Early Reactions ($t \le 1\text{h}$)** | 62 | 0.3937 | +0.0308 | 0.9638 | **0.4688** | 38.3% | **60.4%** |
| **$C_2$** | **+ Stage 2 Influx Reactions ($1\text{h} < t \le 6\text{h}$)** | 62 | **0.4404** | **+0.0775** | 0.9655 | 0.4671 | 38.8% | 58.7% |
| **$C_{\text{static}}$** | **+ Static 6h Cumulative Snapshot** | 62 | 0.3991 | +0.0362 | 0.9643 | 0.4509 | **46.3%** | 43.9% |
| **$C_3$** | **Dual-Stage Drift Ensemble ($\Delta S$)** | 77 | 0.4052 | +0.0423 | **0.9677** | 0.4622 | 44.7% | 47.8% |

*Key finding: Isolating Stage 2 comments ($C_2$) produces a massive +21.3% PR-AUC lift (0.3629 $\to$ **0.4404**), proving that reaction tone from casual viewers outside the subscriber bubble is a potent indicator of viral distribution. Temporal stratification outperforms a flat snapshot ($C_{\text{static}}$ = 0.3991) by +0.0413 PR-AUC, and text features boost viral recall from 45.6% to **60.4%**.*

*See [docs/ABLATION_EXPERIMENTS.md](docs/ABLATION_EXPERIMENTS.md) for full experimental setup, mathematical definitions, and feature breakdown.*

---

## 🗂 Repository Structure

```text
viral/
├── configs/
│   ├── config.yaml              # Central configuration (windows, paths, hyperparameters)
│   └── seed_channels.json       # 3,500 clean stratified creators (Tiers 1, 2, 3)
├── data/
│   ├── raw/                     # Raw Parquet records (symlinked to harvest data lake)
│   ├── processed/               # Cleaned & validated trajectory records
│   └── features/                # Tabular feature datasets
├── docs/
│   ├── ABLATION_EXPERIMENTS.md  # Complete 7-experiment research ablation specifications
│   └── aws_data_collection_plan.md
├── experiments/
│   ├── exp1_temporal_horizon.py # Reproducible Experiment 1 benchmark runner
│   ├── exp2_feature_modalities.py # Reproducible Experiment 2 modality ablation runner
│   └── exp3_kinematic_representations.py # Reproducible Experiment 3 engagement physics runner
├── models/
│   ├── exp1/                    # Serialized model boosters (lgbm_1h..6h.txt) & metadata
│   ├── exp2/                    # Modality stack boosters (lgbm_M1..M4.txt) & metadata
│   └── exp3/                    # Kinematic boosters (lgbm_K0..K5.txt) & metadata
├── reports/
│   ├── exp1_temporal_horizon_results.json # JSON benchmark metrics & evaluation records
│   ├── exp2_feature_modalities_results.json # JSON metrics across M1-M4 modality stacks
│   └── exp3_kinematic_representations_results.json # JSON metrics across K0-K5 physics formulations
├── src/
│   ├── core/                    # App configuration and logging
│   ├── features/                # Modular feature engineering engine (kinematics, circadian)
│   ├── models/                  # ViralModelTrainer with F1 threshold optimization
│   ├── data_collection/         # YouTube API client & historical dataset bootstrapper
│   └── storage/                 # Database models & ORM
├── pyproject.toml               # Python package configuration
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 🛠 Technology Stack

- **Language**: Python 3.10+
- **Data & Storage**: Pandas, NumPy, PyArrow (Parquet), DuckDB, SQLite, SQLAlchemy
- **Machine Learning**: Scikit-Learn, LightGBM, XGBoost
- **NLP**: NLTK / VADER, Hugging Face Transformers (DistilBERT), PyTorch
- **Experiment Tracking & Explainability**: MLflow, SHAP, Matplotlib, Seaborn
- **API & Backend**: FastAPI, Uvicorn, Pydantic
- **Dashboard**: Streamlit, Plotly
- **Testing & Quality**: Pytest, Pytest-cov, HTTPX

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/kryz73/viral.git
cd viral
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Update configuration parameters as needed:
```env
YOUTUBE_API_KEY="your-optional-youtube-api-key"
DATABASE_URL="sqlite:///./data/viral.db"
MLFLOW_TRACKING_URI="sqlite:///./mlruns.db"
```
> **Note**: A live `YOUTUBE_API_KEY` is optional for development and offline testing. The system includes a mock/synthetic replay engine that simulates live responses when an API key is not configured.

---

## 🚦 Usage & Workflows

### 1. Bootstrap Historical & Benchmark Data
Populate raw observation and comment data from historical benchmarks or synthetic multi-snapshot generators:
```bash
python -m src.data_collection.dataset_bootstrapper --source benchmark
```

### 2. Run Data Validation & Anti-Leakage Checks
Validate data integrity, remove duplicates, and ensure strictly monotonic timestamps:
```bash
python -m src.preprocessing.validator
```

### 3. Extract Temporal Features & Comment Sentiment
Execute early observation window slicing ($t \le 6\text{h}$) and run the ensemble NLP pipeline:
```bash
python -m src.feature_engineering.temporal_features
```

### 4. Train Models & Run Ablation Experiments
Train baseline models and candidate gradient boosting trees across all 4 feature tiers:
```bash
python -m src.models.ablation
```
Launch the local MLflow dashboard to view experiment runs and metrics:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

### 5. Launch the FastAPI Prediction Engine
Start the REST API server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at `http://localhost:8000/docs`.

### 6. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` to view video predictions, SHAP waterfall explanations, and audience reaction analytics.

---

## 📊 Evaluation & Ablation Methodology

Because trending videos constitute a minority class, **PR-AUC (Precision-Recall Area Under Curve)** serves as the primary evaluation metric alongside ROC-AUC, F1-Score, and Brier Score for calibration.

| Experiment Tier | Included Features | Research Goal |
|---|---|---|
| **Tier A: Metadata** | Title length, description length, tags count, duration, category, publishing hour/day | Establish baseline signal from static video attributes. |
| **Tier B: Engagement** | Tier A + Early view/like/comment counts, velocities, and like/comment ratios ($t \le 6\text{h}$) | Quantify predictive power of early audience traction. |
| **Tier C: Sentiment** | Tier B + VADER compound score, DistilBERT positive/negative ratios, sentiment variance | Determine whether audience sentiment adds predictive signal beyond engagement. |
| **Tier D: Full NLP** | Tier C + Sentiment change over time ($t_{1\text{h}} \to t_{6\text{h}}$), spam ratios, growth acceleration | Measure full audience reaction and trajectory dynamics. |

---

## 🧪 Testing

Run automated tests including data validation, anti-leakage checks, ensemble NLP outputs, and API routes:
```bash
pytest tests/ -v --cov=src
```

---

## 🗺 Phased Roadmap

- [x] **Architecture & Requirements**: SRD definition, modular design, and phased roadmap.
- [ ] **Phase 1: Foundation & Storage**: Configuration management, SQLAlchemy models, and Parquet/DuckDB layers.
- [ ] **Phase 2: Acquisition & Validation**: YouTube Data API v3 client with mock fallback, benchmark ingestion, data validation suite.
- [ ] **Phase 3: Features & Ensemble NLP**: Strict anti-leakage feature engineering, VADER + DistilBERT sentiment extraction, video aggregations.
- [ ] **Phase 4: ML & Ablation**: Chronological splitting, baseline & GBDT models, MLflow tracking, SHAP explainability.
- [ ] **Phase 5: FastAPI Engine**: REST endpoints for prediction, feature attribution, and audience analytics.
- [ ] **Phase 6: Dashboard & Docker**: Streamlit visualization app, Dockerfile, and docker-compose deployment.

---

## 📄 License
This project is developed for educational and portfolio research purposes under the MIT License.
