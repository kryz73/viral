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
| **3** | **Kinematic Representations** (Engagement Physics) | *Does modeling derivatives ($v = \frac{dV}{dt}, a = \frac{d^2V}{dt^2}$) outperform raw cumulative counts?* | Raw counts vs. Velocity ($v$) vs. Acceleration ($a$) vs. Growth log-ratios | Planned |
| **4** | **Dual-Stage Comment Dynamics** & Sentiment Drift | *Does temporal sentiment shift ($\Delta S = S_{6\text{h}} - S_{1\text{h}}$) catch breakout hits better than a static snapshot?* | No text vs. Stage 1 only vs. Stage 2 only vs. Dual-stage drift ($\Delta S$) | Planned |
| **5** | **Circadian & Multi-Region Contagion** | *Do publisher local circadian time and early overseas charting (GB/AU) boost North American virality prediction?* | UTC baseline vs. Publisher local hour ($\sin/\cos$) vs. Cross-market contagion | Planned |
| **6** | **Model Architecture Family** | *Which inductive bias best fits early trajectory data: decision trees, recurrent neural nets, or multi-modal transformers?* | Logistic vs. LightGBM/XGBoost vs. Bi-LSTM vs. Late-Fusion Transformer | Planned |
| **7** | **Content Format Stratification** | *Do YouTube Shorts ($\le 60\text{s}$) and Long-Form videos share the same viral dynamics, or should they be modeled separately?* | Pooled model vs. Dedicated Long-form model vs. Dedicated Shorts model | Planned |

### Experiment 1 Benchmark Results (Completed)

Evaluated on an out-of-time test cohort of **20,234 matched videos** (post-fix clean cohort):

| Horizon | Features | PR-AUC | ROC-AUC | Optimal $F_1$ | Precision | Recall | Median Lead Time |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$t \le 1.0\text{h}$** | 26 | 0.3299 | 0.9574 | 0.4719 | 37.5% | 63.6% | **7.9 hours** |
| **$t \le 2.0\text{h}$** | 34 | 0.3467 | 0.9588 | 0.4485 | 37.4% | 56.1% | **7.9 hours** |
| **$t \le 4.0\text{h}$** | 41 | 0.3321 | 0.9604 | **0.5368** | **41.1%** | **77.3%** | **7.9 hours** |
| **$t \le 6.0\text{h}$** | 49 | **0.4015** | **0.9664** | 0.5051 | 37.9% | 75.8% | **7.9 hours** |

### Experiment 2 Benchmark Results (Completed)

Evaluated at fixed cutoff horizon $t \le 4.0\text{h}$ across 4 progressive modality stacks:

| Stack | Modality Included | Features | PR-AUC | $\Delta$ PR-AUC | ROC-AUC | Optimal $F_1$ | Precision | Recall |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$M_1$** | **Metadata Only** | 16 | **0.2847** | *Baseline* | 0.9088 | 0.4352 | 33.1% | 63.6% |
| **$M_2$** | **+ Creator Authority** | 18 | **0.2810** | -0.0037 | 0.9117 | 0.4023 | 32.4% | 53.0% |
| **$M_3$** | **+ Engagement Kinematics** | 35 | **0.3941** | **+0.1131** | **0.9644** | **0.4859** | **38.7%** | **65.2%** |
| **$M_4$** | **+ Comments Arrival Dynamics** | 41 | **0.3315** | -0.0626 | 0.9626 | 0.4706 | 38.5% | 60.6% |

*Key finding: Engagement kinematics delivers the decisive predictive leap (+40.2% relative PR-AUC lift). Raw comment arrival volume proves redundant with observation snapshots without NLP sentiment drift.*

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
│   └── exp2_feature_modalities.py # Reproducible Experiment 2 modality ablation runner
├── models/
│   ├── exp1/                    # Serialized model boosters (lgbm_1h..6h.txt) & metadata
│   └── exp2/                    # Modality stack boosters (lgbm_M1..M4.txt) & metadata
├── reports/
│   ├── exp1_temporal_horizon_results.json # JSON benchmark metrics & evaluation records
│   └── exp2_feature_modalities_results.json # JSON metrics across M1-M4 modality stacks
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
