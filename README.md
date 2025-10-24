
# Job Market Analysis Tool – Big Data (Solo Starter)

This repo is a clean, minimal starter that matches the original team plan but streamlined for a **solo** build. It gives you a runnable FastAPI service, ingestion stubs, ETL scaffolding, and a place to add models—not heavy infra yet.

## Quickstart (Local)

1) **Python 3.11** recommended.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) **Run the API** (for now it just has a health check and a stub salary endpoint).
```bash
make api
# visit http://127.0.0.1:8000/docs
```

3) **Try a tiny sample ingest** (one day of GH Archive, one BLS table).
```bash
python src/ingest/download_gharchive.py --date 2024-01-01
python src/ingest/download_bls.py --series oe
```

> We’re intentionally starting small. Once the skeleton works end‑to‑end, we’ll scale to the full 20GB+ sources and add Spark/Delta/Airflow.

---

## Project Structure

```
job-market-analysis/
├── 📁 data/                           # Data Lake Architecture (129.68 GB)
│   ├── 📁 raw/                        # Original data sources
│   │   ├── 📁 github/                 # GitHub Archive data (123.43 GB)
│   │   ├── 📁 kaggle/                 # Job market datasets (2.23 GB)
│   │   ├── 📁 stackoverflow/          # Developer surveys (0.88 GB)
│   │   └── 📁 bls/                    # BLS employment data (0.002 GB)
│   ├── 📁 bronze/                     # Cleaned and standardized data (3.14 GB)
│   ├── 📁 silver/                     # Unified datasets (0.004 GB)
│   └── 📁 gold/                       # ML-ready data (0 GB)
│
├── 📁 src/                            # Source Code
│   ├── 📁 api/                        # FastAPI Application
│   │   └── 📄 app.py                  # Main API server
│   ├── 📁 common/                     # Shared Utilities
│   │   ├── 📄 logs.py                 # Logging configuration
│   │   └── 📄 paths.py                # Path management
│   ├── 📁 etl/                        # ETL Pipeline
│   │   ├── 📄 bronze_to_silver.py     # Bronze to Silver processing
│   │   ├── 📄 build_unified_postings.py # Unified job postings
│   │   └── 📄 silver_to_gold.py       # Silver to Gold processing
│   ├── 📁 ingest/                     # Data Ingestion
│   │   ├── 📄 comprehensive_data_processor.py # Main data processor
│   │   ├── 📄 download_gharchive.py   # GitHub Archive downloader
│   │   ├── 📄 download_stackoverflow.py # StackOverflow downloader
│   │   ├── 📄 download_bls.py         # BLS data downloader
│   │   └── 📄 massive_github_collector.py # Large-scale GitHub collection
│   ├── 📁 ml/                         # Machine Learning
│   │   ├── 📄 salary_prediction_model.py # XGBoost salary model
│   │   ├── 📄 skill_forecasting.py    # Skills trend analysis
│   │   └── 📄 train_salary_model.py   # Model training pipeline
│   ├── 📁 spark/                      # Apache Spark Processing
│   │   ├── 📄 etl_pipeline.py         # Spark ETL pipeline
│   │   └── 📄 simple_etl.py           # Simple Spark processing
│   ├── 📁 streaming/                  # Real-time Streaming
│   │   └── 📄 kafka_demo.py           # Apache Kafka streaming demo
│   ├── 📄 app_streamlit.py            # Main Streamlit dashboard
│   └── 📄 app_streamlit_simple.py     # Simple Streamlit interface
│
├── 📁 dags/                           # Apache Airflow DAGs
│   ├── 📄 job_market_airflow_dag.py   # Main Airflow workflow
│   └── 📄 job_market_dag.py           # Job market processing DAG
│
├── 📁 notebooks/                      # Jupyter Notebooks
│   └── 📄 EDA.ipynb                   # Exploratory Data Analysis
│
├── 📁 reports/                        # Analysis Reports
│   └── 📄 job_market_analysis.md      # Comprehensive analysis report
│
├── 📁 models/                         # Trained Models
│   └── 📄 salary_prediction_model.pkl # XGBoost salary prediction model
│
├── 📁 mlruns/                         # MLflow Experiment Tracking
│   └── 📁 [experiment_runs]/          # ML experiment runs and artifacts
│
├── 📄 fast_api.py                     # FastAPI server launcher
├── 📄 smart_career_api.py             # Career insights API
├── 📄 optimized_streamlit.py          # Optimized Streamlit app
├── 📄 simple_streamlit.py             # Simple Streamlit app
├── 📄 demo_academic_presentation.py   # Academic presentation demo
├── 📄 demo_all_tools_working.py       # Big Data tools demonstration
├── 📄 requirements.txt                # Python dependencies
├── 📄 requirements-dev.txt            # Development dependencies
├── 📄 Makefile                        # Build automation
├── 📄 start_apps.sh                   # Application startup script
└── 📄 README.md                       # This file
```

## Key Features

### 🏗️ **Data Lake Architecture**
- **Raw Layer**: 126.54 GB of original data from 4 sources
- **Bronze Layer**: 3.14 GB of cleaned and standardized data
- **Silver Layer**: Unified datasets across sources
- **Gold Layer**: ML-ready feature engineering

### 🚀 **Big Data Tools**
- **Apache Spark**: Distributed data processing
- **Delta Lake**: Versioned data storage
- **Apache Airflow**: Workflow orchestration
- **Apache Kafka**: Real-time streaming
- **MLflow**: ML experiment tracking

### 🤖 **Machine Learning**
- **XGBoost**: Salary prediction model
- **Feature Engineering**: Automated pipeline
- **Model Serving**: FastAPI endpoints
- **Experiment Tracking**: MLflow integration

### 🌐 **API & Visualization**
- **FastAPI**: REST API with 5+ endpoints
- **Streamlit**: Interactive dashboards
- **Real-time Data**: Live GitHub activity
- **Predictive Analytics**: Salary predictions

---

## Solo Build Roadmap (Week 1–2)

- [ ] **Environment ready**: venv, requirements, `make api` runs, `/health` healthy
- [ ] **Sample ingest works**: pull 1 day GHArchive JSON, 1 BLS table to `data/raw`
- [ ] **Bronze → Silver**: basic cleaner normalizes schema for GH/BLS
- [ ] **EDA notebook**: quick profiling + first plots
- [ ] **ML stub**: scaffold training for salary regression (no real model yet)
- [ ] **Streamlit stub**: tiny dashboard that can hit the API

Once this is stable, we’ll add **Spark**, **Delta Lake**, and **Airflow** so you can scale up.

---

## Make Targets

```bash
make api         # run FastAPI (uvicorn)
make format      # format with black (optional if installed)
make test        # run tests (placeholder)
```

---

## Configuration

Use `.env` (local only) to store simple secrets/paths:

```
DATA_DIR=./data
BLS_BASE=https://download.bls.gov/pub/time.series
```

> For production, move secrets to a proper store (Keychain, cloud secret manager).

