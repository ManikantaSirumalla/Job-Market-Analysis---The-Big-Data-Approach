
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

## Repo Layout

```
data/               # raw/bronze/silver/gold lake-style folders (local for now)
notebooks/          # EDA, prototypes
src/
  common/           # shared utils (paths, logging, config)
  ingest/           # downloaders for GHArchive, BLS, StackOverflow
  etl/              # bronze->silver cleaning, silver->gold features
  ml/               # training scripts & model utils
  api/              # FastAPI app
configs/            # YAML/ENV configs
dags/               # (placeholder) Airflow DAGs later
ml/models/          # trained model artifacts (gitignored)
tests/              # unit tests
```

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

---

## Next Steps After Week 2

- Add Spark + Delta (local or dockerized) for large-scale ETL
- Add MLflow for experiment tracking
- Implement salary model (XGBoost/LightGBM) and text features from job descriptions
- Add Streamlit dashboard linked to the API
