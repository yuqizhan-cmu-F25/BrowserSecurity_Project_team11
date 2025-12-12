# LLM Tracking Project (VisibleV8 Crawler + Analysis)

End-to-end workflow for crawling sites with the VisibleV8 crawler and running local LLM-based tracking analysis. This repository bundles:
- `visiblev8-crawler/`: the upstream crawler stack (Dockerized services, CLI runner).
- `src/`: local analysis scripts that pull data from the crawler’s databases, classify tracking behavior, and generate reports/dashboards.

## Get the crawler
Clone the upstream crawler into this repo (already present here as `visiblev8-crawler`, but steps for a fresh setup):
```bash
git clone https://github.com/wspr-ncsu/visiblev8-crawler visiblev8-crawler
```

## Build and run the crawler stack
From `visiblev8-crawler/`:
```bash
# Upstream defaults (no override)
docker compose up --build
```

Key services & ports:
- API: `backend` on `4000`
- Postgres: `database` on `5434` (maps container 5432)
- Redis (Celery broker): `6380`
- MongoDB: `27017`
- Flower (task monitor): `5555`
- VNC/noVNC for workers: `5901` / `6901`

Persistent volumes (host-mounted): `vv8db2/` (Postgres), `redis_data/`, `mongo/data/`, `screenshots/`, `har/`, `raw_logs/`, `flower/data/`, `vv8_worker` artifacts.

Environment knobs (compose defaults shown):
- Postgres: `POSTGRES_USER=vv8`, `POSTGRES_PASSWORD=vv8`, `POSTGRES_DB=vv8_backend`
- MongoDB: `MONGO_INITDB_ROOT_USERNAME=vv8`, `MONGO_INITDB_ROOT_PASSWORD=vv8`
- Celery broker host: `task_queue_broker`

## Run crawls with the CLI
From `visiblev8-crawler/`:
```bash
pip install -r scripts/requirements.txt
python scripts/vv8-cli.py setup        # one-time setup

# Single URL with Mfeatures postprocessor
python scripts/vv8-cli.py crawl -u "https://example.com" -pp "Mfeatures"

# Multiple URLs from a newline-delimited file
python scripts/vv8-cli.py crawl -f urls.txt -pp "Mfeatures"

# Tranco CSV input
python scripts/vv8-cli.py crawl -c tranco.csv -pp "Mfeatures"
```

Monitoring:
- Flower UI: http://localhost:5555
- Logs: `python scripts/vv8-cli.py docker -f`

Database access:
```bash
psql --host=0.0.0.0 --port=5434 --dbname=vv8_backend --username=vv8
```
(Password defaults to `vv8`.)

Artifacts:
- Screenshots, HARs, and raw logs are mounted under `visiblev8-crawler/{screenshots,har,raw_logs}/`.
- Postgres data lives in `visiblev8-crawler/vv8db2/`.

## Analysis workflow (this repo)
Create a virtual environment (recommended) in the project root:
```bash
python -m venv .venv
source .venv/bin/activate
# Minimal deps used across scripts
pip install pandas numpy matplotlib seaborn scikit-learn psycopg2-binary anthropic
```

### 1) Extract crawl data from Postgres
```bash
python src/extract_data.py
```
Outputs to `data/processed/`:
- `crawls_summary.csv`
- `mega_features_<domain>.csv` for crawls that used the `Mfeatures` postprocessor

The script expects Postgres at `0.0.0.0:5434`, DB `vv8_backend`, user/password `vv8`.

### 2) Run LLM classifications
`src/models/llm_classifier_final.py` provides a `TrackingScriptClassifier` that consumes the `mega_features_*.csv` files and produces LLM classifications. Supply an `ANTHROPIC_API_KEY` and write results to `data/processed/llm_classifications.csv` (one row per URL).

### 3) Evaluate tracking detection
```bash
python src/analysis/evaluate_vv8.py
```
Requires:
- `data/processed/llm_classifications.csv`
- `data/processed/authoritative_ground_truth.csv` (EasyList/EasyPrivacy-derived ground truth)

Outputs to `data/dashboards/`:
- `vv8_evaluation_report.txt`
- `vv8_evaluation_metrics.json`
- `vv8_evaluation_dashboard.png` plus supporting charts

### 4) Build per-site dashboards
```bash
python src/create_dashboards.py
```
Reads `data/processed/llm_classifications.csv` and corresponding `mega_features_*.csv`, then writes dashboards to `data/dashboards/`.

### 5) Feature-importance report
```bash
python src/analysis/feature_importance.py
```
Reads `data/processed/llm_classifications.csv` and `mega_features_*.csv`, then writes `data/analysis_reports/feature_importance_report.md` with TF-IDF + random forest feature importances.

