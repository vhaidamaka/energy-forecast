# ⚡ Energy Predictor

A full-stack energy consumption prediction platform supporting ARIMA, LSTM, BiLSTM, and Wavelet+LSTM models.

---

## 🐳 Docker (recommended)

### Prerequisites
- Docker 24+ and Docker Compose v2

### Production — one command

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| **UI** | http://localhost |
| **API** | http://localhost:8000 |
| **API docs** | http://localhost:8000/docs |

All data (SQLite DB, uploaded datasets, model runs) is persisted in a named Docker volume `energy_storage`.

### Development — hot reload for both services

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

- Backend: FastAPI with `--reload` watching source files → http://localhost:8000
- Frontend: Vite dev server with HMR → http://localhost:5173

### Useful Docker commands

```bash
# Start in background
docker-compose up -d --build

# View logs
docker-compose logs -f
docker-compose logs -f backend
docker-compose logs -f frontend

# Restart a single service
docker-compose restart backend

# Stop everything
docker-compose down

# Stop and wipe all data (volumes)
docker-compose down -v

# Rebuild after code changes
docker-compose up --build

# Open a shell in the backend container
docker-compose exec backend bash

# Run CLI commands inside the container
docker-compose exec backend python -m cli.predict models
docker-compose exec backend python -m cli.predict dataset list
docker-compose exec backend python -m cli.predict run list
```

### Environment variables

Copy `.env.example` to `.env` to customise:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:////app/storage/energy_predictor.db` | SQLite path |
| `DATA_DIR` | `/app/storage/data` | Uploaded dataset storage |
| `RUNS_DIR` | `/app/storage/runs` | Model run artifacts |
| `CORS_ORIGINS` | `http://localhost,http://localhost:80` | Allowed frontend origins |
| `DEBUG` | `false` | FastAPI debug mode |

---

## 🖥 Local Development (without Docker)

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## CLI Usage

Run from `backend/` (venv activated), or inside the Docker container via `docker-compose exec backend`:

```bash
# One-shot: upload + train + results
python -m cli.predict predict data_set.csv --model wavelet_lstm --horizon 7

# Dataset management
python -m cli.predict dataset upload hospital_building_dataset.xlsx --name "Hospital 2023"
python -m cli.predict dataset list
python -m cli.predict dataset info 1
python -m cli.predict dataset preview 1 --rows 20
python -m cli.predict dataset delete 1

# Run management
python -m cli.predict run create --dataset 1 --model bilstm --horizon 7 --name "BiLSTM test"
python -m cli.predict run list
python -m cli.predict run results 3 --forecast
python -m cli.predict run compare "1,2,3,4"
python -m cli.predict run delete 2

# Export results
python -m cli.predict export 1 --format csv
python -m cli.predict export 1 --format excel --out my_results.xlsx

# Show model defaults
python -m cli.predict models

# Custom hyperparameters via JSON
python -m cli.predict predict data.csv \
  --model lstm \
  --params '{"epochs": 100, "lookback": 48, "units_1": 128, "dropout": 0.3}'
```

---

## Supported Models

| Model | Description | Best for |
|---|---|---|
| `arima` | ARIMA / SARIMAX | Univariate, seasonal, fast |
| `lstm` | 2-layer LSTM | General multivariate |
| `bilstm` | Bidirectional LSTM | Higher accuracy, more memory |
| `wavelet_lstm` | DWT (db4/haar/sym8) + LSTM | Noisy signals, multi-frequency |

### Wavelet+LSTM approach

1. Apply `pywt.wavedec(energy, wavelet='db4', level=2)` to extract approximation + detail coefficients
2. Concat into a fixed 10-element global feature vector
3. Tile across all timesteps and concatenate with scaled time features
4. Feed into 2-layer LSTM (64→32 units, 20% dropout)

---

## Project Structure

```
energy-predictor/
├── docker-compose.yml            Production stack
├── docker-compose.dev.yml        Dev override (hot reload)
├── .env.example                  Environment template
├── backend/
│   ├── Dockerfile                Multi-stage Python build
│   ├── .dockerignore
│   ├── main.py                   FastAPI app
│   ├── config.py                 Settings (env-aware)
│   ├── schemas.py                Pydantic models
│   ├── requirements.txt
│   ├── core/
│   │   ├── database.py           SQLAlchemy models
│   │   ├── preprocessing.py      CSV/XLSX ingestion
│   │   ├── seq_utils.py          Sliding window utils
│   │   └── training_runner.py    Async training + SSE logs
│   ├── models/
│   │   ├── arima_model.py
│   │   ├── lstm_model.py
│   │   ├── bilstm_model.py
│   │   └── wavelet_lstm_model.py
│   ├── api/routes/
│   │   ├── datasets.py
│   │   ├── runs.py
│   │   └── export.py
│   └── cli/predict.py            Typer CLI
└── frontend/
    ├── Dockerfile                Multi-stage Node → nginx
    ├── Dockerfile.dev            Vite dev server
    ├── .dockerignore
    ├── nginx.conf                Nginx: SPA + /api proxy + SSE
    ├── vite.config.js
    └── src/
        ├── pages/
        │   ├── Dashboard.jsx
        │   ├── Datasets.jsx
        │   ├── Configure.jsx
        │   ├── Training.jsx      Live SSE log stream
        │   ├── Results.jsx       Charts + export
        │   └── Compare.jsx       Side-by-side comparison
        └── api/client.js
```

---

## REST API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/datasets/upload` | Upload CSV/XLSX |
| GET | `/api/datasets/` | List datasets |
| GET | `/api/datasets/{id}/preview` | Preview rows |
| PUT | `/api/datasets/{id}` | Rename/update |
| DELETE | `/api/datasets/{id}` | Delete dataset |
| POST | `/api/runs/` | Create run |
| POST | `/api/runs/{id}/start` | Start training |
| GET | `/api/runs/{id}/logs` | SSE live log stream |
| GET | `/api/runs/` | List runs |
| GET | `/api/runs/{id}` | Get run + result |
| GET | `/api/runs/compare/metrics?run_ids=1,2,3` | Compare metrics |
| GET | `/api/export/{id}/csv` | Download CSV |
| GET | `/api/export/{id}/excel` | Download Excel |
| GET | `/api/models/defaults` | Default hyperparams |
