# MediStock AI

**AI-powered pharmacy medicine demand forecasting and expiry-risk intelligence.**

A decision-support platform that forecasts 30-day product demand, identifies batch-level
expiry exposure under FEFO rotation, compares stock against predicted demand, and produces
explainable replenishment and rotation recommendations through a FastAPI backend and a
Streamlit dashboard.

> **Data disclosure.** All data is **synthetic**, produced by this repository's documented
> simulator. It is not real pharmacy transaction data. Outputs are inventory decision
> support only — never medical, dosage, diagnostic or treatment advice. No purchase or
> disposal is executed; every recommendation requires human review.

---

## Build status

| Phase | Scope | Status |
|---:|---|---|
| 1 | Architecture, contracts, config, stub API, tests | ✅ |
| 2 | Data generation, validation, cleaning, profiling, EDA | ✅ |
| 3 | Leakage-safe feature engineering | ✅ |
| 4 | Baselines + candidates, artifact bundle | ✅ |
| 5 | Evaluation, selection, error analysis | ✅ |
| 6 | FEFO expiry-risk engine + realised labels | ✅ |
| 7 | Inventory optimisation + explainability | ✅ |
| 8 | Database + real API endpoints | ✅ |
| 9 | Streamlit dashboard (7 pages) | ✅ |
| 10 | Automated tests (99 passing) | ✅ |
| 11 | Documentation, Docker, deployment prep | ✅ |
| 12 | Demo script, interview pack, review | ✅ |

**Measured results** (see [technical report](reports/technical_report.md)):

| Metric | Value |
|---|---|
| Test WAPE (best model) | **0.108** |
| Seasonal-naive baseline WAPE | 0.132 (18% relative reduction) |
| Expiry-risk precision / recall | **0.818 / 0.900** |
| Critical-band precision / recall | 1.00 / 1.00 (n=7) |
| Tests passing | **99** |

---

## 1. Project overview

A pharmacy holds stock that perishes. Understock loses sales; overstock expires. MediStock AI
is an end-to-end system — data pipeline, model, risk engine, API, UI, tests, docs — that turns
historical sales and batch expiry data into reviewable inventory decisions.

It is deliberately **not** a notebook and **not** a mock: every figure in the dashboard comes
from a checksummed model artifact served over HTTP.

## 2. Business problem

Traditional inventory practice uses trailing averages and memory, which cannot answer:

* Which products will we run out of during the supplier lead time?
* Which batches will still be on the shelf when they expire?
* Where are we simultaneously over-stocked and under-stocked?

The system answers those three questions and shows the arithmetic behind each answer.

## 3. Objectives

1. Forecast 30-day demand per product without using future information.
2. Identify batches whose forecast demand before expiry cannot clear their remaining quantity.
3. Compare inventory against forecast demand and surface both failure modes.
4. Produce explainable, auditable recommendations — never autonomous actions.
5. Be reproducible: seeded generation, chronological evaluation, checksummed artifacts.

## 4. Key features

* **Leakage-safe forecasting** — features structurally cannot read past the prediction origin.
* **Fair model comparison** — seasonal naive, moving average, Random Forest, HistGradientBoosting on identical rows; selection by rolling-origin WAPE, test scored once.
* **FEFO expiry engine** — explicit allocation rule and a published severity table; evaluated against realised outcomes with proper censoring.
* **Explainability with honest labels** — permutation importance globally, one-at-a-time ablation per prediction, never described as SHAP.
* **One artifact bundle** — 12 files with SHA-256 manifest, horizon-locked; the API refuses to answer a different question than the model was trained for.
* **One inference path** — dashboard → API → bundle → model. No duplicated business logic.
* **Honest failure states** — 503 without artifacts, 422 on horizon mismatch, explicit error UI, no placeholder numbers anywhere.

## 5. Architecture

```
┌──────────────────── OFFLINE (ml/, scripts/) ─────────────────────┐
│ generate → validate → clean → features → train → evaluate        │
│                │                                                 │
│                └────────► artifacts/v1/ (manifest + SHA-256)     │
└───────────────────────────────┬──────────────────────────────────┘
                                │ read-only contract
┌───────────────────────────────▼───────────── ONLINE ─────────────┐
│ ArtifactStore.load() → ServiceContainer                          │
│   ├── ForecastService   (model.predict)                          │
│   ├── ExpiryService     (FEFO + severity policy)                  │
│   ├── InventoryService  (coverage + status)                       │
│   └── RecommendationService (ordered rule table)                  │
│ FastAPI /api/v1/... ──httpx──► Streamlit (7 pages)                │
│ SQLite: operational catalogue + persisted scoring snapshot        │
└───────────────────────────────────────────────────────────────────┘
```

Detailed diagrams: [docs/architecture.md](docs/architecture.md). Why each choice was made:
[DECISIONS.md](DECISIONS.md) (D-001 … D-013).

## 6. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | Required by the brief; modern typing support |
| Data | pandas 3.0, numpy 2.4, pyarrow | Parquet for processed data, CSV for raw |
| ML | scikit-learn 1.8 | Pipelines give leakage-safe preprocessing; no extra deps to justify |
| Backend | FastAPI 0.115 + Pydantic 2 | Typed contracts, free OpenAPI, dependency injection |
| Validation | Pydantic v2 + pydantic-settings | Request contracts and typed config in one idiom |
| Persistence | SQLAlchemy 2 + SQLite | Real constraints/indexes; Alembic configured but not used (see CUTLIST) |
| UI | Streamlit 1.56 + Plotly 5.24 | Fast to build, and it is a pure API client here |
| Testing | pytest 8, httpx, Streamlit AppTest | Includes rendering every dashboard page |
| Lint | ruff 0.8 | Fast, single tool for lint + import order |
| Deploy | Docker + Compose | API and dashboard as separate services |

## 7. Repository structure

```
medistock-ai/
├── app/                     # FastAPI service
│   ├── api/                 #   routers, dependencies, error handlers
│   ├── core/                #   settings, logging, domain exceptions
│   ├── database/            #   ORM base, entities, sessions
│   ├── schemas/             #   Pydantic request/response contracts
│   ├── services/            #   artifact store, forecast, expiry, inventory, recommendations
│   └── main.py
├── dashboard/               # Streamlit UI (theme, components, 7 pages)
├── ml/                      # offline pipeline
│   ├── data_generation/     #   seeded simulator + catalogue + config
│   ├── preprocessing/       #   validation, cleaning, profiling
│   ├── features/            #   leakage-safe feature builder
│   ├── forecasting/         #   baselines, pipelines, splits, training, bundle writer
│   ├── expiry_risk/         #   FEFO, severity policy, realised labels, evaluation
│   ├── inventory/           #   replenishment math, recommendation rules
│   ├── evaluation/          #   metrics, EDA, error analysis
│   ├── explainability/      #   permutation importance, per-prediction ablation
│   └── common/              #   canonical I/O, seeding, thread limits
├── scripts/                 # operator entrypoints (generate → train → serve)
├── tests/                   # unit / integration / regression
├── docs/                    # architecture, leakage, model card, dictionary, interview pack
├── reports/                 # technical report, EDA, error analysis, test report, figures
├── artifacts/v1/            # the served model bundle (committed for the demo)
├── data/                    # raw / interim / processed (regenerated, git-ignored)
├── DECISIONS.md · LIMITATIONS.md · CUTLIST.md
├── Dockerfile · docker-compose.yml
└── requirements.txt · requirements.lock.txt · pyproject.toml · .env.example
```

## 8. Dataset description

| Property | Value |
|---|---|
| Products | 120 across 8 therapeutic categories |
| History | 2024-05-01 → 2025-12-31 (610 days) |
| Sales transactions | 67,558 |
| Demand panel (product × day) | 73,200 rows |
| Batches at snapshot | 298 |
| Snapshot / prediction origin | 2025-10-01 |
| Zero-demand share | 7.71% |

Column-level detail: [docs/data_dictionary.md](docs/data_dictionary.md). Profile output:
`reports/data_profile.md`.

## 9. Data generation assumptions

Full statement: [docs/data_generation.md](docs/data_generation.md). In brief — category
demand levels, category-specific weekday profiles, annual seasonality with per-category
amplitude and peak, lognormal product scale, per-product trend, 3% promotion days with
1.5–3× uplift, 18% intermittent products, Poisson noise, multi-batch inventory split
multinomially, and expiry windows drawn from three bands (30% within 12–90 days).

Structural invariants (tested): batch quantities sum exactly to on-hand stock; manufacturing
precedes expiry; arrivals never come from the future; remaining never exceeds batch quantity.

## 10. Installation (Windows 11)

```powershell
cd medistock_ai
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt   # exact verified versions
# or, for development:  python -m pip install -r requirements-dev.txt
```

> **OneDrive note.** If the repo lives in a synced folder, venv creation is slow because
> thousands of small files get synced. Either create the venv outside the folder
> (`py -3.11 -m venv C:\venvs\medistock`) or pause sync during install. If `pip` appears to
> hang while *resolving*, install `requirements.lock.txt` — exact pins skip resolver
> backtracking.

## 11. Environment setup

```powershell
Copy-Item .env.example .env
```

Every variable is `MEDITSTOCK_`-prefixed (see `.env.example`). Key ones:
`MEDITSTOCK_RANDOM_SEED`, `MEDITSTOCK_ARTIFACT_VERSION`, `MEDITSTOCK_DB_URL`,
`MEDITSTOCK_FORECAST_HORIZON_DAYS`, `MEDITSTOCK_RISK_CRITICAL_DAYS`.

`.env` is git-ignored; `.env.example` is the documented contract.

## 12. Database initialisation

```powershell
python -m scripts.init_db                # schema + master data
python -m scripts.init_db --with-scores  # + persisted forecast/risk/recommendation snapshot
```

Creates `data/medistock.db` with 8 tables, CHECK constraints and indexes. SQLite foreign-key
enforcement is enabled per connection (`PRAGMA foreign_keys=ON`) — off by default in SQLite.

## 13. Data preparation

```powershell
python -m scripts.generate_data     # seeded; ~2 s
python -m scripts.validate_data     # 24 checks + cleaning + profile
python -m scripts.build_features    # 2,400 (product, origin) rows
```

Expected: `raw_errors=0 raw_warnings=1 post_clean_errors=0`. The warning is 46 below-cost
sales on promotion days, retained deliberately.

## 14. Model training

```powershell
python -m scripts.train_pipeline
```

Trains 4 candidates, tunes on validation, selects by mean rolling-origin WAPE, refits on
train+validation, scores test once, and writes `artifacts/v1/`.

Output:

```
model                     split             MAE     RMSE    WAPE
hist_gradient_boosting    test            29.66    39.96   0.108
random_forest             test            33.56    45.99   0.122
seasonal_naive_30d        test            36.32    48.14   0.132
moving_average_90d        test            58.90    77.50   0.214
```

## 15. Evaluation and diagnostics

```powershell
python -m scripts.run_expiry_risk     # FEFO + realised labels -> metrics_expiry.json
python -m scripts.run_error_analysis  # by category / demand decile / sparsity
python -m scripts.run_eda             # 8 figures + narrative report
```

Real results: expiry precision **0.818**, recall **0.900**, F1 0.857 on 101 evaluable batches
(197 censored and excluded). Error analysis shows sparse products at WAPE 0.255 vs 0.099 for
dense ones.

## 16. API execution

```powershell
.\scripts\run_api.ps1          # or: python -m uvicorn app.main:app --port 8000
```

Interactive docs: <http://127.0.0.1:8000/docs>

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + per-component readiness |
| `GET /api/v1/products` | Product catalogue (filter by category) |
| `GET /api/v1/inventory/overview` | Stock, value, coverage, status |
| `GET /api/v1/forecast/{product_id}` | 30-day forecast + interval + drivers |
| `POST /api/v1/forecast` | Same, with explicit horizon/origin (validated) |
| `GET /api/v1/expiry-risk` | Ranked batch risk (filter by severity) |
| `GET /api/v1/expiry-risk/{batch_id}` | Single batch + audit inputs |
| `GET /api/v1/recommendations` | Ordered recommendations (filter by action) |
| `GET /api/v1/model/metrics` | Demand + expiry metrics, importance |
| `GET /api/v1/model/info` | Artifact identity and provenance |

Errors use one envelope: `{"error": {"code", "message", "details"}}`.

## 17. Dashboard execution

```powershell
streamlit run dashboard/streamlit_app.py
```

<http://127.0.0.1:8501> — seven pages: Executive Overview, Demand Forecasting, Expiry
Intelligence, Inventory Optimisation, Product Explorer, Model Performance, Data Quality &
System Health.

The dashboard is a pure API client. It imports no ML code and reads no CSVs, so its numbers
cannot diverge from the API's.

## 18. Docker execution

```powershell
docker compose up --build     # API :8000, dashboard :8501
```

Or the API alone:

```powershell
docker build -t medistock-api .
docker run -p 8000:8000 -v "${PWD}/artifacts:/app/artifacts:ro" medistock-api
```

The artifact bundle is mounted read-only rather than baked in, so a retrain does not require
a rebuild and the served manifest stays inspectable.

> **Not verified in this environment**: no Docker daemon was available where this project was
> assembled. The files are syntactically valid and follow the same commands used natively,
> but the image build itself was not executed. Treat Docker as untested.

## 19. Testing

```powershell
python -m pytest                      # 99 tests
python -m pytest -m integration       # API + dashboard contract tests
python -m ruff check .                # lint
```

Coverage by area: configuration, artifact integrity, leakage, generator determinism,
FEFO/severity, replenishment rules, and every API endpoint with and without a bundle.
Evidence and the bugs found: [reports/test_report.md](reports/test_report.md).

## 20. Example predictions (real output)

`GET /api/v1/forecast/MED-0001`:

```json
{
  "product_id": "MED-0001",
  "medicine_name": "Amoxicillin 5mg Capsule",
  "as_of_date": "2025-10-01",
  "horizon_days": 30,
  "predicted_demand": 151.6023,
  "prediction_interval": {
    "lower": 103.1763, "upper": 189.5423,
    "method": "empirical_test_residual_quantiles_10_90", "coverage": 0.8
  },
  "model_name": "hist_gradient_boosting",
  "artifact_version": "v1",
  "confidence": "high",
  "drivers": [
    {"feature": "sum_30", "contribution": 62.6, "direction": "increases_demand",
     "method": "one_at_a_time_ablation_vs_population_reference"}
  ]
}
```

`GET /api/v1/recommendations?action=reorder&limit=1`:

```json
{
  "product_id": "MED-0030", "action": "reorder", "priority": "high",
  "suggested_replenishment_qty": 173, "forecast_demand": 262.0,
  "current_stock": 40, "safety_stock": 18, "lead_time_days": 12,
  "coverage_days": 4.6,
  "reasons": ["Current stock covers 4.6 days of forecast demand but the supplier lead time is 12 days."]
}
```

Error case — `GET /api/v1/forecast/MED-0001?horizon_days=14`:

```json
{"error": {"code": "invalid_request",
  "message": "This model is trained for a 30-day horizon and cannot answer a 14-day question.",
  "details": {"requested_horizon_days": 14, "trained_horizon_days": 30}}}
```

## 21. Known limitations

Full list: [LIMITATIONS.md](LIMITATIONS.md). The load-bearing ones:

1. **Synthetic data.** Accuracy measures recovery of a known generator, not real performance.
2. **No stockout censoring.** Sales are not hidden by availability, unlike reality.
3. **FEFO assumes perfect discipline.** Real exposure is likely worse than predicted.
4. **One prediction origin.** The artifact answers exactly one question and refuses others.
5. **197 of 298 batches censored** in expiry evaluation; excluded rather than counted as safe.
6. **No auth, SQLite, single node.** Not production-ready.

## 22. Future improvements

1. Real data with stockout flags + demand un-censoring.
2. Quantile-regression forecasts with verified coverage.
3. Scheduled retraining and feature-drift monitoring.
4. Auth, audit trail, role-based access.
5. Promotion and supplier-shock features (largest current error source).
6. ERP integration for batch-level truth instead of a nightly snapshot.
7. Alembic migrations and an automated CI pipeline.

---

## Documentation map

| File | Contents |
|---|---|
| [reports/technical_report.md](reports/technical_report.md) | Full methodology, results, error analysis, ethics |
| [reports/test_report.md](reports/test_report.md) | Executed test evidence + bugs found |
| [reports/eda_report.md](reports/eda_report.md) | EDA figures with business interpretation |
| [reports/error_analysis.md](reports/error_analysis.md) | Where the model fails, by category/decile/sparsity |
| [docs/architecture.md](docs/architecture.md) | Component and data-flow diagrams |
| [docs/leakage.md](docs/leakage.md) | Leakage threat model and controls |
| [docs/model_card.md](docs/model_card.md) | Model card |
| [docs/data_generation.md](docs/data_generation.md) | Simulator assumptions |
| [docs/data_dictionary.md](docs/data_dictionary.md) | Every table and column |
| [docs/interview_pack.md](docs/interview_pack.md) | 10 questions + rapid-fire answers |
| [docs/demo_script.md](docs/demo_script.md) | Rehearsed 5-minute walkthrough |
| [docs/CHECKLIST.md](docs/CHECKLIST.md) | Phase-by-phase gates |
| [DECISIONS.md](DECISIONS.md) | ADR log with trade-offs |

## Non-goals

No medication/dosage/diagnosis/treatment guidance. No autonomous purchasing or disposal. No
patient data. No claim of production readiness.

## Licence

MIT (code). Data generated by this repository is synthetic.
