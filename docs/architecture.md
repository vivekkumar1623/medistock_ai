# Architecture

## 1. Two planes

MediStock AI separates an **offline training plane** from an **online serving
plane**. They meet at exactly one artefact: the versioned artifact bundle.

```
┌──────────────────────────── OFFLINE (ml/) ────────────────────────────┐
│                                                                       │
│  data_generation ──► data/raw/*.csv                                   │
│        │                     │                                        │
│        ▼                     ▼                                        │
│   (seed=42)          validation + cleaning ──► data/processed/*.parquet│
│                              │                                        │
│                              ▼                                        │
│                      feature engineering  (origin-date, lag-only)     │
│                              │                                        │
│              ┌───────────────┼────────────────┐                       │
│              ▼               ▼                ▼                       │
│        baseline        random_forest   hist_gradient_boosting         │
│              └───────────────┼────────────────┘                       │
│                     chronological eval (val → test)                  │
│                              │                                        │
│                              ▼                                        │
│                 expiry-risk engine (FEFO + realised labels)           │
│                              │                                        │
│                              ▼                                        │
│                    artifacts/<version>/  ◄── manifest.json (SHA-256)  │
└───────────────────────────────┬───────────────────────────────────────┘
                                │  read-only contract
┌───────────────────────────────▼───────────── ONLINE ──────────────────┐
│                                                                       │
│   ArtifactStore.load()  ──verify──►  ArtifactBundle                   │
│            │                              │                           │
│            │                    ┌─────────┴──────────┐                │
│            ▼                    ▼                    ▼                │
│    forecast_service     expiry_service      inventory_service         │
│            └──────────────────┬─────────────────────┘                 │
│                               ▼                                       │
│                        FastAPI  /api/v1/...                           │
│                               │  (httpx, server-side)                │
│                               ▼                                       │
│                        Streamlit dashboard                            │
└───────────────────────────────────────────────────────────────────────┘
```

## 2. Module boundaries

| Layer | Package | May import | Must never import |
|---|---|---|---|
| Config/logging | `app/core` | stdlib, pydantic | `app.api`, `ml` |
| Domain errors | `app/core/exceptions` | stdlib | FastAPI |
| Contracts | `app/schemas` | pydantic | `app.services`, `ml` |
| Artifact access | `app/services/artifact_store` | `app/core`, joblib, pandas | `app.api` |
| Business services | `app/services/*` | `artifact_store`, `app/schemas` | FastAPI `Request` |
| HTTP edge | `app/api` | everything below + FastAPI | `ml` |
| Persistence | `app/database` | SQLAlchemy, `app/core` | `app.api` |
| UI | `dashboard` | `httpx`, `plotly`, `streamlit` | `ml`, `app.database` |

The rule that matters: **the dashboard does not import the ML pipeline.** If the
UI could recompute a forecast locally, the API and the UI could disagree, and
there would be two sources of truth for the same number.

## 3. Request lifecycle (serving)

```
GET /api/v1/forecast/MED-0001
  │
  ├─ route (app/api/v1/forecast.py)
  │    └─ BundleDep ──► ArtifactStore.load()
  │                         ├─ manifest.json present?      no → 503 artifact_not_ready
  │                         ├─ contract major matches?     no → 500 artifact_integrity_error
  │                         ├─ SHA-256 of every file?      no → 500 artifact_integrity_error
  │                         └─ horizon == configured?      no → 500 artifact_integrity_error
  │
  ├─ forecast_service.predict(bundle, product_id)
  │    ├─ knows the product?                    no → 404 not_found
  │    ├─ builds the feature row from bundle.demand_history (origin-locked)
  │    ├─ bundle.model.predict(row)
  │    └─ attaches drivers, interval, notes from bundle.importance / metrics
  │
  └─ ForecastResponse (schema-validated) ──► JSON
```

Every failure path is a typed exception raised below the HTTP layer and rendered
by one handler pair in `app/api/errors.py`.

## 4. The artifact bundle

```
artifacts/v1/
├── manifest.json              identity, provenance, per-file SHA-256
├── model.joblib               fitted Pipeline (preprocessing + estimator)
├── baseline.joblib            seasonal-naive parameters (fair-comparison baseline)
├── feature_spec.json          ordered features, dtypes, category levels, lookback
├── metrics.json               per-model val/test metrics + selection rationale
├── training_config.json       seed, split dates, hyperparameters
├── demand_history.parquet     product × date demand history (lag source)
├── products.parquet           product master snapshot at the origin
├── inventory_snapshot.parquet stock + batch snapshot at the origin
├── importance.json            global feature importance
├── metrics_expiry.json        precision / recall / F1 / confusion matrix
└── expiry_policy.json         the exact risk rule and thresholds used
```

Why a directory of files instead of one pickle: each component is independently
inspectable, diffable and replaceable. Why a manifest instead of trusting the
files: a model that loads is not the same as a model that *matches its features*.
The horizon check alone prevents the most plausible silent bug in the system.

## 5. Data flow invariants

1. **Origin locking.** Every feature for a prediction at origin `T` uses only
   observations with `date <= T`. Enforced in `ml/features/` and re-checked by
   the leakage tests.
2. **No target in features.** The target `demand_next_30d` is constructed only
   in the training dataset builder, from rows strictly after the origin.
3. **No preprocessing fitted on test data.** Imputers/scalers live inside the
   sklearn `Pipeline` stored in `model.joblib`.
4. **One evaluation window.** Baseline, RF and HistGB are scored on identical
   rows of the identical test split.
5. **Labels for expiry risk come from realised futures**, never from features.

## 6. Design trade-offs (summary)

| Choice | Alternative rejected | Reason |
|---|---|---|
| Aggregate 30-day demand target | Daily demand forecasting | Matches the replenishment decision; far less noisy; easier to evaluate per product |
| Chronological split | Random k-fold | Time order is the whole problem; random splits leak the future |
| HistGB candidate | LSTM / Prophet | No meaningful gain at ~1k products × 2y synthetic data; heavy deps; harder to defend |
| Artifact bundle + manifest | Pickle the model only | Prevents feature/horizon mismatch; makes the "saved model" claim checkable |
| Server-side dashboard → API | Dashboard reads parquet directly | One source of truth; forces the inference path to be real |
| 503 for stubs | Shape-correct dummy JSON | Fabricated numbers violate the project's own hard rules |
