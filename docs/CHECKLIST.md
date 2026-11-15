# Implementation Checklist

Status legend: `[x]` done and verified · `[~]` done, not verified · `[ ]` not started

Update this file at the end of every phase. A phase is only `[x]` when its
verification command has been run and its evidence is recorded in
`reports/test_report.md`.

## Phase 1 — Architecture, contracts, stub API

- [x] Repository skeleton (`app/`, `ml/`, `dashboard/`, `tests/`, `docs/`, `scripts/`, `reports/`)
- [x] `pyproject.toml` (packaging, ruff, pytest markers) + `requirements.txt` + `requirements.lock.txt`
- [x] `.env.example`, `.gitignore`, `.env` prefix isolation
- [x] Typed settings singleton (`app/core/config.py`) with path resolution
- [x] Structured logging, text + JSON modes (`app/core/logging.py`)
- [x] Domain exception hierarchy + one HTTP error envelope
- [x] **ArtifactBundle contract** (`app/services/artifact_store.py`) with SHA-256 verification
- [x] Pydantic contracts for every endpoint (`app/schemas/`)
- [x] App factory, lifespan, `/health`, OpenAPI tags (`app/main.py`)
- [x] Stub routes returning 503 (never fake data)
- [x] Database engine/session scaffold with SQLite FK pragma
- [x] Test factory for contract-valid bundles (`tests/factories.py`)
- [x] 30 tests passing, lint clean
- [x] `DECISIONS.md` (D-001…D-009 + open decisions), `LIMITATIONS.md`, `CUTLIST.md`
- [x] `docs/architecture.md`, `docs/leakage.md`, `data/README.md`
- [x] Live API verified over HTTP (`/health`, 503 envelope, OpenAPI paths)

**Phase 1 gate:** `python -m pytest` → 30 passed; `python -m ruff check .` → clean. ✅

## Phase 2 — Data generation, validation, cleaning, EDA

- [x] Product master generator (category demand profiles, lead times, prices)
- [x] Sales generator (weekly seasonality, monthly/seasonal effects, spikes, zero-sales days)
- [x] Inventory + multi-batch generator with varied expiry windows
- [x] Deterministic output (seeded; byte-identical regeneration test)
- [x] Validation rules + quarantine report (24 checks)
- [x] Cleaning pipeline (duplicates, missing values, outliers)
- [x] Data profiling report
- [x] EDA notebook + reusable scripts with business interpretation (8 figures + narrative)
- [x] `docs/data_dictionary.md` and generator assumptions

**Phase 2 gate:** `generate_data` twice → identical frames (test); validation report 0 errors, 1 documented warning. ✅

## Phase 3 — Feature engineering (leakage-safe)

- [x] Origin-grid dataset builder (`product × origin`, 2,400 rows)
- [x] Lag / rolling features (strictly `date <= origin`)
- [x] Calendar features (legitimately future-known)
- [x] Category encodings frozen into `feature_spec.json`
- [x] Leakage tests (7 structural tests)

**Phase 3 gate:** leakage tests green; features proven invariant to future data. ✅

## Phase 4 — Baselines and candidates

- [x] Seasonal-naive baseline + moving-average baseline
- [x] Random Forest pipeline
- [x] HistGradientBoosting pipeline
- [x] Chronological splits + rolling-origin folds
- [x] Artifact bundle writer producing a manifest that `ArtifactStore` accepts (12 files, SHA-256)
- [x] `scripts/train_pipeline.py` end to end

**Phase 4 gate:** `ArtifactStore.load()` succeeds on the trained bundle. ✅

## Phase 5 — Evaluation and error analysis

- [x] MAE / RMSE / WAPE on identical test rows for all candidates
- [x] Selection rationale recorded in `metrics.json` (rolling-origin mean WAPE)
- [x] Error analysis by category, demand decile and sparsity
- [x] Residual/interval diagnostics (empirical quantiles, method disclosed)

## Phase 6 — Expiry risk + FEFO

- [x] FEFO allocator with documented tie-breaking
- [x] Risk score + severity bands from configuration
- [x] Realised-outcome labels from a held-out future window (censoring handled)
- [x] Precision / recall / F1 / confusion matrix (0.818 / 0.900 / 0.857)
- [x] False-positive vs false-negative business cost discussion

## Phase 7 — Inventory optimisation + explainability

- [x] Reorder quantity with lead time + safety stock
- [x] Coverage days, excess, overstock detection
- [x] Rotation guidance (7-branch ordered rule table)
- [x] Global feature importance (permutation) + per-prediction ablation drivers
- [x] Uncertainty stated with its method

## Phase 8 — Database + real API

- [x] ORM entities: products, sales, inventory, batches, forecasts, risk, recommendations, dataset versions
- [x] `scripts/init_db.py` + sample-data loader + persisted scoring snapshot
- [x] Every stub replaced with a real service
- [x] API integration tests against the trained bundle

## Phase 9 — Dashboard

- [x] Theme + layout components (typography, spacing, status chips)
- [x] All 7 pages implemented
- [x] Loading / error / empty states wired to real API responses
- [x] Every page rendered successfully via Streamlit AppTest (0 exceptions)

## Phase 10 — Integration and hardening

- [x] End-to-end test: dashboard → API → artifact → number
- [x] Break-it drills executed (checksum tamper, horizon mismatch, missing bundle)
- [x] FEFO allocation bug found and fixed with a regression test

## Phase 11 — Documentation and deployment

- [x] README sections 1–22 complete
- [x] Technical report with diagrams and measured results
- [x] Dockerfile + Compose (⚠ not built — no daemon available)
- [x] Locked dependency set (`requirements.lock.txt`)

## Phase 12 — Demo and review

- [x] 5-minute demo script, rehearsed structure
- [x] Interview question bank with answers
- [x] Independent rubric review with evidence per dimension (`reports/review.md`)
