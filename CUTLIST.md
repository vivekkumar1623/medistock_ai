# Cut List (pre-committed)

Written **before** implementation, so that scope decisions under deadline are
decisions, not panic. Rule: Tier-3 items are cut the moment a Tier-1 checkpoint
slips. Nothing on the "Keep" list is ever traded for a Tier-3 item.

## Never cut (Tier 1 — 75% of the rubric)

| # | Item | Where it lives |
|---|---|---|
| 1 | Reproducible synthetic generator with documented assumptions | `ml/data_generation/` |
| 2 | Leakage analysis (explicit, written, with the rule enforced in code) | `docs/leakage.md` |
| 3 | Chronological train/val/test split + rolling-origin validation | `ml/forecasting/splits.py` |
| 4 | Baseline vs candidate comparison on one untouched test set | `ml/evaluation/` |
| 5 | Real MAE / RMSE / WAPE from executed evaluation | `artifacts/*/metrics.json` |
| 6 | FEFO expiry-risk math with precision/recall/F1/confusion matrix | `ml/expiry_risk/` |
| 7 | One clean end-to-end inference path: artifact → service → HTTP → UI | `artifacts/` → `app/services/` → `app/api/` → `dashboard/` |
| 8 | Technical report + demo video | `reports/` |
| 9 | Honest `LIMITATIONS.md` | repo root |

## Cut order (first to go when behind)

| Order | Item | Degradation if cut | Compensating evidence |
|---:|---|---|---|
| 1 | Alembic migrations | Schema created with `SQLAlchemy.metadata.create_all` | README section "Migrations not used; schema is created idempotently" |
| 2 | Docker Compose | `Dockerfile` only, single service | README shows `docker run` for API and dashboard |
| 3 | Dashboard 7 pages → 3 | Executive, Forecast, Expiry only | Remaining views documented as "planned" in README |
| 4 | SHAP | Permutation importance + one manual worked example printed from the bundle | `docs/explainability.md` shows the arithmetic by hand |
| 5 | Full test pyramid → ~15 focused tests | Unit + contract + one integration | Test report states exactly what is and is not covered |
| 6 | Full SQLAlchemy ORM | Parquet-backed read layer | "SQLite via SQLAlchemy is future work" note |

## Tier-3 polish (do only after Tier-1 is green)

* Executive dashboard visual refinement (typography, spacing, status chips).
* Extra pages: Product Explorer, Model Performance, Data Quality & System Health.
* Docker Compose with health checks.
* Alembic baseline migration.
* CI configuration.

## Tripwires

If by the end of a working block any of these is true, cut immediately:

* No executed metric in `artifacts/*/metrics.json` → cut dashboard polish and finish the model.
* `/api/v1/forecast/{id}` does not return a real artifact-backed number → cut Docker and finish inference.
* `LIMITATIONS.md` still contains `TODO` → cut extra dashboard pages and write the report.
* Report or video not started → stop all feature work.
