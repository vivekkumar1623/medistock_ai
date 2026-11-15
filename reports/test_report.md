# Test Report

**Build:** complete (Phases 1–12)
**Date:** 2026-11-14
**Environment:** Windows 11 · Python 3.11.4 · pandas 3.0.2 · numpy 2.4.4 · scikit-learn 1.8.0
**Command:** `python -m pytest`

---

## Result

```
........................................................................  [ 78%]
....................                                                     [100%]
99 passed
```

Lint: `python -m ruff check .` → **All checks passed!** (Phase 1 baseline; re-verified at the
end of the build).

The only warning is Starlette's own `testclient.py` `anyio.abc.BlockingPortal` deprecation —
third-party, not project code.

## What each area proves

| Area | Tests | Guarantee |
|---|---:|---|
| Configuration | 9 | Env-prefix isolation (a stray `RANDOM_SEED` cannot hijack the pipeline), path resolution against repo root, bounds validation, log-level normalisation |
| Artifact contract | 12 | Missing bundle → 503 not a crash; cache identity; checksum mismatch, post-signing tamper, missing file, contract-major mismatch, horizon lock, version mismatch all rejected; `status()` never raises |
| Leakage | 7 | Features invariant to future data; target equals actual future demand; target unobservable past data end; target columns rejected as features; inventory columns excluded; origin outside panel rejected; window feature equals manual computation |
| Data pipeline | 12 | Generator determinism (same seed → identical frames), different seed changes data, batch↔stock reconciliation, date coherence, valid transactions, complete panel grid, panel↔ledger totals, weekday structure, WAPE formula/zero-guard/bias |
| Expiry risk | 14 | FEFO ordering, cumulative allocation, expired-units regression guard, multi-product independence, unknown product, deterministic tie-break, extrapolation cap, 10-case severity table, consistent columns, empty input |
| Inventory logic | 12 | Safety-stock formula, zero-lead-time, documented intermediates, no-order when covered, zero-demand edge case, negative clamp, all seven rule branches, inputs/reasons present |
| API contract | 16 | `/health` with and without a bundle; one error envelope for every failure; `/model/info` and `/model/metrics` serve real manifest data; products/forecast/inventory/expiry/recommendations all serve artifact-backed data; 404 unknown product, 422 horizon mismatch with both horizons, 503 without bundle; OpenAPI publishes all 10 paths |

## Bugs found and fixed during the build

| # | Symptom | Root cause | Fix |
|---:|---|---|---|
| 1 | 2 tests failed: `/health` gave `degraded`, test expected `ok` | Test asserted an all-green system while the SQLite file did not exist | Materialise a temp DB in the fixture; `degraded` is the correct answer for a half-initialised deployment |
| 2 | `test_artifact_path_combines_root_and_version` failed (`C:/tmp` vs `/tmp`) | Unix absolute path used on Windows, where `/tmp` is drive-relative and pathlib re-anchors it | Use `tmp_path` |
| 3 | 13 ruff violations | Long `Field(...)` lines, `typing.Iterator` imports | `ruff check --fix` + manual splits |
| 4 | **FEFO allocated too much risk to later batches** | Consumption cursor advanced by earlier batches' *remaining* quantity, but expired units are discarded and cannot absorb later demand | Advance by *allocated* (sold) quantity, on both prediction and label paths; regression test added |
| 5 | Panel↔ledger reconciliation test produced 1,041,111 rows | Joined on index level `transaction_date` against `date` → pandas cross-join | Rename the index level before joining |
| 6 | `inventory` dashboard page raised `KeyError: stock_status` | Column renamed to "Status" *before* styling referenced the old name | Style after renaming, referencing the displayed name |
| 7 | `model_performance` page raised `KeyError: critical_band` | UI read a field the API schema did not expose | Added `CriticalBandMetrics` to `ExpiryMetrics` and populated it in the router |
| 8 | Forecast endpoint returned `prediction_interval: null` | Bundle predated the residual-quantile change | Retrained; artifact regenerated correctly |
| 9 | `AppTest.from_function` reported `page_header is not defined` on all pages | `from_function` does not carry module-level imports — a harness artefact, not a bug | Rendered pages from real script files on disk instead |

## Acceptance evidence

| Claim | Evidence |
|---|---|
| Reproducible generation | `test_generation_is_deterministic` (identical frames from identical seeds) |
| No future leakage | `test_features_are_invariant_to_future_data` |
| Real metrics | `artifacts/v1/metrics.json` (test WAPE 0.108; 4 models, identical rows) |
| Expiry metrics | `artifacts/v1/metrics_expiry.json` (P 0.818 / R 0.900 / F1 0.857) |
| Inference uses the saved artifact | `test_forecast_endpoint_runs_the_loaded_model`; dashboard rendered from live API |
| Errors are honest | `test_endpoints_requiring_a_model_are_503_without_a_bundle` |
| Bundle cannot answer a different question | `test_horizon_mismatch_is_rejected_with_an_explanation` |
| Dashboard pages render | all 7 rendered via Streamlit `AppTest` with 0 exceptions |

## Not covered (stated, not hidden)

* **Docker build** — not executed; no daemon available in the build environment.
* **Alembic migrations** — not used by design (CUTLIST item 1).
* **Concurrency/load** — no load test; SQLite single-node assumed.
* **Coverage percentage** — deliberately not gated; a coverage number is a weak proxy for the
  properties that matter here (integrity, leakage, determinism).

## Honest note on test count

99 tests is not a large suite by industry standards. The claim is not "comprehensive"; it is
that each test asserts a *property that would otherwise be a silent failure* (leakage,
integrity, reconciliation, rule ordering), and that three genuine defects were caught by it
(#4, #6, #7) before the demo.
