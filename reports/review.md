# Independent Engineering Review

Self-assessment against the six grading dimensions. Deliberately **not** inflated: each
score names the evidence for it and the specific gap that stopped it going higher.

**Overall: 83 / 100** (self-assessed, not awarded).

---

## Dimension 1 — Problem formulation and data engineering · 12.5 / 15

**Complete**

* Problem split into three linked tasks (demand forecasting, batch expiry risk, decision
  support) with an explicit prediction origin and horizon (`reports/technical_report.md` §1).
* Seeded simulator with every assumption as a named parameter
  (`ml/data_generation/config.py`, `docs/data_generation.md`).
* 24 validation checks + separable cleaning pass + re-validation + profile
  (`reports/data_quality_report.json`).
* Structural invariants asserted in tests: batch quantities reconcile exactly with on-hand
  stock, manufacturing precedes expiry, arrivals never come from the future.
* EDA that changed design decisions rather than decorating a page (`reports/eda_report.md`).

**Incomplete / weak**

* Data is entirely synthetic; no real extract was available or licensed.
* Only one inventory snapshot exists, which forced stock out of the demand model (D-011).
* No stockout censoring, so the hardest real-world effect is absent.
* The generator is smoother than reality; measured accuracy is an upper bound.

**To move up:** real data with stockout flags and historical stock snapshots; generator
calibrated against real category demand profiles.

---

## Dimension 2 — Baseline and model development · 20 / 25

**Complete**

* Two genuine baselines (seasonal naive, moving average) plus Random Forest and
  HistGradientBoosting, all as preprocessing-owning sklearn `Pipeline`s.
* Tuning on validation only, with the full grid and all losing configurations recorded in
  `metrics.json`.
* Selection by mean rolling-origin WAPE restricted to pre-test origins, asserted in code.
* Test scored exactly once, after selection.
* Artifact bundle with checksummed manifest, feature spec, metrics, config and snapshots.

**Incomplete / weak**

* Only an **18% relative** improvement over seasonal naive (0.108 vs 0.132). Honest, but not
  a decisive ML win — and on synthetic data a large win would be suspicious.
* No probabilistic model (quantile regression), so intervals remain empirical.
* Systematic category bias remains unaddressed (antihistamines +34.4 units, analgesics −18.0).
* Trees cannot extrapolate beyond the training demand range.
* Four test origins only; the ranking of the top two candidates rests on three folds.

**To move up:** quantile-regression objective with verified coverage; per-category bias
correction; more origins so the selection statistic has a defensible variance.

---

## Dimension 3 — Validation and evaluation · 16.5 / 20

**Complete**

* Strictly chronological splits with an assertion that partitions do not overlap.
* MAE, RMSE and WAPE reported for all candidates on identical rows of each split.
* WAPE chosen over MAPE with a written justification (genuine zero-demand days), and `None`
  returned when the denominator is zero rather than 0.0.
* Error analysis by category, demand decile and sparsity (`reports/error_analysis.md`).
* Expiry risk evaluated against **realised** outcomes with explicit censoring: 197 of 298
  batches excluded rather than counted as safe.
* Critical-band metrics reported separately (precision 1.00, recall 1.00, n=7).

**Incomplete / weak**

* Precision/recall describe 101 evaluable batches — a subset biased toward near-term expiry.
* Prediction intervals are pooled across products of very different scale; per-product
  coverage is unverified.
* No statistical significance test on the model ranking (e.g. paired bootstrap over origins).
* Single dataset version; no evaluation across multiple seeds to show stability.

**To move up:** paired bootstrap or Diebold–Mariano test on the forecast errors; multi-seed
stability run; per-product interval calibration curves.

---

## Dimension 4 — Inference interface · 13 / 15

**Complete**

* `ArtifactStore` verifies contract version, per-file SHA-256, required files and the
  **horizon lock** before unpickling anything.
* One inference path: dashboard → API → bundle → model; the dashboard imports no ML code.
* Ten documented endpoints, one error envelope, meaningful status codes
  (503 no artifact, 500 integrity, 404 unknown product/batch, 422 invalid horizon/origin).
* Every route exercised by tests, with and without a bundle.
* OpenAPI published from the real schemas; no hand-written docs drift.

**Incomplete / weak**

* No authentication or authorisation — localhost only.
* Exactly one prediction origin; a different date is refused rather than served (correct
  behaviour, real limitation).
* No online update path: a new day of sales requires a full pipeline rerun.
* No rate limiting, load test or latency budget recorded.

**To move up:** auth + audit log; incremental scoring; a small locust/k6 load profile with a
documented p95 target.

---

## Dimension 5 — Explainability and responsible AI · 8.5 / 10

**Complete**

* Global permutation importance on the test set (not impurity importance on train).
* Per-prediction one-at-a-time ablation with `value`, `reference_value`, `contribution`,
  `direction`, `method` and a plain-language explanation.
* The API and dashboard both state these are **not** Shapley values and are **not** additive.
* Zero-contribution features are omitted rather than reported as influential.
* Prediction interval method disclosed (`empirical_test_residual_quantiles_10_90`) with the
  caveat that coverage is not verified per product.
* Explicit non-clinical scope on every forecast response; synthetic provenance exposed via
  `DataFreshness.is_synthetic` and the dashboard footer.
* Confidence is a documented heuristic (sparsity, magnitude, interval width), labelled as
  such rather than as a probability.

**Incomplete / weak**

* No SHAP — a deliberate Tier-3 cut, but an additive attribution is a legitimate request a
  regulator might make.
* Confidence heuristic is not calibrated against realised error.
* No counterfactual or "what would change this recommendation" view beyond the ablation bars.

**To move up:** calibrated confidence via observed error by bucket; optional SHAP with a
documented background sample; a counterfactual panel on the recommendation detail.

---

## Dimension 6 — Reproducibility, testing and documentation · 12.5 / 15

**Complete**

* Seeded generation with independent per-stage streams; determinism asserted by test.
* `requirements.lock.txt` with exact verified versions; training library versions recorded in
  every manifest.
* 99 tests: configuration, artifact integrity, leakage (7 structural), generator
  determinism, FEFO/severity, replenishment rules, all API endpoints, and all 7 dashboard
  pages rendered via Streamlit `AppTest`.
* Documentation: README (22 sections), technical report, architecture, leakage analysis,
  data dictionary, data-generation assumptions, model card, ADR log, limitations, EDA and
  error-analysis reports, demo script, interview pack.
* Three genuine defects were caught by tests during the build, including a real FEFO
  allocation bug — all documented in `reports/test_report.md`.

**Incomplete / weak**

* **Docker was never built** — no daemon available. Files are valid and mirror the verified
  native commands, but the image build is unverified. This is the clearest gap.
* No CI pipeline; tests run locally only.
* No fresh-venv reproducibility dry run from a clean clone.
* Alembic installed but unused (pre-committed cut).
* 99 tests is modest; coverage percentage is deliberately not gated.

**To move up:** GitHub Actions running lint + tests + a pipeline smoke test; a `docker build`
verified in CI; a clean-clone dry run recorded as evidence.

---

## Summary

| Dimension | Weight | Score | Evidence quality |
|---|---:|---:|---|
| Problem formulation and data engineering | 15% | 12.5 | Strong, with an unavoidable synthetic-data caveat |
| Baseline and model development | 25% | 20.0 | Complete methodology, modest measured margin |
| Validation and evaluation | 20% | 16.5 | Rigorous protocol, limited origins |
| Inference interface | 15% | 13.0 | Strong contract enforcement, no auth |
| Explainability and responsible AI | 10% | 8.5 | Honest labelling, no calibrated uncertainty |
| Reproducibility, testing and documentation | 15% | 12.5 | Extensive docs, Docker unbuilt, no CI |
| **Total** | **100%** | **83.0** | |

## The three things that decide pass/fail

1. **Fresh clone → working prediction.** The pipeline runs end to end from a locked
   dependency set; the artifact bundle is committed so the demo needs no retraining. Partially
   verified — a clean-clone dry run was not performed in a brand-new venv.
2. **Explain one prediction and one error on camera.** Available and rehearsed in
   `docs/demo_script.md`: MED-0001 at 151.6 units driven by `sum_30`, and the antihistamine
   category over-forecast of +34.4 units explained by peak-shoulder smoothing.
3. **Honest limitations.** `LIMITATIONS.md` names counts, not adjectives: 197 censored
   batches, +34.4 unit bias, 2.6× worse WAPE on sparse products, unbuilt Docker image.

## What I would fix first with more time

1. Build and verify the Docker image (closes the largest documentation gap).
2. Add CI (lint + tests + pipeline smoke) — makes every claim above reproducible by a third party.
3. Per-category bias correction, since the error analysis shows a clear, fixable pattern.
4. Paired bootstrap over origins so "HGB beats RF" has a confidence interval attached.
