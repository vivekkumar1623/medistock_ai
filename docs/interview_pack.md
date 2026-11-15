# Interview Pack

Answers you should be able to give without notes. Each answer names the file that proves it.

---

## 1. Why this problem?

Inventory decisions in a pharmacy are a genuine two-sided optimisation: stock out and you
lose the sale, over-stock and the product expires. Most small pharmacies resolve this with
trailing averages and memory. The problem is attractive technically because it forces you to
combine forecasting, a business rule (FEFO), and a decision layer — and because the failure
modes are measurable in currency rather than in abstract loss.

*Proof: `reports/technical_report.md` §1, §12.*

## 2. How was the data generated, and why synthetic?

No licensed dataset with batch-level expiry and per-product sales was available. The
simulator (`ml/data_generation/`) encodes documented assumptions: category demand levels,
weekday profiles, annual seasonality, per-product scale and trend, promotions,
intermittency, supplier lead times, and multi-batch inventory with varied expiry windows.
Every parameter is named in `GeneratorConfig`, and determinism is enforced with per-stage
seeded streams.

The honest consequence: accuracy here measures recovery of a known signal. It is an upper
bound, not a real-world estimate.

*Proof: `docs/data_generation.md`, `ml/data_generation/config.py`.*

## 3. Why these models?

Two baselines and two candidates. Seasonal naive (next 30 days = last 30) is genuinely
strong for an aggregate horizon — it absorbs level and seasonality to first order, so if
your ML cannot beat it, the ML is not earning its complexity. Random Forest handles
non-linear interactions and is robust to outliers. HistGradientBoosting is usually the
strongest tabular learner and handles missing values natively.

I did not use an LSTM or Prophet because with ~2,400 training rows of largely
level-and-seasonality data they add dependencies and tuning risk without an expected gain —
and I would not be able to defend the extra complexity.

*Proof: `ml/forecasting/pipeline.py`, `ml/forecasting/baseline.py`.*

## 4. How did you prevent future leakage?

Structurally, not by discipline:

1. The demand matrix is a prefix-sum array; a feature is `cumsum[origin+1] - cumsum[start]`.
   There is no API to read past `origin` while building features.
2. The target is computed by a **separate function** (`compute_target`) that is only called
   when building training rows.
3. All preprocessing lives inside the sklearn `Pipeline`, so it can only be fitted on
   training folds.
4. Splits are chronological by origin date, and the split function raises if partitions
   overlap.
5. Expiry labels come from *realised* future sales and are never part of the bundle that
   serving reads.

The decisive test builds features at an origin twice — history-only versus the full panel —
and asserts the two frames are identical. If any feature ever reads the future, that test
fails.

*Proof: `tests/unit/test_features_no_leakage.py`, `docs/leakage.md`.*

## 5. How is expiry risk calculated?

```
allocatable(b) = max(0, cumulative_demand_at_expiry(b) - demand absorbed by earlier batches)
estimated_unsold(b) = max(0, remaining(b) - allocatable(b))
at_risk = estimated_unsold > 0
```

FEFO means earlier expiry is offered demand first. Severity is an explicit 2-D table:
unsold ratio × days to expiry, with thresholds from configuration. A batch with zero
exposure is always `low`. There is no learned classifier here — the rule is transparent and
auditable, which is the right choice for a decision a human must defend.

**The bug worth mentioning:** my first version advanced "demand consumed" by each earlier
batch's *remaining* quantity. But expired units are discarded — they cannot have absorbed
demand that arrives later — so every later batch looked at risk. A test caught it; the fix
was to advance by *allocated* (actually sold) quantity, on both the prediction and label
paths.

*Proof: `ml/expiry_risk/fefo.py`, `ml/expiry_risk/risk.py`, `tests/unit/test_expiry_risk.py`.*

## 6. How did you evaluate, and why WAPE?

Chronological split: train (11 origins), validation (5), test (4). The test set is scored
**once**, after selection.

I report MAE (units, what a manager reasons in), RMSE (punishes large misses — dangerous for
expiry decisions), and WAPE = `sum|error| / sum|actual|`. MAPE is unusable because this data
contains genuine zero-demand days where it is undefined. WAPE only needs the aggregate
denominator to be non-zero, and when that denominator *is* zero the code returns `None`
rather than 0.0, because 0.0 would look like a perfect model.

WAPE is volume-weighted, so high-sellers dominate it; that is deliberate for an inventory
decision, and the per-category error analysis covers the small products.

*Proof: `ml/evaluation/metrics.py`, `reports/error_analysis.md`.*

## 7. Why did you change the selection metric mid-build?

Because the first version was fragile. Single-window validation chose Random Forest over
HistGradientBoosting by 0.001 WAPE on 600 rows — statistically meaningless — while the
rolling-origin folds favoured HGB consistently (0.125/0.117/0.112 versus
0.122/0.123/0.132).

I switched to mean WAPE across three expanding-window folds that end **before** the test
period, and asserted in code that those folds never include test origins. That is still
validation, not test peeking. HGB was then selected at 0.118 mean, and the untouched test
set confirmed it at 0.108.

This is the honest answer to "did you tune on test?" — no, and here is the commit-level
evidence of why the method changed.

*Proof: `ml/forecasting/train.py::_rolling_origin_evaluation`.*

## 8. How does the API load the model?

`ArtifactStore.load()` reads `manifest.json`, checks the contract major version, verifies
the SHA-256 of every file, checks required files are present, and requires the bundle's
horizon to equal the configured horizon. Only then does it unpickle the pipeline. Result is
cached per artifact version.

If the bundle is missing → `503 artifact_not_ready`. If it is corrupt → `500
artifact_integrity_error`. There is no fallback to a default prediction anywhere.

*Proof: `app/services/artifact_store.py`.*

## 9. How does the dashboard consume predictions?

Over HTTP. The dashboard never imports the ML pipeline and never reads a CSV. `httpx` calls
`/api/v1/...`, responses are validated by Pydantic models, and the cache is explicitly
invalidated by a "Refresh data" button. If the API is down, pages render an error state
that says so rather than showing placeholder numbers.

Both services read the same bundle through the same code path, so "API says 42, chart says
39" cannot happen.

*Proof: `dashboard/api_client.py`, `dashboard/streamlit_app.py`.*

## 10. What would you change before commercial deployment?

1. Real data with **stockout flags** — sales are censored by availability, and that is the
   single biggest unmodelled effect.
2. Probabilistic forecasts (quantile regression) with verified coverage, instead of
   residual-quantile intervals.
3. Scheduled retraining plus drift monitoring; the current artifact is a fixed snapshot.
4. Auth, audit logging, role-based access.
5. Promotion and supplier-shock features — the largest errors sit on promotion spikes the
   feature set cannot see.
6. Replace the nightly snapshot with ERP integration for batch-level truth.

*Proof: `reports/technical_report.md` §15, `LIMITATIONS.md`.*

---

## Rapid-fire questions

| Question | Answer |
|---|---|
| What is FEFO? | First Expiry, First Out: dispense the soonest-expiring batch first. FIFO (first received) is the accounting convention and is wrong for perishables |
| Why not predict daily demand? | The decision is periodic (order this month?), daily forecasts are much noisier, and expiry needs a rate — which the aggregate gives directly |
| What if a product has no sales history? | It cannot be a test row; at serving time the feature row needs ≥30 days of history, and unknown products return 404 |
| What does `confidence: low` mean? | A documented heuristic: >50% zero-demand days, forecast <5 units, or a very wide interval. It is a flag, not a calibrated probability |
| Are the SHAP values real? | No — there are no SHAP values. Per-prediction explanations are one-at-a-time ablation against a population reference, and the API labels the method |
| How do you know the artifact matches the code? | Manifest contract version + per-file SHA-256 + horizon lock, all verified before load |
| What happens on a 14-day request? | `422 invalid_request` naming both the requested and trained horizons. The model refuses to answer a question it was not trained for |
| Biggest limitation? | Synthetic data with no stockout censoring. Every accuracy number is an upper bound |
