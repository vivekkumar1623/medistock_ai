# Demo Script (≈5 minutes)

Rehearse this twice. It is written to be spoken, not read.

---

## 0. Before you start (2 minutes before recording)

```powershell
cd medistock_ai
.\.venv\Scripts\Activate.ps1
.\scripts\run_api.ps1          # terminal 1 — leave running
# terminal 2:
streamlit run dashboard/streamlit_app.py
```

Pre-open: `http://127.0.0.1:8000/docs` and `http://127.0.0.1:8501`.
Check `/health` says `ok` (if it says `degraded`, the database is not initialised — run
`python -m scripts.init_db --with-scores`).

**Have these numbers ready** (they are in the current build; re-check before recording):
6,058 estimated unsold units · 24 at-risk batches · test WAPE 0.108 · expiry precision 0.82 /
recall 0.90.

---

## 1. The problem (30 s)

> "A pharmacy has two ways to lose money on the same shelf: stock out and lose the sale, or
> over-stock and watch it expire. Most small pharmacies manage this with trailing averages
> and memory. MediStock AI forecasts 30-day demand per product, works out which batches will
> not clear before expiry, and turns that into reviewable actions."

## 2. Executive Overview (45 s)

Show the KPI row.

> "Six numbers, all from the loaded model artifact at one prediction origin: 120 products,
> 65,060 units on hand worth about 8.9 million at cost, a 30-day forecast, 24 batches at
> risk, and about 955 thousand of estimated exposure at cost.
>
> Note what is *not* here — no fake sparklines. Every figure traces to the API."

Point at the recommendation mix donut.

> "120 recommendations: 8 reorder, 10 urgent rotation, 11 expiry exposure, 17 overstock,
> 74 sufficient. Both failure modes exist at once, which is the point."

## 3. Demand Forecasting (60 s)

Pick a product. Point at the forecast, interval, confidence.

> "151.6 units over 30 days, interval 103 to 190. The interval is empirical residual
> quantiles from the held-out test period — not a made-up ±20%.

Point at the driver bars.

> "This is one-at-a-time ablation: replace one feature with the population median, re-predict,
> and see how far the forecast moves. `sum_30` dominates, which is exactly right for a
> 30-day aggregate. These are not SHAP values and the API says so."

Point at the comparison table.

> "HistGradientBoosting, test WAPE 0.108 against 0.132 for a seasonal-naive baseline — an
> 18% reduction. The test set was scored once, after model selection."

## 4. Expiry Intelligence (60 s)

> "298 batches assessed at the snapshot. 24 at risk, 6,058 units, about 955 thousand at cost.
>
> The rule is explicit: FEFO gives each batch the demand available before its expiry, minus
> what earlier batches actually sold. Unsold is what is left. Severity is a published 2-D
> table — unsold ratio by days remaining."

Open the audit expander for one batch.

> "Every flag shows its arithmetic. A manager can disagree with the model intelligently,
> because the numbers are on screen."

> "Offline, against realised outcomes on the 101 batches we could actually observe:
> precision 0.82, recall 0.90, and 7 of 7 critical batches correctly identified. I
> prioritised recall deliberately — a missed expiry costs the full unit cost, an unnecessary
> rotation costs staff time."

## 5. Inventory Optimisation (40 s)

Filter to `reorder`.

> "Reorder point is lead-time demand plus safety stock, z=1.65 for roughly 95% service level.
> Suggested quantity is a suggestion — no order is placed anywhere in this system."

Open the arithmetic expander.

> "Here is the whole calculation, per product."

## 6. Model Performance + Health (40 s)

> "Confusion matrix, permutation importance, and the rolling-origin folds. Selection used the
> fold mean, not one window, because the single-window difference between the top two models
> was 0.001 WAPE — noise."

Switch to Data Quality & Health.

> "Live component health, the artifact's SHA-256 manifest, and the data-quality report from
> the pipeline. Zero validation errors; one warning about 46 below-cost sales on promotion
> days, which we kept and reported."

## 7. Close (30 s)

> "Three things I'd emphasise.
>
> One: the artifact is checksummed and horizon-locked. Ask it a 14-day question and it
> refuses.
>
> Two: there is one inference path. Dashboard → API → bundle → model. No second copy of the
> logic anywhere.
>
> Three: the honest limits. Synthetic data with no stockout censoring, so every accuracy
> number is an upper bound. FEFO assumes perfect rotation discipline, so real exposure is
> probably worse. 197 of 298 batches could not be labelled within the observed window, and
> I excluded them rather than counting them as safe."

---

## If something breaks

| Symptom | Fix |
|---|---|
| Pages show "Cannot reach the API" | API not running — `.\scripts\run_api.ps1` |
| `/health` says `degraded` | `python -m scripts.init_db --with-scores` |
| Everything 503 | No artifact bundle — `python -m scripts.train_pipeline` then `run_expiry_risk` |
| Charts empty | Click "Refresh data" in the sidebar (cache) |

## Do not do

* Do not claim real-world accuracy from synthetic data.
* Do not call the driver bars "SHAP values".
* Do not say the system "orders stock" or "recommends medication".
* Do not quote a metric you have not just displayed.
