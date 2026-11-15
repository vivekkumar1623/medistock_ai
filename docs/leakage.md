# Leakage Analysis

Leakage is the failure mode that makes a forecast look excellent in a notebook
and useless in production. This document states the threat model, the rules that
prevent it, and where each rule is enforced in code. If a reviewer asks "how do
you know there is no future information in your features?", this file is the
answer.

## 1. Definition used here

> Leakage is any information available to the model at prediction time that would
> **not** be available at that same moment in a real pharmacy on that date.

Two independent axes:

* **Temporal leakage** — using an observation dated after the prediction origin.
* **Preprocessing leakage** — statistics (means, scales, encodings, imputation
  values, hyperparameters) computed using data that includes the evaluation set.

## 2. Prediction origin is the anchor

```
        history only                    target window
 ──────────────────────────────┬──────────────────────────────►
      ...──────────────────►   T │ T+1 ................ T+30
                            origin
```

* Origin `T` = the last date whose data may be used.
* Target = total units sold for the product in `(T, T+30]`.
* Target construction reads only rows with `date > T`.
* Feature construction reads only rows with `date <= T`.

Every training row is produced by *(product, origin)* pairs on a fixed calendar
grid. The same code path builds training rows and live inference rows; the only
difference is that live inference uses the bundle's frozen history.

## 3. Rules and enforcement

| # | Rule | Enforced by |
|---|---|---|
| R1 | Features only from `date <= origin` | `ml/features/build.py` asserts `max(feature_date) <= origin` per row |
| R2 | Target only from `date > origin` | `ml/features/targets.py` masks `date > origin` before summing |
| R3 | No target-derived column may appear as a feature | `feature_spec.json` allow-list; `assert_no_target_features()` |
| R4 | All preprocessing fitted inside the sklearn `Pipeline` | `ml/forecasting/pipeline.py` — imputer/scaler/encoder are pipeline steps, fitted only on train folds |
| R5 | Chronological splits only | `ml/forecasting/splits.py` refuses overlapping or shuffled date ranges; no `train_test_split` call in the repo |
| R6 | Encode categories from a frozen vocabulary | `feature_spec.json` stores the category levels; unknown categories map to `__unknown__`, never to a new column |
| R7 | Hyperparameters chosen on validation, never on test | `ml/forecasting/train.py` writes val metrics to select, then evaluates the frozen model once on test |
| R8 | Expiry-risk labels come from realised outcomes after the origin | `ml/expiry_risk/labels.py` computes unsold from the *future* window; the risk model sees only origin-time features |

## 4. The three leaks that pass code review

These are the ones to look for in an interview; they all produce plausible
numbers rather than errors.

**L1 — The scaling leak.** A `StandardScaler` fitted on the full dataset before
splitting. Accuracy improves because the mean and variance encode test-period
information. *Prevented by R4*: the scaler is a pipeline step, so
`pipeline.fit(X_train)` is the only way it can learn.

**L2 — The "last known value" leak.** A feature like `stock_level_at_prediction`
is fine, but `stock_level_after_restock` is not — the restock has not happened at
the origin. *Prevented by*: inventory features are taken from the *snapshot*
table written at the origin, not from the live inventory table.

**L3 — The group leak in the label.** For expiry risk, computing
`estimated_unsold` and then using any component of it (e.g. `days_to_expiry × forecast_demand`)
as a *feature* makes the label trivially recoverable. *Prevented by R8*: risk
scoring may use forecast, stock, quantity and dates (all origin-time inputs) but
never the realised-sales window used to build the label.

## 5. What is deliberately *not* leakage

Being precise here matters, because over-correcting destroys signal.

* Using the product's own long history (including periods before the origin) is
  legitimate — it is known at the origin.
* Using the supplier's contracted lead time is legitimate — it is on the purchase
  agreement, not a future outcome.
* Using the batch's expiry date is legitimate — it is printed on the box.
* Using calendar features for future dates (day of week, month, holiday flags) is
  legitimate *because the calendar is known in advance*. The rule is "no future
  **outcomes**", not "no future-dated values".

## 6. Verification plan (tests that fail if leakage returns)

```
tests/regression/test_no_leakage.py
  ├─ test_features_never_exceed_origin()
  ├─ test_target_never_precedes_origin()
  ├─ test_feature_spec_has_no_target_columns()
  ├─ test_pipeline_preprocessing_fitted_on_train_only()
  ├─ test_split_boundaries_are_chronological_and_disjoint()
  └─ test_expiry_label_not_reachable_from_features()
```

A test that only checks "the model is accurate" cannot detect leakage. These
tests assert *structural* properties of the data, which is the only reliable
defence.

## 7. Known residual risk

* With a single 30-day test window, a lucky split can still flatter a model.
  Mitigation: rolling-origin folds, reported alongside the headline metrics.
* Synthetic data has no censoring (see `LIMITATIONS.md` 1.2), so the "stockout
  hides demand" leak cannot occur here — but it *will* occur on real data, and
  the pipeline would need demand un-censoring before it could be trusted.
