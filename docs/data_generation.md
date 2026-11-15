# Data Generation — Assumptions and Logic

The dataset is produced by `ml/data_generation/` and is **entirely synthetic**. This
document states every assumption so that no result can be mistaken for an observation about
a real pharmacy.

Reproduce: `python -m scripts.generate_data` (seed 42 by default). Determinism is asserted
by `tests/unit/test_data_pipeline.py::test_generation_is_deterministic`.

---

## 1. What is simulated, and why

| Element | Simulated how | Rationale |
|---|---|---|
| Product catalogue | 120 products across 8 categories, names composed from ingredient + strength + form | Enough breadth for per-category analysis without an unmanageable catalogue |
| Category demand level | `base_daily_units` per category (4.2–13.0) | Chronic-use categories (cardiovascular, diabetes) sell steadily; dermatology is niche |
| Weekly pattern | Category-specific 7-day multipliers, 0.62–1.20 | Clinic-driven categories dip at weekends; OTC categories rise |
| Annual seasonality | Sinusoid with category-specific amplitude (0.06–0.72) and peak day | Antibiotics peak in winter, antihistamines in spring, vitamins in January |
| Product-level scale | `lognormal(0, 0.38)` clipped to `[0.25, 3.0]` | Products within a category differ substantially in popularity |
| Trend | Per-product slope in `[-0.25, +0.35]` per year | Some products grow, some decline |
| Promotions | 3% of product-days with 1.5–3.0× uplift and 5–25% discount | Demand spikes; also creates the sub-cost sales that validation flags |
| Intermittency | 18% of products have active-day probability 0.25–0.60 | Slow movers produce genuine zero-demand days |
| Noise | Poisson counts around the mean | Integer counts with variance ≈ mean, appropriate for unit sales |
| Supplier lead time | 4–12 days plus jitter, per supplier | Realistic variation in replenishment lag |
| Inventory posture | 20% short cover (5–15 days), 50% normal (20–45), 30% excess (60–150) | Both under-stock and over-stock must exist for the decision layer to matter |
| Batches per product | 1–4, quantities split multinomially | Real shelves hold multiple lots; FEFO needs more than one to be meaningful |
| Batch expiry | 30% within 12–90 days, 40% within 90–240, 30% within 240–520 | A third of batches fall inside the actionable rotation window |
| Shelf life | 240–1080 days by category | Longer for stable formulations |

## 2. Structural invariants

Enforced in the generator and asserted in tests:

1. `sum(batch.remaining_quantity) == inventory.current_stock` for every product.
2. `manufacturing_date < expiry_date` for every batch.
3. `batch_arrival_date <= snapshot_date` (stock cannot arrive from the future).
4. `0 <= remaining_quantity <= batch_quantity`.
5. `quantity_sold > 0` and `selling_price > 0` for every transaction.
6. `transaction_id` and `batch_id` are unique.
7. The demand panel is a complete product × day grid with no duplicates.

## 3. Explicit assumptions and their consequences

| # | Assumption | What it means for results |
|---|---|---|
| A1 | Demand is generated from a known, smooth functional form | Forecast accuracy measures recovery of that form. It is an upper bound on real-world performance |
| A2 | No stockout censoring | A sold-out day would record zero, but the simulator models availability as unconstrained. Real pharmacies face the opposite problem: demand is hidden by stockouts |
| A3 | Promotions are independent of stock | Real promotions are often *caused* by expiring stock, which would create reverse causality |
| A4 | Price does not affect demand | No elasticity; discount columns are descriptive, not causal |
| A5 | Demand is stationary within a product's trend | No structural breaks, no new competitors, no formulary changes |
| A6 | One inventory snapshot only | Historical stock is unknowable, so stock is excluded from the forecasting model (D-011) |
| A7 | Lead times are deterministic per supplier | Real lead times have heavy tails and occasional non-delivery |
| A8 | Batch quality is binary at expiry | No degradation, recall or partial-quantity dispensing |
| A9 | The calendar is the only exogenous signal | No weather, epidemiology, holidays beyond an October–November window flag |

## 4. Why these choices are defensible

* Every parameter is **named and inspectable** in `GeneratorConfig` / `CATEGORY_PROFILES`;
  there are no magic numbers in the generation code.
* The generator produces the **failures the project exists to address** (simultaneous
  over- and under-stock, near-expiry batches, sparse products), rather than flattering a
  model.
* Reproducibility is enforced by seeded child streams (`SeedSequence` per stage), so
  adding a draw in one stage cannot silently change another.
* Every output surface discloses synthetic provenance (API field, dashboard footer,
  manifest flag, report statements).

## 5. What would be required to use real data

1. A licensed extract with `product_id`, `transaction_date`, `quantity`, and **stockout
   indicators or daily stock levels** (to un-censor demand).
2. Batch ledger with manufacturing and expiry dates — usually available from the
   pharmacy's ERP/stock system.
3. Supplier lead-time history, not just the contractual value.
4. Promotion calendar, so uplift is observable rather than inferred.
5. Re-validation of every threshold in `ml/expiry_risk/risk.py` against realised outcomes;
   the policy table is calibrated to this simulator.
