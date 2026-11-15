# Data Dictionary

All data is synthetic. Units are integers unless stated. Dates are calendar dates (no
timezone semantics); timestamps in the artifact manifest are UTC ISO-8601.

---

## `products.csv` (raw) / `products.parquet` (processed)

| Column | Type | Description |
|---|---|---|
| `product_id` | string | Primary key, `MED-0001` … `MED-0120` |
| `medicine_name` | string | Composed from ingredient + strength + form. Fictional |
| `category` | string | One of 8 therapeutic categories |
| `manufacturer` | string | Fictional manufacturer |
| `supplier_id` | string | `SUP-01` … `SUP-06` |
| `unit_price` | float | Selling price per unit (currency units) |
| `unit_cost` | float | Procurement cost per unit, always < `unit_price` in this dataset |
| `minimum_stock_level` | int | Policy floor derived from lead-time demand |
| `lead_time_days` | int | Days from order to delivery |

## `sales.csv` → `sales.parquet`

One row per product-day **with at least one sale**. Zero-sales days are absent here and
recovered in the demand panel.

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | Primary key, `TXN-00000001` … |
| `product_id` | string | FK → products |
| `transaction_date` | date | Sale date, within the dataset window |
| `quantity_sold` | int | Units sold, strictly > 0 |
| `selling_price` | float | Realised price after discount and price noise |
| `discount` | float | Fraction in `[0, 0.25]`; non-zero mainly on promotion days |
| `day_of_week` | int | 0 = Monday … 6 = Sunday |
| `month` | int | 1–12 |
| `season` | string | Winter / Spring / Summer / Autumn |
| `is_weekend` | int | 0/1 |
| `is_festival_window` | int | 0/1, defined as October–November |

## `inventory.csv` → `inventory.parquet`

Snapshot at the dataset's snapshot date (2025-10-01).

| Column | Type | Description |
|---|---|---|
| `product_id` | string | Primary key, FK → products |
| `current_stock` | int | Units on hand at the snapshot; equals the sum of batch remaining quantities |
| `reorder_level` | int | Policy reorder point from the simulator |
| `last_restock_date` | date | Most recent delivery |
| `supplier_lead_time` | int | Mirrors `products.lead_time_days` |

## `batches.csv` → `batches.parquet`

| Column | Type | Description |
|---|---|---|
| `batch_id` | string | Primary key, `BATCH-<product seq>-<batch no>`; batch numbers ascend with expiry (FEFO order) |
| `product_id` | string | FK → products |
| `batch_quantity` | int | Units received in the batch |
| `remaining_quantity` | int | Units still on hand; `0 <= remaining <= batch_quantity` |
| `manufacturing_date` | date | Always strictly before `expiry_date` |
| `expiry_date` | date | Expiry of the batch |
| `batch_arrival_date` | date | Delivery date; always `<= snapshot_date` |

Invariant enforced in tests: `sum(remaining_quantity)` per product equals
`inventory.current_stock`.

## `demand_panel.parquet` (interim + processed)

Derived artefact: the complete product × day grid, zeros included. This is what feature
engineering and expiry labelling consume.

| Column | Type | Description |
|---|---|---|
| `product_id` | string | FK → products |
| `date` | date | Continuous daily calendar over the dataset window |
| `quantity` | int | Units sold that day; 0 is a real observation of no sales |

Row count is exactly `n_products × n_days` (120 × 610 = 73,200).

## `training_frame.parquet` (processed)

One row per `(product, origin)`: 120 × 20 = 2,400 rows.

| Column group | Description |
|---|---|
| `product_id`, `origin_date` | Keys; `origin_date` is the prediction origin |
| 29 feature columns | See `ml/features/build.py::ALL_FEATURES` and `docs/technical_report.md` §3 |
| `demand_next_30d` | **Target**: units sold in `(origin_date, origin_date + 30]` |
| `target_start`, `target_end` | The target window bounds, for auditability |

## Database tables (`app/database/models.py`)

| Table | Purpose | Key constraints |
|---|---|---|
| `products` | Operational catalogue | PK `product_id`; CHECK non-negative prices and lead time |
| `sales` | Transaction ledger (67,558 rows) | PK `transaction_id`; CHECK `quantity_sold > 0`, `0 <= discount < 1`; index `(product_id, transaction_date)` |
| `inventory` | Snapshot stock | PK/FK `product_id`; CHECK non-negative |
| `batches` | Batch ledger | PK `batch_id`; CHECK `0 <= remaining <= batch_quantity`, `manufacturing < expiry`; index `(product_id, expiry_date)` |
| `forecast_results` | Persisted forecast snapshot | UNIQUE `(product_id, as_of_date, artifact_version)` |
| `risk_assessments` | Persisted risk snapshot | UNIQUE `(batch_id, as_of_date)`; CHECK `0 <= risk_score <= 1` |
| `recommendations` | Persisted recommendations | UNIQUE `(product_id, as_of_date)`; CHECK suggested qty >= 0 |
| `dataset_versions` | Provenance of loaded data | PK `dataset_version`; records seed and `is_synthetic` |

SQLite foreign-key enforcement is enabled per connection (`PRAGMA foreign_keys=ON`) because
SQLite ignores FK constraints by default — a silent-failure trap worth knowing.

## Artifact bundle (`artifacts/v1/`)

| File | Contents |
|---|---|
| `manifest.json` | Contract version, artifact version, origin, horizon, dataset fingerprint, library versions, SHA-256 per file |
| `model.joblib` | Fitted Pipeline (preprocessing + estimator) |
| `baseline.joblib` | Seasonal-naive baseline object (fair-comparison reference) |
| `feature_spec.json` | Ordered features, dtypes, roles, categorical vocabulary, lookback |
| `metrics.json` | All candidate metrics (validation + test), selection scores and rationale, residual quantiles |
| `training_config.json` | Seed, split dates, grids, selected params, row counts |
| `demand_history.parquet` | Panel demand up to the prediction origin (lag source at serving time) |
| `products.parquet` | Product master snapshot |
| `inventory_snapshot.parquet` | Stock snapshot |
| `batches.parquet` | Batch ledger snapshot |
| `importance.json` | Global permutation importance |
| `metrics_expiry.json` | Precision / recall / F1 / confusion matrix + critical band |
| `expiry_policy.json` | The exact FEFO rule, severity table and thresholds used |
