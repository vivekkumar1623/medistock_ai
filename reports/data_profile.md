# Data Profile

* Dataset version: `ds-v1`
* Synthetic: **True**
* Seed: `42`
* Date range: 2024-05-01 to 2025-12-31
* Snapshot date: 2025-10-01

## Tables

| Table | Rows | Columns | Duplicate rows |
|---|---:|---:|---:|
| products | 120 | 9 | 0 |
| sales | 67558 | 11 | 0 |
| inventory | 120 | 5 | 0 |
| batches | 298 | 7 | 0 |
| demand_panel | 73200 | 3 | 0 |

## Demand characteristics

* Zero-demand share across the whole panel: **7.71%**
* Products with more than half of days at zero: **9**
* Mean daily units per product-day: **9.01**

## Category summary

| Category | Total units | Mean/day | Std |
|---|---:|---:|---:|
| Analgesics | 152,098 | 16.62 | 8.07 |
| Antibiotics | 63,187 | 6.91 | 5.83 |
| Antihistamines | 56,710 | 6.20 | 5.41 |
| Cardiovascular | 89,324 | 9.76 | 7.08 |
| Dermatology | 40,136 | 4.39 | 3.71 |
| Diabetes Care | 66,377 | 7.25 | 4.46 |
| Gastrointestinal | 80,777 | 8.83 | 5.75 |
| Vitamins & Supplements | 110,954 | 12.13 | 7.59 |

## Column-level outliers (IQR rule)

| Table | Column | Outliers | Min | Max |
|---|---|---:|---:|---:|
| products | unit_price | 1 | 32.43 | 672.03 |
| products | unit_cost | 3 | 19.4 | 502.22 |
| products | minimum_stock_level | 2 | 14.0 | 262.0 |
| sales | quantity_sold | 2071 | 1.0 | 112.0 |
| sales | selling_price | 1055 | 23.78 | 731.82 |
| sales | discount | 4715 | 0.0 | 0.2499 |
| sales | is_festival_window | 13547 | 0.0 | 1.0 |
| inventory | current_stock | 6 | 24.0 | 2462.0 |
| inventory | reorder_level | 4 | 14.0 | 361.0 |
| batches | batch_quantity | 28 | 2.0 | 3406.0 |
| batches | remaining_quantity | 26 | 1.0 | 2221.0 |
| demand_panel | quantity | 1743 | 0.0 | 112.0 |
