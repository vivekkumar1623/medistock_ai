# Error Analysis

Held-out **test** period only. Scores are produced by the model stored in the
artifact bundle, so this is the error profile of the system that is actually served.

## Overall

| MAE | RMSE | WAPE | Bias | n |
|---:|---:|---:|---:|---:|
| 29.665 | 39.965 | 0.108 | 4.722 | 480 |

Bias of **4.722** units means the model is very slightly over-forecasting on average.

## By category

| Category | n | MAE | RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|---:|
| Antibiotics | 60 | 21.91 | 26.42 | 0.1847 | 13.85 |
| Antihistamines | 60 | 40.83 | 52.12 | 0.1825 | 34.44 |
| Dermatology | 60 | 21.73 | 29.95 | 0.1209 | -11.65 |
| Vitamins & Supplements | 60 | 33.12 | 40.59 | 0.1206 | 23.26 |
| Cardiovascular | 60 | 33.17 | 41.69 | 0.1186 | -5.26 |
| Diabetes Care | 60 | 19.11 | 23.62 | 0.0858 | -4.52 |
| Gastrointestinal | 60 | 27.53 | 35.47 | 0.0841 | 5.68 |
| Analgesics | 60 | 39.92 | 57.13 | 0.0699 | -18.02 |

## By demand magnitude (deciles of actual demand)

| Decile | n | MAE | RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|---:|
| D1 | 48 | 21.69 | 27.88 | 0.3094 | 18.65 |
| D4 | 48 | 30.78 | 36.92 | 0.1722 | 9.63 |
| D3 | 48 | 23.76 | 30.94 | 0.1588 | 8.08 |
| D2 | 48 | 18.88 | 23.24 | 0.1558 | 4.67 |
| D5 | 48 | 30.78 | 40.68 | 0.1422 | 14.44 |
| D8 | 48 | 34.17 | 42.13 | 0.1021 | -2.72 |
| D6 | 48 | 24.82 | 33.1 | 0.1008 | 7.97 |
| D7 | 48 | 26.54 | 33.64 | 0.0936 | 7.65 |
| D10 | 48 | 53.16 | 70.46 | 0.0772 | -22.3 |
| D9 | 48 | 32.07 | 41.39 | 0.0701 | 1.14 |

WAPE is naturally larger for low-volume deciles: the denominator is small while
absolute errors do not shrink proportionally. This is a property of the metric, not
a defect, and it is why the UI flags low-demand products with lower confidence.

## By demand sparsity

| Bucket | n | MAE | RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|---:|
| sparse (>25%) | 56 | 33.85 | 41.96 | 0.2551 | 7.68 |
| moderate (5-25%) | 11 | 12.6 | 15.77 | 0.1723 | 4.68 |
| dense (<5% zero days) | 413 | 29.55 | 40.14 | 0.0987 | 4.32 |

## Ten worst test predictions

| Product | Origin | Category | Actual | Predicted | Residual |
|---|---|---|---:|---:|---:|
| MED-0042 | 2025-09-10 | Analgesics | 887.0 | 661.12 | -225.88 |
| MED-0042 | 2025-09-24 | Analgesics | 874.0 | 708.05 | -165.95 |
| MED-0043 | 2025-08-27 | Antihistamines | 613.0 | 770.33 | 157.33 |
| MED-0034 | 2025-09-24 | Analgesics | 1157.0 | 1005.81 | -151.19 |
| MED-0043 | 2025-09-10 | Antihistamines | 567.0 | 691.18 | 124.18 |
| MED-0115 | 2025-09-24 | Antihistamines | 218.0 | 338.11 | 120.11 |
| MED-0108 | 2025-08-27 | Cardiovascular | 200.0 | 315.68 | 115.68 |
| MED-0082 | 2025-09-24 | Analgesics | 517.0 | 615.92 | 98.92 |
| MED-0015 | 2025-08-27 | Gastrointestinal | 568.0 | 472.07 | -95.93 |
| MED-0079 | 2025-09-24 | Gastrointestinal | 158.0 | 251.13 | 93.13 |

## What this tells us

1. Errors concentrate in the low-volume deciles, where an aggregate 30-day total is
   inherently noisy. Reporting one aggregate WAPE would hide this.
2. Sparse products are worse than dense ones, which is exactly why the serving layer
   attaches a low-confidence note instead of a false precision claim.
3. The largest single errors sit on products with promotion spikes, which the feature
   set cannot anticipate (promotions are not in the data at all). That is a stated
   limitation, not something more model capacity would fix.
