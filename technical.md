# Technical Writeup — Delivery-Time Prediction

Companion to `approach.md` (which is the plain-English version). This document is the
engineering detail: data characteristics, feature construction, model configs, validation
design, and the train/serve pipeline. Metric: **MAE** (mean absolute error, days).

---

## 1. Problem framing & data characteristics

- **Task:** regression of `delivery_time_days` using only order-time features.
- **Rows:** train = 85,159, test = 15,029. Each row is one product line; an order can span
  multiple rows but all rows of an order share the same target (delivery is order-level).
- **Target:** continuous, range 0.53–209.63, **median 11.0, mean 13.23, std 9.90**, strong
  right skew (99th pct ≈ 47, a tail out to 210). 301 rows > 60 days, 80 rows > 90.
- **Leakage checks:** no test `order_id` appears in train. Multi-line orders are ~4% (mean 1.04
  rows/order); not separately grouped in CV because the rate is low and test orders are disjoint.
- **Identifier overlap (test ∩ train):** `seller_id` ≈ **76%**, `customer_unique_id` ≈ **2.3%**.
  → seller-level encodings are valuable; customer-level ones are useless (encoded via city instead).
- **Temporal structure (the crux):**
  - train spans 2016-09-15 → 2018-06-21; **test spans 2017-01-19 → 2018-08-29**.
  - Test extends **~2 months beyond** the train horizon → partly an extrapolation problem.
  - Monthly mean delivery drifts from ~19 (late 2016) down to ~9.5 (mid 2018), with bumps at
    Black-Friday/Christmas months (2017-11 ≈ 15.1, 2017-12 ≈ 15.3, 2018-02 ≈ 16.9).
- **Missingness:** customer lat/lng (229 train / 44 test), seller lat/lng (205 / 17), product
  attributes (~1322), category (~1329), a few `order_approved_at` (14 test). Boosters consume NaN
  natively; coordinates are imputed (below).
- **Linear signal (Pearson vs target):** haversine distance **0.395**, freight_value **0.229**,
  weight 0.076, price 0.060. Geography dominates.

---

## 2. Feature engineering (67 features)

Implemented in `build_features.py` (offline cache) and `pipeline.py` (the importable
`prepare(train_df, query_df)` used by train/serve). All transforms are **fit on train, applied to
query**, so test/new-order features are consistent with training.

### 2.1 Coordinate imputation
Missing `{customer,seller}_{lat,lng}` filled with the **city-median centroid**, computed over the
`concat(train, query)` so a city seen only in test still resolves. Reduces missing distances from
229 → ~42 (cities with no coordinates anywhere).

### 2.2 Geometry
- `dist_km` = **haversine**(customer, seller); plus `log_dist`, `abs_dlat`, `abs_dlng`,
  `same_city` (string equality of city names).

### 2.3 Price / freight / product ratios
`total_price = price·qty + freight`, `freight_per_price`, `freight_per_kg`, `freight_per_km`
(freight is Olist's own distance×weight estimate → a strong engineered proxy), `price_per_item`,
`density = weight/volume`, `weight_x_dist`, `max_dim`, and `log1p` of weight/volume/freight.

### 2.4 Temporal (from `order_purchase_timestamp`)
`year, month, day, dayofweek, hour, dayofyear, weekofyear, quarter, is_weekend`;
`days_since_start` (continuous trend term — captures the multi-year speed-up; note a GBM **cannot
extrapolate** a monotone feature past its training range, so test dates beyond 2018-06 fall into
the last bin = "assume the most-recent regime", which is the desired behavior);
cyclical encodings `sin/cos(dayofyear)` and `sin/cos(day)`; `is_holiday_season`
(Nov-15→Jan-10), `days_to_xmas` (clipped −30..60), `days_to_holiday` (min distance to a hardcoded
Brazilian-holiday list 2016–2018).

### 2.5 Approval delay (legitimately known pre-shipment)
`order_approved_at − order_purchase_timestamp` in days (`approval_delay_days`), plus
`approved_hour`, `approved_dow`, `approval_missing`. Median delay ≈ 0.04 d, 95th ≈ 2 d.

### 2.6 Geographic clustering
`KMeans(n_clusters=30, random_state=42)` fit on **train** customer coords and seller coords →
`customer_region`, `seller_region`; `route_region = seller_region·1000 + customer_region`. These
cluster IDs are not fed raw to the model (don't generalize as integers) — they're **target-encoded**
(below) and then dropped.

### 2.7 Frequency encodings
`value_counts` over `concat(train,query)` for `seller_id, product_id, customer_city, seller_city,
category, customer_unique_id` → load/capacity proxies (a high-volume seller/route behaves
differently from a one-off).

### 2.8 Target (mean) encodings — out-of-fold, smoothed
The highest-value engineered clues. For a categorical key `k`:

```
encoded(k) = (mean_y(k)·n(k) + global_mean·α) / (n(k) + α)        # Bayesian smoothing
unseen k   → global_mean
```

- **Train rows** get **out-of-fold** values: 5-fold `KFold(shuffle, seed=42)`; each fold's encoding
  is computed only from the *other* folds → no row sees its own label. Prevents target leakage and
  the optimistic bias it causes.
- **Query rows** get **full-train** statistics.
- Keys & smoothing α: `seller_id`(30), `seller_city`(30), `customer_city`(20), `category`(30),
  `cust_region`(20), `seller_region`(20), `route_region`(15), `product_id`(30), and an interaction
  `seller_id × customer_region`(20) — a seller-specific destination "corridor" signal.

> **Reverted experiment (v2):** a finer corridor — target-encoding a 1°×1° grid `(seller_cell ×
> customer_cell)` pair plus `category × dest-cell` (→ 72 features). It dominated feature importance
> and improved local CV (3.878→3.862) but **worsened the public score (2.741→2.763)** — a classic
> CV/LB divergence from encoding rare, high-cardinality corridors. Dropped.

---

## 3. Models

MAE's optimal point predictor is the **conditional median**, so every model targets the median, not
the mean:

- **L1 objective** (`regression_l1` / `reg:absoluteerror`) directly minimizes absolute error.
- **log1p target + back-transform with expm1**: because `log` is monotonic, the median is
  preserved (`expm1(median(log y)) = median(y)`), so this still targets the median while stabilizing
  the heavy right tail and heteroscedasticity. Empirically the single best variant.

The winning ensemble = **4 models** (selected by the blend optimizer from 8 candidates):

| Model | Library | Objective | Target | Key params | Weight |
|---|---|---|---|---|---|
| lgb_l1_log | LightGBM | `regression_l1` | log1p | leaves 63, lr 0.03, mcs 60, ff/bf 0.8, λ1 1 / λ2 2 | **0.474** |
| lgb_l1_deep | LightGBM | `regression_l1` | raw | leaves 127, lr 0.02, mcs 100, ff 0.7 | **0.335** |
| lgb_l2_log | LightGBM | `regression` (L2) | log1p | leaves 63, lr 0.03 | **0.106** |
| xgb_mae | XGBoost | `reg:absoluteerror` | raw | hist, depth 8, eta 0.03, sub/col 0.8, λ 2 / α 1 | **0.085** |

Candidates that received ≈0 weight: LightGBM `quantile`(α0.5), `huber`(α5), `fair`(c3), and plain
`regression_l1` on raw. CatBoost-MAE was prototyped but excluded (slow, no marginal blend value).
All predictions are **clipped to [0.5, 210]**.

---

## 4. Validation design

Two complementary schemes (in `model_lib.run_model`), because the test set is forward-shifted:

1. **Chronological holdout (primary, forward-honest):** sort by `order_purchase_timestamp`, train
   on earliest 85%, evaluate on most-recent 15%. Mimics "learn past → predict future."
   → best single model **3.87**, blend **3.858**.
2. **5-fold random KFold OOF (for stacking + a full-coverage estimate):** every row gets a
   prediction from a model that didn't train on it. → OOF MAE **4.30** (higher than the chrono
   number because it includes the noisy 2016–17 rows that the chrono holdout excludes from *test*).

**Why local 3.87 but public 2.74:** the public test is dominated by recent, mature-logistics orders
that are intrinsically easier than the average training row, and easier than our deliberately-tough
"predict the most recent slice from the past" holdout. The local number is a conservative lower
bound on quality — which is why it was *beaten* on the LB rather than missed.

**Anti-leakage:** target encodings are strictly out-of-fold on train (§2.8); the chrono split never
lets future rows train the model that predicts them.

---

## 5. Blending

- Convex combination `ŷ = Σ wᵢ ŷᵢ`, `wᵢ ≥ 0`, `Σwᵢ = 1`. MAE is **convex in `w`** over the simplex
  (composition of a convex norm with a linear map), so it's a well-behaved optimization.
- Weights are optimized to **minimize MAE on the chronological-holdout predictions** (the
  forward-honest signal), via `scipy.optimize.minimize(method="Nelder-Mead")` from multiple
  restarts (uniform + each single-model vertex), keeping the best.
- Candidates compared: chrono-optimized, OOF-optimized, and uniform; the chrono-optimized set won
  and was applied to the **KFold-averaged test predictions** (each model's test prediction is the
  mean over its 5 fold-models → variance reduction).

---

## 6. Train / serve artifact

`train_and_save.py` → `model/`, consumed by `predict.py`. Both go through `pipeline.prepare()` so
features are identical at train and inference time.

- **Round selection:** for each model, run the chronological holdout with early stopping
  (`stopping_rounds=200`) to get `best_iteration`, then **refit on 100% of train** for
  `round(best_iteration × 1.1)` (the bump compensates for the ~15% more data).
- **Persistence:** LightGBM boosters → `.txt` (`save_model`), XGBoost → `.json`; `model/meta.json`
  stores per-model `{path, framework, log_target, weight, rounds}` plus the **exact feature-column
  order** and the clip range.
- **Inference (`predict.py`):** `prepare(train, new_orders)` rebuilds features (encoders refit on
  train deterministically — same `seed`, so reproducible), each booster predicts, `expm1` where
  `log_target`, clip, weighted blend, write `id,delivery_time_days`.
- **Fidelity:** this single-full-data 4-model blend reproduces the original 5-fold 8-candidate
  submission at **corr 0.9961, mean|Δ| 0.25 days** — a faithful, instantly-loadable copy.

---

## 7. Results & key takeaways

| Stage | Local CV (chrono) | Public LB |
|---|---|---|
| Single best model (lgb_l1_log) | 3.877 | — |
| **v1 blend (shipped, locked)** | 3.869 | **2.741** |
| v2 fine-corridor blend (reverted) | 3.858 | 2.763 |

- **Geography + timing dominate**; product attributes are weak. Freight value is a strong proxy
  for route difficulty.
- **Median-targeting (L1 / log1p)** is the right objective for MAE on a skewed target.
- **Out-of-fold target encoding** is what makes the high-cardinality geo/seller signals usable
  without leakage.
- **The LB is ground truth.** A change that improved local CV (fine corridor) hurt the LB →
  reverted. CV/LB divergence is the thing to watch, not the local number in isolation.

---

## 8. What I'd try next (not pursued — diminishing returns / over-fitting risk)

- Recency-weighted training (exponential half-life on sample weight) or dropping 2016 rows, to lean
  into the forward-shifted, recent-heavy test. Infra is in place (`run_model(half_life=…,
  min_date=…)`); validate on LB, not just local CV.
- Seeded bagging / multi-seed averaging of the same configs for a small variance reduction.
- A constrained linear/ridge **stack** instead of a pure convex blend (marginal, higher overfit
  risk on the misleading local signal).
- Reverse-geocoded Brazilian **state** pairs (origin→destination state corridors) — likely more
  robust than the 1° grid that overfit.
