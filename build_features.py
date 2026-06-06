"""
Feature engineering for AMGE delivery-time prediction (Olist).
Produces cached parquet: feat_train.parquet, feat_test.parquet, plus meta.json.
All target-encodings are out-of-fold (KFold) on train and full-train stats on test.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

SEED = 42
TARGET = "delivery_time_days"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def load():
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    return tr, te

def impute_city_coords(tr, te):
    """Fill missing lat/lng from city-level centroid (computed over train+test)."""
    both = pd.concat([tr, te], ignore_index=True, sort=False)
    for side in ["customer", "seller"]:
        city = f"{side}_city"
        for ax in ["lat", "lng"]:
            col = f"{side}_{ax}"
            cent = both.groupby(city)[col].transform("median")
            both[col] = both[col].fillna(cent)
    # split back
    ntr = len(tr)
    return both.iloc[:ntr].copy(), both.iloc[ntr:].reset_index(drop=True).copy()

# ---------------- base feature builder (no target leakage) ----------------
BR_HOLIDAYS = pd.to_datetime([
    # New Year, Carnival, Tiradentes, Labour, Corpus, Independence, N.S.Aparecida,
    # Finados, Proclamacao, Christmas across 2016-2018
    "2016-12-25","2017-01-01","2017-02-27","2017-02-28","2017-04-14","2017-04-21",
    "2017-05-01","2017-06-15","2017-09-07","2017-10-12","2017-11-02","2017-11-15",
    "2017-11-24","2017-12-25","2018-01-01","2018-02-12","2018-02-13","2018-03-30",
    "2018-04-21","2018-05-01","2018-05-31","2018-09-07","2018-10-12","2018-11-02",
    "2018-11-15","2018-11-23","2018-12-25",
])

def base_features(df):
    f = pd.DataFrame(index=df.index)
    # ---- numeric passthrough ----
    num = ["quantity","price","freight_value","product_weight_g","product_length_cm",
           "product_height_cm","product_width_cm","volume_cm3","product_photos_qty",
           "product_name_length","product_description_length",
           "customer_lat","customer_lng","seller_lat","seller_lng"]
    for c in num:
        f[c] = df[c].astype("float32")

    # ---- distance ----
    f["dist_km"] = haversine(df.customer_lat, df.customer_lng,
                             df.seller_lat, df.seller_lng).astype("float32")
    f["abs_dlat"] = (df.customer_lat - df.seller_lat).abs().astype("float32")
    f["abs_dlng"] = (df.customer_lng - df.seller_lng).abs().astype("float32")
    f["same_city"] = (df.customer_city.astype(str) == df.seller_city.astype(str)).astype("int8")

    # ---- pricing / freight ratios ----
    f["total_price"] = (df.price * df.quantity + df.freight_value).astype("float32")
    f["freight_per_price"] = (df.freight_value / (df.price + 1e-3)).astype("float32")
    f["freight_per_kg"] = (df.freight_value / (df.product_weight_g / 1000 + 1e-3)).astype("float32")
    f["freight_per_km"] = (df.freight_value / (f["dist_km"] + 1.0)).astype("float32")
    f["price_per_item"] = (df.price / (df.quantity + 1e-3)).astype("float32")
    f["density"] = (df.product_weight_g / (df.volume_cm3 + 1e-3)).astype("float32")
    f["weight_x_dist"] = (df.product_weight_g * f["dist_km"]).astype("float32")
    f["log_weight"] = np.log1p(df.product_weight_g).astype("float32")
    f["log_volume"] = np.log1p(df.volume_cm3).astype("float32")
    f["log_freight"] = np.log1p(df.freight_value).astype("float32")
    f["log_dist"] = np.log1p(f["dist_km"]).astype("float32")
    f["max_dim"] = df[["product_length_cm","product_height_cm","product_width_cm"]].max(axis=1).astype("float32")

    # ---- temporal: purchase ----
    ts = pd.to_datetime(df["order_purchase_timestamp"])
    f["pur_year"] = ts.dt.year.astype("int16")
    f["pur_month"] = ts.dt.month.astype("int8")
    f["pur_day"] = ts.dt.day.astype("int8")
    f["pur_dow"] = ts.dt.dayofweek.astype("int8")
    f["pur_hour"] = ts.dt.hour.astype("int8")
    f["pur_doy"] = ts.dt.dayofyear.astype("int16")
    f["pur_woy"] = ts.dt.isocalendar().week.astype("int16")
    f["pur_quarter"] = ts.dt.quarter.astype("int8")
    f["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    ref = pd.Timestamp("2016-09-01")
    f["days_since_start"] = ((ts - ref).dt.total_seconds() / 86400).astype("float32")
    f["doy_sin"] = np.sin(2*np.pi*ts.dt.dayofyear/365.25).astype("float32")
    f["doy_cos"] = np.cos(2*np.pi*ts.dt.dayofyear/365.25).astype("float32")
    f["dom_sin"] = np.sin(2*np.pi*ts.dt.day/31).astype("float32")
    f["dom_cos"] = np.cos(2*np.pi*ts.dt.day/31).astype("float32")
    # holiday-season effects (high volume late Nov - early Jan)
    f["is_holiday_season"] = (((ts.dt.month == 11) & (ts.dt.day >= 15)) |
                              (ts.dt.month == 12) |
                              ((ts.dt.month == 1) & (ts.dt.day <= 10))).astype("int8")
    days_to_xmas = ((pd.to_datetime(ts.dt.year.astype(str) + "-12-25") - ts)
                    .dt.total_seconds() / 86400)
    f["days_to_xmas"] = days_to_xmas.clip(-30, 60).astype("float32")
    # nearest holiday distance
    hol = BR_HOLIDAYS.values.astype("datetime64[ns]").astype("int64")
    tsv = ts.values.astype("datetime64[ns]").astype("int64")
    nearest = np.min(np.abs(tsv[:, None] - hol[None, :]), axis=1) / 86400e9
    f["days_to_holiday"] = nearest.astype("float32")

    # ---- temporal: approval delay (available pre-shipment) ----
    ap = pd.to_datetime(df["order_approved_at"])
    delay = (ap - ts).dt.total_seconds() / 86400
    f["approval_delay_days"] = delay.astype("float32")
    f["approved_hour"] = ap.dt.hour.astype("float32")
    f["approved_dow"] = ap.dt.dayofweek.astype("float32")
    f["approval_missing"] = ap.isna().astype("int8")

    return f

def add_geo_clusters(ftr, fte, n=30):
    """KMeans regions on customer and seller coords (fit on train, apply both)."""
    out = {}
    for side in ["customer", "seller"]:
        cols = [f"{side}_lat", f"{side}_lng"]
        tr_xy = ftr[cols].fillna(ftr[cols].median())
        te_xy = fte[cols].fillna(ftr[cols].median())
        km = KMeans(n_clusters=n, random_state=SEED, n_init=4)
        ftr[f"{side}_region"] = km.fit_predict(tr_xy).astype("int16")
        fte[f"{side}_region"] = km.predict(te_xy).astype("int16")
    ftr["route_region"] = (ftr["seller_region"].astype(int) * 1000 + ftr["customer_region"].astype(int))
    fte["route_region"] = (fte["seller_region"].astype(int) * 1000 + fte["customer_region"].astype(int))
    return ftr, fte

# ---------------- out-of-fold target / count encodings ----------------
def oof_target_encode(tr_key, te_key, y, n_splits=5, smoothing=20.0, seed=SEED):
    """Return oof-encoded train array + full-train-stats test array."""
    tr_key = tr_key.astype(str).values
    te_key = te_key.astype(str).values
    global_mean = y.mean()
    oof = np.full(len(tr_key), global_mean, dtype="float64")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yv = y.values
    for tr_idx, va_idx in kf.split(tr_key):
        d = pd.DataFrame({"k": tr_key[tr_idx], "y": yv[tr_idx]})
        agg = d.groupby("k")["y"].agg(["mean", "count"])
        sm = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        m = pd.Series(te_key[0:0])  # placeholder
        oof[va_idx] = pd.Series(tr_key[va_idx]).map(sm).fillna(global_mean).values
    # full-train stats for test
    d = pd.DataFrame({"k": tr_key, "y": yv})
    agg = d.groupby("k")["y"].agg(["mean", "count"])
    sm = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
    te_enc = pd.Series(te_key).map(sm).fillna(global_mean).values
    return oof.astype("float32"), te_enc.astype("float32")

def freq_encode(tr_s, te_s):
    both = pd.concat([tr_s.astype(str), te_s.astype(str)])
    vc = both.value_counts()
    return (tr_s.astype(str).map(vc).astype("float32").values,
            te_s.astype(str).map(vc).astype("float32").values)

def main():
    tr_raw, te_raw = load()
    y = tr_raw[TARGET].astype("float32")
    tr_raw, te_raw = impute_city_coords(tr_raw, te_raw)

    ftr = base_features(tr_raw)
    fte = base_features(te_raw)
    ftr, fte = add_geo_clusters(ftr, fte, n=30)

    # frequency encodings (capacity/load proxies)
    for col in ["seller_id", "product_id", "customer_city", "seller_city",
                "product_category_name_english", "customer_unique_id"]:
        ftr[f"{col}_freq"], fte[f"{col}_freq"] = freq_encode(tr_raw[col], te_raw[col])

    # out-of-fold target encodings
    te_specs = [
        ("seller_id", tr_raw["seller_id"], te_raw["seller_id"], 30.0),
        ("seller_city", tr_raw["seller_city"], te_raw["seller_city"], 30.0),
        ("customer_city", tr_raw["customer_city"], te_raw["customer_city"], 20.0),
        ("category", tr_raw["product_category_name_english"], te_raw["product_category_name_english"], 30.0),
        ("cust_region", ftr["customer_region"], fte["customer_region"], 20.0),
        ("seller_region", ftr["seller_region"], fte["seller_region"], 20.0),
        ("route_region", ftr["route_region"], fte["route_region"], 15.0),
        ("product_id", tr_raw["product_id"], te_raw["product_id"], 30.0),
    ]
    for name, ks, kt, sm in te_specs:
        o, t = oof_target_encode(ks, kt, y, smoothing=sm)
        ftr[f"te_{name}"] = o
        fte[f"te_{name}"] = t

    # also encode interaction: seller_id x customer_region (logistics corridor)
    ftr_corridor = tr_raw["seller_id"].astype(str) + "_" + ftr["customer_region"].astype(str)
    fte_corridor = te_raw["seller_id"].astype(str) + "_" + fte["customer_region"].astype(str)
    o, t = oof_target_encode(ftr_corridor, fte_corridor, y, smoothing=20.0)
    ftr["te_seller_custregion"] = o
    fte["te_seller_custregion"] = t

    # drop region id cols (keep encoded versions; ids themselves not generalizable)
    drop_ids = ["customer_region", "seller_region", "route_region"]
    ftr = ftr.drop(columns=drop_ids)
    fte = fte.drop(columns=drop_ids)

    # time index for chronological CV
    ts_tr = pd.to_datetime(tr_raw["order_purchase_timestamp"])
    ftr["__time"] = ts_tr.values.astype("int64")
    ftr["__order_id"] = tr_raw["order_id"].values
    ftr[TARGET] = y.values

    assert list(ftr.drop(columns=["__time","__order_id",TARGET]).columns) == list(fte.columns), \
        "train/test feature columns mismatch"

    ftr.to_parquet("feat_train.parquet")
    fte.to_parquet("feat_test.parquet")
    feat_cols = [c for c in fte.columns]
    json.dump({"features": feat_cols, "target": TARGET},
              open("meta.json", "w"), indent=2)
    print("train feat shape", ftr.shape, "test feat shape", fte.shape)
    print("n features:", len(feat_cols))
    print("sample features:", feat_cols[:20])
    print("NaN counts (train) top:")
    print(ftr[feat_cols].isna().sum().sort_values(ascending=False).head(10).to_dict())

if __name__ == "__main__":
    main()
