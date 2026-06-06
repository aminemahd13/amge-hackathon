"""
Reusable feature pipeline for delivery-time prediction.
`prepare(train_df, query_df)` rebuilds the exact same features used for the winning
submission: out-of-fold target encodings on the TRAIN rows (so the model can't cheat
while learning) and full-train statistics on the QUERY rows (test / new orders).
Both training (train_and_save.py) and scoring (predict.py) call this, so they stay
perfectly consistent. Reuses the helper functions in build_features.py.
"""
import numpy as np, pandas as pd
from build_features import (impute_city_coords, base_features, add_geo_clusters,
                            oof_target_encode, freq_encode, TARGET)

FREQ_COLS = ["seller_id", "product_id", "customer_city", "seller_city",
             "product_category_name_english", "customer_unique_id"]

def prepare(train_df, query_df):
    """Returns (X_train_oof, y, X_query, feat_cols).
    train_df must contain the target column; query_df need not."""
    train_df = train_df.copy()
    query_df = query_df.copy()
    y = train_df[TARGET].astype("float32")

    # fit city centroids on train+query, fill missing coords
    train_df, query_df = impute_city_coords(train_df, query_df)

    ftr = base_features(train_df)
    fte = base_features(query_df)
    ftr, fte = add_geo_clusters(ftr, fte, n=30)        # KMeans fit on train (deterministic)

    for col in FREQ_COLS:
        ftr[f"{col}_freq"], fte[f"{col}_freq"] = freq_encode(train_df[col], query_df[col])

    te_specs = [
        ("seller_id", train_df["seller_id"], query_df["seller_id"], 30.0),
        ("seller_city", train_df["seller_city"], query_df["seller_city"], 30.0),
        ("customer_city", train_df["customer_city"], query_df["customer_city"], 20.0),
        ("category", train_df["product_category_name_english"], query_df["product_category_name_english"], 30.0),
        ("cust_region", ftr["customer_region"], fte["customer_region"], 20.0),
        ("seller_region", ftr["seller_region"], fte["seller_region"], 20.0),
        ("route_region", ftr["route_region"], fte["route_region"], 15.0),
        ("product_id", train_df["product_id"], query_df["product_id"], 30.0),
    ]
    for name, ks, kt, sm in te_specs:
        o, t = oof_target_encode(ks, kt, y, smoothing=sm)
        ftr[f"te_{name}"] = o
        fte[f"te_{name}"] = t

    corr_s = train_df["seller_id"].astype(str) + "_" + ftr["customer_region"].astype(str)
    corr_t = query_df["seller_id"].astype(str) + "_" + fte["customer_region"].astype(str)
    o, t = oof_target_encode(corr_s, corr_t, y, smoothing=20.0)
    ftr["te_seller_custregion"] = o
    fte["te_seller_custregion"] = t

    drop_ids = ["customer_region", "seller_region", "route_region"]
    ftr = ftr.drop(columns=drop_ids)
    fte = fte.drop(columns=drop_ids)

    feat_cols = list(fte.columns)
    # time index for chronological round-selection during training
    t_train = pd.to_datetime(train_df["order_purchase_timestamp"]).values.astype("int64")
    return ftr[feat_cols], y.values.astype("float64"), fte[feat_cols], feat_cols, t_train
