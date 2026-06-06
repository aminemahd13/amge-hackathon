"""
Score new orders with the saved model (no retraining).

    python predict.py                      # scores data/test.csv  -> predictions.csv
    python predict.py my_orders.csv         # scores my_orders.csv  -> predictions.csv
    python predict.py my_orders.csv out.csv # custom output path

The input CSV must have the same columns as data/test.csv (everything known at order
time: ids, customer/seller city+coords, product info, price, freight, timestamps).
Needs data/train.csv present (used to rebuild the historical-average features).
"""
import sys, json
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from pipeline import prepare

def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"

    meta = json.load(open("model/meta.json"))
    train = pd.read_csv("data/train.csv")
    query = pd.read_csv(in_path)
    print(f"Scoring {len(query)} orders from {in_path} ...")

    # rebuild features for the query rows (encoders fit on train, applied to query)
    _, _, Xq, feat_cols, _ = prepare(train, query)
    Xq = Xq[meta["feat_cols"]]   # exact column order the models were trained on

    clip = tuple(meta["clip"])
    blend = np.zeros(len(Xq)); wsum = 0.0
    for m in meta["models"]:
        if m["framework"] == "lgb":
            b = lgb.Booster(model_file=m["path"]); p = b.predict(Xq)
        else:
            b = xgb.Booster(); b.load_model(m["path"]); p = b.predict(xgb.DMatrix(Xq))
        if m["log_target"]: p = np.expm1(p)
        p = np.clip(p, *clip)
        blend += m["weight"] * p; wsum += m["weight"]
    blend = np.clip(blend / wsum, *clip)

    out = pd.DataFrame({"id": np.arange(len(blend)),
                        "delivery_time_days": np.round(blend, 2)})
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}  rows={len(out)}  "
          f"predicted days: median {np.median(blend):.2f}  mean {blend.mean():.2f}")

if __name__ == "__main__":
    main()
