"""
Train the winning 4-model blend on ALL training data and save a reusable model
to model/ (boosters + meta.json with blend weights). Run once:
    python train_and_save.py
Then use predict.py to score new orders without retraining.
"""
import os, json
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from sklearn.metrics import mean_absolute_error
from pipeline import prepare

os.makedirs("model", exist_ok=True)
CLIP = (0.5, 210.0)

LGB_BASE = dict(num_leaves=63, max_depth=-1, min_child_samples=60, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l1=1.0, lambda_l2=2.0,
                learning_rate=0.03, verbosity=-1, seed=42, num_threads=0)
def lgbp(extra):
    p = dict(LGB_BASE); p.update(extra); return p

# (name, framework, params, log_target, blend_weight)  -- the winning v1 blend
SPECS = [
    ("lgb_l1_log",  "lgb", lgbp({"objective":"regression_l1","metric":"mae"}), True,  0.474),
    ("lgb_l1_deep", "lgb", lgbp({"objective":"regression_l1","metric":"mae","num_leaves":127,
                                 "min_child_samples":100,"feature_fraction":0.7,"learning_rate":0.02}), False, 0.335),
    ("lgb_l2_log",  "lgb", lgbp({"objective":"regression","metric":"l2"}), True, 0.106),
    ("xgb_mae",     "xgb", dict(objective="reg:absoluteerror", eval_metric="mae", tree_method="hist",
                                eta=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                                min_child_weight=5, reg_lambda=2.0, reg_alpha=1.0, seed=42, nthread=0), False, 0.085),
]

def best_rounds(framework, params, log_t, X, y, t):
    """Find a good number of boosting rounds via a chronological holdout, then bump
    slightly because the final model trains on more (100%) data."""
    order = np.argsort(t); cut = int(len(y)*0.85)
    tr, va = order[:cut], order[cut:]
    yt = np.log1p(y[tr]) if log_t else y[tr]
    yv = np.log1p(y[va]) if log_t else y[va]
    if framework == "lgb":
        d = lgb.Dataset(X.iloc[tr], label=yt)
        dv = lgb.Dataset(X.iloc[va], label=yv)
        b = lgb.train(params, d, num_boost_round=6000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        return max(int(b.best_iteration*1.1), 100)
    else:
        d = xgb.DMatrix(X.iloc[tr], label=yt); dv = xgb.DMatrix(X.iloc[va], label=yv)
        b = xgb.train(params, d, num_boost_round=6000, evals=[(dv,"v")],
                      early_stopping_rounds=200, verbose_eval=False)
        return max(int(b.best_iteration*1.1), 100)

def fit_full(framework, params, log_t, rounds, X, y):
    yt = np.log1p(y) if log_t else y
    if framework == "lgb":
        return lgb.train(params, lgb.Dataset(X, label=yt), num_boost_round=rounds,
                         callbacks=[lgb.log_evaluation(0)])
    else:
        return xgb.train(params, xgb.DMatrix(X, label=yt), num_boost_round=rounds)

def main():
    print("Loading data + building features...")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    X, y, Xq, feat_cols, t = prepare(train, test)
    print(f"features: {len(feat_cols)}  train rows: {len(y)}")

    meta = {"feat_cols": feat_cols, "clip": CLIP, "models": []}
    test_blend = np.zeros(len(Xq)); wsum = 0.0
    for name, fw, params, log_t, w in SPECS:
        print(f"\n-> {name} (weight {w})")
        r = best_rounds(fw, params, log_t, X, y, t)
        print(f"   training on FULL data for {r} rounds...")
        booster = fit_full(fw, params, log_t, r, X, y)
        if fw == "lgb":
            path = f"model/{name}.txt"; booster.save_model(path)
            pq = booster.predict(Xq)
        else:
            path = f"model/{name}.json"; booster.save_model(path)
            pq = booster.predict(xgb.DMatrix(Xq))
        if log_t: pq = np.expm1(pq)
        pq = np.clip(pq, *CLIP)
        test_blend += w * pq; wsum += w
        meta["models"].append({"name": name, "framework": fw, "path": path,
                               "log_target": log_t, "weight": w, "rounds": r})
        print(f"   saved -> {path}")

    test_blend /= wsum
    json.dump(meta, open("model/meta.json", "w"), indent=2)
    print("\nsaved model/meta.json")

    # sanity check: compare this saved model's test prediction to the locked submission
    try:
        sub = pd.read_csv("submission.csv")["delivery_time_days"].values
        corr = np.corrcoef(test_blend, sub)[0,1]
        print(f"\nSanity vs locked submission.csv: corr={corr:.4f}  "
              f"mean {test_blend.mean():.3f} vs {sub.mean():.3f}  "
              f"mean|diff|={np.abs(test_blend-sub).mean():.3f} days")
    except Exception as e:
        print("(no submission.csv to compare)", e)
    print("\nDONE. Use:  python predict.py  [input.csv]  [output.csv]")

if __name__ == "__main__":
    main()
