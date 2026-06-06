"""
Shared modeling library: loads cached features, trains a model spec with
(1) random 5-fold OOF preds + OOF MAE, (2) chronological-holdout MAE,
(3) full test preds (avg over folds). Saves oof_<name>.npy and test_<name>.npy.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

SEED = 42
TARGET = "delivery_time_days"

def load_data():
    suf = os.environ.get("FEAT_SUFFIX", "")   # "" -> v1, "_v2" -> v2
    ftr = pd.read_parquet(f"feat_train{suf}.parquet")
    fte = pd.read_parquet(f"feat_test{suf}.parquet")
    meta = json.load(open(f"meta{suf}.json"))
    feats = meta["features"]
    X = ftr[feats].copy()
    y = ftr[TARGET].astype("float64").values
    t = ftr["__time"].values
    Xte = fte[feats].copy()
    return X, y, t, Xte, feats

# ---------- model builders: each returns (fit_predict_fn) ----------
def lgbm_fit(params, log_target=False):
    import lightgbm as lgb
    def f(Xtr, ytr, Xva, yva, Xte, wtr=None):
        ytr_ = np.log1p(ytr) if log_target else ytr
        yva_ = np.log1p(yva) if (log_target and yva is not None) else yva
        dtr = lgb.Dataset(Xtr, label=ytr_, weight=wtr)
        valid = [lgb.Dataset(Xva, label=yva_, reference=dtr)] if Xva is not None else None
        callbacks = [lgb.log_evaluation(0)]
        if Xva is not None:
            callbacks.append(lgb.early_stopping(params.get("early", 150), verbose=False))
        p = {k: v for k, v in params.items() if k not in ("early", "n_estimators")}
        booster = lgb.train(p, dtr, num_boost_round=params.get("n_estimators", 4000),
                            valid_sets=valid, callbacks=callbacks)
        bi = booster.best_iteration or params.get("n_estimators", 4000)
        pva = booster.predict(Xva, num_iteration=bi) if Xva is not None else None
        pte = booster.predict(Xte, num_iteration=bi)
        if log_target:
            if pva is not None: pva = np.expm1(pva)
            pte = np.expm1(pte)
        return pva, pte, bi
    return f

def xgb_fit(params, log_target=False):
    import xgboost as xgb
    def f(Xtr, ytr, Xva, yva, Xte, wtr=None):
        ytr_ = np.log1p(ytr) if log_target else ytr
        yva_ = np.log1p(yva) if (log_target and yva is not None) else yva
        dtr = xgb.DMatrix(Xtr, label=ytr_, weight=wtr)
        dte = xgb.DMatrix(Xte)
        evals = []
        dva = None
        if Xva is not None:
            dva = xgb.DMatrix(Xva, label=yva_)
            evals = [(dva, "val")]
        p = {k: v for k, v in params.items() if k not in ("early", "n_estimators")}
        booster = xgb.train(p, dtr, num_boost_round=params.get("n_estimators", 4000),
                            evals=evals, early_stopping_rounds=params.get("early", 150),
                            verbose_eval=False)
        bi = getattr(booster, "best_iteration", None)
        rng = (0, (bi + 1) if bi is not None else params.get("n_estimators", 4000))
        pva = booster.predict(dva, iteration_range=rng) if dva is not None else None
        pte = booster.predict(dte, iteration_range=rng)
        if log_target:
            if pva is not None: pva = np.expm1(pva)
            pte = np.expm1(pte)
        return pva, pte, (bi or 0)
    return f

def cat_fit(params, log_target=False):
    from catboost import CatBoostRegressor, Pool
    def f(Xtr, ytr, Xva, yva, Xte, wtr=None):
        ytr_ = np.log1p(ytr) if log_target else ytr
        yva_ = np.log1p(yva) if (log_target and yva is not None) else yva
        m = CatBoostRegressor(**{k: v for k, v in params.items() if k != "early"},
                              early_stopping_rounds=params.get("early", 150))
        m.fit(Xtr, ytr_, sample_weight=wtr,
              eval_set=(Xva, yva_) if Xva is not None else None, verbose=False)
        pva = m.predict(Xva) if Xva is not None else None
        pte = m.predict(Xte)
        if log_target:
            if pva is not None: pva = np.expm1(pva)
            pte = np.expm1(pte)
        return pva, pte, m.get_best_iteration()
    return f

# ---------- recency weighting / old-data masking ----------
def _recency_weight(t, half_life_days):
    if not half_life_days:
        return None
    days = (t - t.min()) / 86400e9
    return np.power(0.5, (days.max() - days) / half_life_days).astype("float64")

def _keep_mask(t, min_date):
    if min_date is None:
        return np.ones(len(t), dtype=bool)
    cut = np.datetime64(min_date).astype("datetime64[ns]").astype("int64")
    return t >= cut

# ---------- evaluation harness ----------
def run_model(name, fit_fn, n_splits=5, chrono_frac=0.15, clip=(0.5, 210.0), seed=SEED,
              half_life=None, min_date=None):
    X, y, t, Xte, feats = load_data()
    n = len(y)
    W = _recency_weight(t, half_life)          # full-length weights or None
    keep = _keep_mask(t, min_date)             # full-length training-eligibility mask

    def sub(idx):
        idx = idx[keep[idx]]                    # drop old (ineligible) rows from TRAINING only
        return idx

    # ---- chronological holdout (honest forward MAE) ----
    order = np.argsort(t)
    cut = int(n * (1 - chrono_frac))
    tr_idx_c, va_idx_c = sub(order[:cut]), order[cut:]
    wtr_c = None if W is None else W[tr_idx_c]
    pva_c, _, bi_c = fit_fn(X.iloc[tr_idx_c], y[tr_idx_c], X.iloc[va_idx_c], y[va_idx_c], Xte, wtr_c)
    pva_c = np.clip(pva_c, *clip)
    chrono_mae = mean_absolute_error(y[va_idx_c], pva_c)
    np.save(f"chrono_{name}.npy", pva_c)
    np.save("chrono_idx.npy", va_idx_c)
    np.save("chrono_y.npy", y[va_idx_c])

    # ---- random KFold OOF + test preds ----
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(n)
    test_pred = np.zeros(len(Xte))
    for k, (tr_idx, va_idx) in enumerate(kf.split(X)):
        tr_idx = sub(tr_idx)
        wtr = None if W is None else W[tr_idx]
        pva, pte, bi = fit_fn(X.iloc[tr_idx], y[tr_idx], X.iloc[va_idx], y[va_idx], Xte, wtr)
        oof[va_idx] = pva
        test_pred += pte / n_splits
    oof = np.clip(oof, *clip)
    test_pred = np.clip(test_pred, *clip)
    oof_mae = mean_absolute_error(y, oof)

    np.save(f"oof_{name}.npy", oof)
    np.save(f"test_{name}.npy", test_pred)
    print(f"[{name}] chrono_holdout_MAE={chrono_mae:.4f} (best_iter={bi_c})  KFold_OOF_MAE={oof_mae:.4f}")
    return {"name": name, "chrono_mae": chrono_mae, "oof_mae": oof_mae}
