"""
Blend model predictions with non-negative convex weights.
Weights are optimized on the CHRONOLOGICAL holdout (forward-validated) predictions,
which is the honest proxy for the recent/forward-shifted test set.
Reports OOF and simple-average for comparison; picks the most robust by chrono MAE.
Usage: python blend.py m1 m2 ...   (names match oof_<name>.npy & chrono_<name>.npy)
"""
import sys, json
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

TARGET = "delivery_time_days"

def opt_weights(P, ytrue, n_restarts=True):
    m = P.shape[1]
    def f(w):
        w = np.clip(w, 0, None); s = w.sum()
        if s <= 0: return 1e9
        return mean_absolute_error(ytrue, P @ (w/s))
    best = None
    # full restarts (each single-model corner) only when cheap (chrono); else uniform start
    starts = [np.ones(m)/m] + ([np.eye(m)[i] for i in range(m)] if n_restarts else [])
    for w0 in starts:
        r = minimize(f, w0, method="Nelder-Mead",
                     options={"maxiter": 6000, "xatol": 1e-6, "fatol": 1e-7})
        if best is None or r.fun < best.fun:
            best = r
    w = np.clip(best.x, 0, None); return w / w.sum()

def main(names):
    ftr = pd.read_parquet("feat_train.parquet")
    y = ftr[TARGET].astype("float64").values

    chrono_y = np.load("chrono_y.npy")
    oofs, tests, chronos, kept = [], [], [], []
    for nm in names:
        try:
            o = np.load(f"oof_{nm}.npy"); c = np.load(f"chrono_{nm}.npy"); t = np.load(f"test_{nm}.npy")
        except FileNotFoundError:
            print(f"  (skip missing {nm})"); continue
        oofs.append(o); tests.append(t); chronos.append(c); kept.append(nm)

    OOF = np.vstack(oofs).T; TEST = np.vstack(tests).T; CHR = np.vstack(chronos).T
    m = len(kept)
    print(f"Blending {m} models")
    print(f"{'model':18s} {'OOF_MAE':>9s} {'chrono_MAE':>11s}")
    for i, nm in enumerate(kept):
        print(f"{nm:18s} {mean_absolute_error(y,OOF[:,i]):9.4f} {mean_absolute_error(chrono_y,CHR[:,i]):11.4f}")

    # --- candidate weightings ---
    w_chrono = opt_weights(CHR, chrono_y, n_restarts=True)   # forward holdout (cheap, full restarts)
    w_oof = opt_weights(OOF, y, n_restarts=False)            # random OOF (big, single start)
    w_uniform = np.ones(m)/m

    cands = {
        "chrono_opt": w_chrono,
        "oof_opt":    w_oof,
        "uniform":    w_uniform,
    }
    print("\n--- candidate blends (evaluated on chrono holdout) ---")
    scored = []
    for nm, w in cands.items():
        c_mae = mean_absolute_error(chrono_y, CHR @ w)
        o_mae = mean_absolute_error(y, OOF @ w)
        scored.append((nm, w, c_mae, o_mae))
        print(f"{nm:12s} chrono={c_mae:.4f}  oof={o_mae:.4f}")
    best_single_chr = min(mean_absolute_error(chrono_y, CHR[:,i]) for i in range(m))
    print(f"{'best_single':12s} chrono={best_single_chr:.4f}")

    # choose blend with best chrono MAE (the forward-honest objective)
    scored.sort(key=lambda z: z[2])
    chosen_name, w, c_mae, o_mae = scored[0]
    print(f"\n>>> CHOSEN: {chosen_name}  chrono_MAE={c_mae:.4f}  oof_MAE={o_mae:.4f}")
    print("weights:")
    for nm, wi in sorted(zip(kept, w), key=lambda z: -z[1]):
        if wi > 1e-4: print(f"  {nm:18s} {wi:.3f}")

    blend_test = np.clip(TEST @ w, 0.5, 210.0)
    sub = pd.DataFrame({"id": np.arange(len(blend_test)),
                        "delivery_time_days": np.round(blend_test, 2)})
    sub.to_csv("submission.csv", index=False)
    json.dump({"models": kept, "chosen": chosen_name, "weights": dict(zip(kept, w.tolist())),
               "chrono_mae": float(c_mae), "oof_mae": float(o_mae)},
              open("blend_result.json", "w"), indent=2)
    print(f"\nwrote submission.csv rows={len(sub)} mean={blend_test.mean():.3f} median={np.median(blend_test):.3f}")

if __name__ == "__main__":
    main(sys.argv[1:])
