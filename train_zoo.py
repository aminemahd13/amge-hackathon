"""Train a diverse model zoo for MAE. Each writes oof_<name>.npy + test_<name>.npy."""
import sys
from model_lib import lgbm_fit, xgb_fit, cat_fit, run_model

LGB_BASE = dict(learning_rate=0.03, num_leaves=63, max_depth=-1, min_child_samples=60,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                lambda_l1=1.0, lambda_l2=2.0, n_estimators=6000, early=200,
                verbosity=-1, seed=42, num_threads=0)

def lgb(obj_extra):
    p = dict(LGB_BASE); p.update(obj_extra); return p

SPECS = {
    # --- LightGBM family: diverse losses & transforms ---
    "lgb_l1":      (lgbm_fit(lgb({"objective":"regression_l1","metric":"mae"})), ),
    "lgb_l1_log":  (lgbm_fit(lgb({"objective":"regression_l1","metric":"mae"}), log_target=True), ),
    "lgb_huber":   (lgbm_fit(lgb({"objective":"huber","alpha":5.0,"metric":"mae"})), ),
    "lgb_q50":     (lgbm_fit(lgb({"objective":"quantile","alpha":0.5,"metric":"mae"})), ),
    "lgb_fair":    (lgbm_fit(lgb({"objective":"fair","fair_c":3.0,"metric":"mae"})), ),
    "lgb_l2_log":  (lgbm_fit(lgb({"objective":"regression","metric":"l2"}), log_target=True), ),
    # deeper / more-regularized variant for diversity
    "lgb_l1_deep": (lgbm_fit(lgb({"objective":"regression_l1","metric":"mae",
                                  "num_leaves":127,"min_child_samples":100,
                                  "feature_fraction":0.7,"learning_rate":0.02})), ),
    # --- XGBoost family ---
    "xgb_mae":     (xgb_fit(dict(objective="reg:absoluteerror", eval_metric="mae",
                                 tree_method="hist", eta=0.03, max_depth=8,
                                 subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                 reg_lambda=2.0, reg_alpha=1.0, n_estimators=6000, early=200,
                                 seed=42, nthread=0)), ),
    "xgb_huber":   (xgb_fit(dict(objective="reg:pseudohubererror", eval_metric="mae",
                                 huber_slope=2.0, tree_method="hist", eta=0.03, max_depth=8,
                                 subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                                 reg_lambda=2.0, reg_alpha=1.0, n_estimators=6000, early=200,
                                 seed=42, nthread=0)), ),
    # --- CatBoost family ---
    "cat_mae":     (cat_fit(dict(loss_function="MAE", eval_metric="MAE", iterations=5000,
                                 learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                                 random_seed=42, thread_count=-1, early=200)), ),
    "cat_mae_log": (cat_fit(dict(loss_function="MAE", eval_metric="MAE", iterations=5000,
                                 learning_rate=0.03, depth=8, l2_leaf_reg=3.0,
                                 random_seed=42, thread_count=-1, early=200), log_target=True), ),
}

if __name__ == "__main__":
    which = sys.argv[1:] if len(sys.argv) > 1 else list(SPECS.keys())
    results = []
    for name in which:
        fit_fn = SPECS[name][0]
        results.append(run_model(name, fit_fn))
    print("\n==== SUMMARY ====")
    for r in sorted(results, key=lambda z: z["chrono_mae"]):
        print(f"{r['name']:16s} chrono={r['chrono_mae']:.4f}  oof={r['oof_mae']:.4f}")
