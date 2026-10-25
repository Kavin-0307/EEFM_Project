"""
Model Training & Evaluation Pipeline (Cleaned Official Crunchbase Dataset)
==========================================================================
Methodology
-----------
* Uses mature cohorts (>= 3 years follow-up) to eliminate right-censoring bias.
* Clean multi-label taxonomy and informative missingness indicators.
* Strict chronological splits on first-funding year:
      train <= 2009 | validation = 2010 | test = 2011 (touched once)
* Model family and hyper-parameters are selected on the VALIDATION set only.
* Probabilities are calibrated (Platt scaling) on the validation set.
* The test set is evaluated once with bootstrap 95% CIs alongside baselines.
* Feature ablation and temporal-vs-random leakage checks are generated.
"""

import itertools
import os
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, "data", "processed", "seed_features.parquet")
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
RESULTS_DIR = os.path.join(ROOT, "artifacts")

SEED = 42
TRAIN_END, VAL_YEAR, TEST_YEAR = 2009, 2010, 2011

NUM_FINANCIAL = ["log_seed_usd", "log_angel_usd", "log_early_funding_usd",
                 "has_seed", "has_angel", "angel_share"]
NUM_TIMING = ["months_to_first_funding", "founding_date_missing"]
NUM_CONTEXT = [
    "is_top_hub", "num_categories", "has_homepage",
    "is_b2b_saas", "is_ai_analytics", "is_biotech_health",
    "is_fintech", "is_consumer_mobile", "is_ecommerce"
]
CAT_CONTEXT = ["country_tier", "macro_sector"]

NUM_ALL = NUM_FINANCIAL + NUM_TIMING + NUM_CONTEXT
CAT_ALL = CAT_CONTEXT


# ---------------------------------------------------------------- metrics
def precision_at_k(y, p, frac=0.10):
    y = np.asarray(y)
    k = max(1, int(len(p) * frac))
    return float(y[np.argsort(p)[-k:]].mean())


def lift_at_k(y, p, frac=0.10):
    return precision_at_k(y, p, frac) / max(float(np.mean(y)), 1e-9)


def score_all(y, p):
    return {
        "PR-AUC": average_precision_score(y, p),
        "ROC-AUC": roc_auc_score(y, p),
        "P@10%": precision_at_k(y, p, 0.10),
        "Lift@10%": lift_at_k(y, p, 0.10),
        "Brier": brier_score_loss(y, np.clip(p, 0, 1)),
    }


def bootstrap_ci(y, p, n_boot=500, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    out = {k: [] for k in ["PR-AUC", "ROC-AUC", "P@10%", "Lift@10%"]}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        s = score_all(y[idx], p[idx])
        for k in out:
            out[k].append(s[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


# ----------------------------------------------------------------- models
def make_preprocessor(num_cols, cat_cols, scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num_cols)]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    return ColumnTransformer(transformers)


def candidate_models():
    """(family, params, estimator factory, needs_scaling)."""
    c = []
    for C in (0.05, 0.2, 1.0):
        c.append(("LogisticRegression", {"C": C},
                  lambda C=C: LogisticRegression(C=C, max_iter=2000, random_state=SEED), True))
    for depth, leaf in itertools.product((6, 10), (10, 30)):
        c.append(("RandomForest", {"max_depth": depth, "min_samples_leaf": leaf},
                  lambda d=depth, l=leaf: RandomForestClassifier(
                      n_estimators=300, max_depth=d, min_samples_leaf=l,
                      random_state=SEED, n_jobs=-1), False))
    for depth, lr in itertools.product((3, 5), (0.03, 0.08)):
        c.append(("XGBoost", {"max_depth": depth, "learning_rate": lr},
                  lambda d=depth, l=lr: xgb.XGBClassifier(
                      n_estimators=250, max_depth=d, learning_rate=l, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
                      random_state=SEED, n_jobs=4), False))
        c.append(("LightGBM", {"max_depth": depth, "learning_rate": lr},
                  lambda d=depth, l=lr: lgb.LGBMClassifier(
                      n_estimators=250, max_depth=d, learning_rate=l, subsample=0.8,
                      subsample_freq=1, colsample_bytree=0.8, min_child_samples=30,
                      random_state=SEED, verbose=-1, n_jobs=4), False))
    return c


def fit_pipe(factory, scale, Xtr, ytr, num_cols=NUM_ALL, cat_cols=CAT_ALL):
    pipe = Pipeline([("prep", make_preprocessor(num_cols, cat_cols, scale)), ("clf", factory())])
    pipe.fit(Xtr[num_cols + cat_cols], ytr)
    return pipe


def predict(pipe, X, num_cols=NUM_ALL, cat_cols=CAT_ALL):
    return pipe.predict_proba(X[num_cols + cat_cols])[:, 1]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(p_val, y_val):
    return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_val), y_val)


def apply_platt(cal, p):
    return cal.predict_proba(_logit(p))[:, 1]


# ------------------------------------------------------------------- main
def load_splits(target_col="target_followon"):
    df = pd.read_parquet(DATA_PATH)
    df = df[(df["is_mature"] == 1)].dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)
    tr = df[df["first_funding_year"] <= TRAIN_END]
    va = df[df["first_funding_year"] == VAL_YEAR]
    te = df[df["first_funding_year"] == TEST_YEAR]
    return df, tr, va, te


def train_and_evaluate(target_col="target_followon", verbose=True):
    df, tr, va, te = load_splits(target_col)
    ytr, yva, yte = tr[target_col], va[target_col], te[target_col]
    print(f"\n=======================================================")
    print(f"Target: {target_col} | Mature Cohorts (>=3y follow-up)")
    print(f"train(<= {TRAIN_END}) n={len(tr)} pos={ytr.mean():.1%} | "
          f"val({VAL_YEAR}) n={len(va)} pos={yva.mean():.1%} | test({TEST_YEAR}) n={len(te)} pos={yte.mean():.1%}")

    # ---- 1. model selection on VALIDATION only ----------------------------
    rows, fitted = [], {}
    for fam, params, factory, scale in candidate_models():
        pipe = fit_pipe(factory, scale, tr, ytr)
        pv = predict(pipe, va)
        rows.append({"family": fam, "params": params, "val PR-AUC": average_precision_score(yva, pv),
                     "val ROC-AUC": roc_auc_score(yva, pv)})
        fitted[(fam, str(params))] = (pipe, scale, factory)
    sel = pd.DataFrame(rows)
    best_per_family = sel.sort_values("val PR-AUC", ascending=False).groupby("family").head(1)
    print("\nBest configuration per family (selected on validation):")
    print(best_per_family.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    winner = best_per_family.iloc[0]
    win_pipe, win_scale, win_factory = fitted[(winner["family"], str(winner["params"]))]
    print(f"\nSelected model: {winner['family']} {winner['params']}")

    # ---- 2. calibration on validation ------------------------------------
    p_val_raw = predict(win_pipe, va)
    calibrator = fit_platt(p_val_raw, yva)

    # ---- 3. one-shot test evaluation + baselines --------------------------
    p_te_raw = predict(win_pipe, te)
    p_te_cal = apply_platt(calibrator, p_te_raw)

    table = {}
    table["Prior (train base rate)"] = score_all(yte, np.full(len(yte), ytr.mean()) + 1e-9 * np.arange(len(yte)))
    fin_pipe = fit_pipe(lambda: LogisticRegression(max_iter=2000), True, tr, ytr, ["log_early_funding_usd"], [])
    table["Baseline: seed amount only (LR)"] = score_all(yte, predict(fin_pipe, te, ["log_early_funding_usd"], []))
    for _, r in best_per_family.iterrows():
        pipe = fitted[(r["family"], str(r["params"]))][0]
        table[f"{r['family']}"] = score_all(yte, predict(pipe, te))
    table[f"** Selected: {winner['family']} (calibrated)"] = score_all(yte, p_te_cal)
    res = pd.DataFrame(table).T
    print(f"\nTEST (cohort {TEST_YEAR}) results:")
    print(res.to_string(float_format=lambda x: f"{x:.4f}"))

    ci = bootstrap_ci(yte.values, p_te_raw)
    print("\nSelected model, 95% bootstrap CI on test:")
    for k, (lo, hi) in ci.items():
        print(f"  {k:9s} [{lo:.4f}, {hi:.4f}]")
    print(f"Test base rate: {yte.mean():.4f}")

    # ---- 4. ablation (feature groups) ------------------------------------
    abl = {}
    groups = {
        "Financing only": (NUM_FINANCIAL, []),
        "+ timing & missingness": (NUM_FINANCIAL + NUM_TIMING, []),
        "+ geography, cleaned sectors & domain flags (all)": (NUM_ALL, CAT_ALL),
    }
    for name, (nc, cc) in groups.items():
        pipe = fit_pipe(win_factory, win_scale, tr, ytr, nc, cc)
        abl[name] = score_all(yte, predict(pipe, te, nc, cc))
    abl = pd.DataFrame(abl).T
    print("\nAblation (selected model family, test set):")
    print(abl[["PR-AUC", "ROC-AUC", "P@10%", "Lift@10%"]].to_string(float_format=lambda x: f"{x:.4f}"))

    # ---- 5. look-ahead inflation: random K-fold vs temporal ---------------
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    X_all, y_all = df, df[target_col]
    oof = np.zeros(len(df))
    for a, b in skf.split(X_all, y_all):
        pipe = fit_pipe(win_factory, win_scale, X_all.iloc[a], y_all.iloc[a])
        oof[b] = predict(pipe, X_all.iloc[b])
    rnd = score_all(y_all, oof)
    leak = pd.DataFrame({"Random 5-fold (leaky)": rnd,
                         "Temporal split (honest)": score_all(yte, p_te_raw)}).T
    print("\nLook-ahead inflation check:")
    print(leak[["PR-AUC", "ROC-AUC", "P@10%", "Lift@10%"]].to_string(float_format=lambda x: f"{x:.4f}"))

    # ---- 6. persist ------------------------------------------------------
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    bundle = {
        "target_col": target_col,
        "model_name": winner["family"],
        "params": winner["params"],
        "pipeline": win_pipe,
        "calibrator": calibrator,
        "num_cols": NUM_ALL,
        "cat_cols": CAT_ALL,
        "feature_names": list(win_pipe.named_steps["prep"].get_feature_names_out()),
        "train_base_rate": float(ytr.mean()),
        "test_base_rate": float(yte.mean()),
        "test_metrics": table[f"** Selected: {winner['family']} (calibrated)"],
        "test_ci": ci,
        "splits": {"train_end": TRAIN_END, "val": VAL_YEAR, "test": TEST_YEAR},
        "background_sample": tr[NUM_ALL + CAT_ALL].sample(min(300, len(tr)), random_state=SEED),
    }
    path = os.path.join(ARTIFACTS_DIR, f"best_seed_model_target_series_a.joblib")
    joblib.dump(bundle, path)
    res.to_csv(os.path.join(RESULTS_DIR, "test_results.csv"))
    abl.to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"))
    leak.to_csv(os.path.join(RESULTS_DIR, "leakage_check.csv"))
    print(f"\nSaved model bundle -> {path}")
    return bundle


if __name__ == "__main__":
    train_and_evaluate("target_followon")
