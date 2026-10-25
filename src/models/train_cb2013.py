"""
Train & evaluate on the Crunchbase-2013 dated-round dataset
===========================================================
Target : Series A within 24 months of the first angel round.
Splits : train t0<=2008 | validation 2009 | test 2010-2011 (touched once).
Also   : baselines, feature-group ablation, random-vs-temporal leakage check,
         and forward-chaining (expanding window) robustness over 2009/2010/2011.
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.train_models import (apply_platt, bootstrap_ci, candidate_models, fit_pipe,  # noqa: E402
                                     fit_platt, predict, score_all)

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "processed", "cb2013_seed_features.parquet")
ART = os.path.join(ROOT, "artifacts")
TARGET = "target_series_a_24m"
TRAIN_END, VAL_YEAR, TEST_YEARS = 2008, 2009, (2010, 2011)

G_FIN = ["log_first_amount", "amount_missing"]
G_TIME = ["months_since_founding", "founding_date_missing"]
G_GEO = ["is_top_hub"]
G_MKT = ["market_heat_12m"]
G_INV = ["n_investors", "investors_listed", "log_max_prior_deals", "log_mean_prior_deals", "n_experienced_investors"]
CAT = ["country_tier", "category_group"]
NUM_ALL = G_FIN + G_TIME + G_GEO + G_MKT + G_INV


def main():
    df = pd.read_parquet(DATA)
    df = df[df.is_mature == 1].copy()
    tr = df[df.t0_year <= TRAIN_END]
    va = df[df.t0_year == VAL_YEAR]
    te = df[df.t0_year.isin(TEST_YEARS)]
    ytr, yva, yte = tr[TARGET], va[TARGET], te[TARGET]
    print(f"train(<= {TRAIN_END}) n={len(tr)} pos={ytr.mean():.1%} ({int(ytr.sum())}) | "
          f"val({VAL_YEAR}) n={len(va)} pos={yva.mean():.1%} ({int(yva.sum())}) | "
          f"test({TEST_YEARS[0]}-{TEST_YEARS[1]}) n={len(te)} pos={yte.mean():.1%} ({int(yte.sum())})")

    # ---- selection on validation ----------------------------------------
    rows, fitted = [], {}
    for fam, params, factory, scale in candidate_models():
        pipe = fit_pipe(factory, scale, tr, ytr, NUM_ALL, CAT)
        pv = predict(pipe, va, NUM_ALL, CAT)
        rows.append({"family": fam, "params": params, "val PR-AUC": average_precision_score(yva, pv),
                     "val ROC-AUC": roc_auc_score(yva, pv)})
        fitted[(fam, str(params))] = (pipe, scale, factory)
    sel = pd.DataFrame(rows)
    best = sel.sort_values("val PR-AUC", ascending=False).groupby("family").head(1)
    print("\nBest per family (validation):")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    w = best.iloc[0]
    pipe, scale, factory = fitted[(w["family"], str(w["params"]))]
    print(f"\nSelected: {w['family']} {w['params']}")

    cal = fit_platt(predict(pipe, va, NUM_ALL, CAT), yva)
    p_raw = predict(pipe, te, NUM_ALL, CAT)
    p_cal = apply_platt(cal, p_raw)

    # ---- test table -------------------------------------------------------
    table = {"Prior (train base rate)": score_all(yte, np.full(len(yte), ytr.mean()) + 1e-9 * np.arange(len(yte)))}
    b1 = fit_pipe(lambda: LogisticRegression(max_iter=2000), True, tr, ytr, ["log_first_amount", "amount_missing"], [])
    table["Baseline: first-round amount only (LR)"] = score_all(yte, predict(b1, te, ["log_first_amount", "amount_missing"], []))
    for _, r in best.iterrows():
        table[r["family"]] = score_all(yte, predict(fitted[(r["family"], str(r["params"]))][0], te, NUM_ALL, CAT))
    table[f"** Selected: {w['family']} (calibrated)"] = score_all(yte, p_cal)
    res = pd.DataFrame(table).T
    print(f"\nTEST ({TEST_YEARS[0]}-{TEST_YEARS[1]}):")
    print(res.to_string(float_format=lambda x: f"{x:.4f}"))
    ci = bootstrap_ci(yte.values, p_raw)
    print("\n95% bootstrap CI (selected model):", {k: (round(a, 3), round(b, 3)) for k, (a, b) in ci.items()})

    # ---- ablation ----------------------------------------------------------
    groups = {"Funding amount only": (G_FIN, []),
              "+ company age & hub": (G_FIN + G_TIME + G_GEO, []),
              "+ country & category": (G_FIN + G_TIME + G_GEO, CAT),
              "+ market heat": (G_FIN + G_TIME + G_GEO + G_MKT, CAT),
              "+ investor experience (all)": (NUM_ALL, CAT)}
    abl = {}
    for name, (nc, cc) in groups.items():
        pp = fit_pipe(factory, scale, tr, ytr, nc, cc)
        abl[name] = score_all(yte, predict(pp, te, nc, cc))
    abl = pd.DataFrame(abl).T
    print("\nAblation (test):")
    print(abl[["PR-AUC", "ROC-AUC", "P@10%", "Lift@10%"]].to_string(float_format=lambda x: f"{x:.4f}"))

    # ---- leakage check ----------------------------------------------------
    oof = np.zeros(len(df))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(df, df[TARGET]):
        oof[b] = predict(fit_pipe(factory, scale, df.iloc[a], df.iloc[a][TARGET], NUM_ALL, CAT), df.iloc[b], NUM_ALL, CAT)
    leak = pd.DataFrame({"Random 5-fold": score_all(df[TARGET], oof),
                         "Temporal split": score_all(yte, p_raw)}).T
    print("\nLook-ahead check:")
    print(leak[["PR-AUC", "ROC-AUC", "P@10%", "Lift@10%"]].to_string(float_format=lambda x: f"{x:.4f}"))

    # ---- forward chaining -------------------------------------------------
    fc = []
    for y in (2009, 2010, 2011):
        a, b = df[df.t0_year < y], df[df.t0_year == y]
        pp = fit_pipe(factory, scale, a, a[TARGET], NUM_ALL, CAT)
        s = score_all(b[TARGET], predict(pp, b, NUM_ALL, CAT))
        fc.append({"test year": y, "n_train": len(a), "n_test": len(b), "base": b[TARGET].mean(),
                   "PR-AUC": s["PR-AUC"], "ROC-AUC": s["ROC-AUC"], "Lift@10%": s["Lift@10%"]})
    fc = pd.DataFrame(fc)
    print("\nForward-chaining (train on all earlier years):")
    print(fc.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    os.makedirs(ART, exist_ok=True)
    joblib.dump({"pipeline": pipe, "calibrator": cal, "model_name": w["family"], "params": w["params"],
                 "num_cols": NUM_ALL, "cat_cols": CAT, "target": TARGET,
                 "test_metrics": table[f"** Selected: {w['family']} (calibrated)"], "test_ci": ci},
                os.path.join(ART, "cb2013_series_a_24m.joblib"))
    res.to_csv(os.path.join(ART, "cb2013_test_results.csv"))
    abl.to_csv(os.path.join(ART, "cb2013_ablation.csv"))
    leak.to_csv(os.path.join(ART, "cb2013_leakage_check.csv"))
    fc.to_csv(os.path.join(ART, "cb2013_forward_chaining.csv"), index=False)


if __name__ == "__main__":
    main()
