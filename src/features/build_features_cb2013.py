"""
Crunchbase-2013 seed-stage dataset with dated rounds (point-in-time)
====================================================================
Cohort : companies whose FIRST financing event is an angel round (date T0).
Label  : raised a Series A within 24 months AFTER T0 (strictly later date).
         A broader `target_followon_24m` (series-a/b/c+/venture) is also stored.
Censor : the export ends 2013-10-01, so only companies with T0 <= 2011-10-01
         have a full 24-month window; later companies are dropped (not negatives).
Leakage: features use only information dated on or before T0:
         - first-round amount, company age at T0, geography, category
         - investors in the first round and how many deals each had done BEFORE T0
         - market heat: number of angel rounds in the 12 months before T0
         Never used: status, funding_total_usd, funding_rounds, last_funding_at,
         last_milestone_at (all contain post-T0 information).
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "external", "crunchbase-october-2013")
OUT = os.path.join(ROOT, "data", "processed", "cb2013_seed_features.parquet")

SNAPSHOT = pd.Timestamp("2013-10-01")
HORIZON_DAYS = 730
MIN_YEAR = 2000
HUB = "SF Bay Area|New York|Boston|London|Los Angeles|Seattle|Austin|Bangalore"
TOP_COUNTRIES = ["USA", "GBR", "CAN", "IND", "DEU", "FRA", "ISR", "ESP"]
FOLLOWON = ["series-a", "series-b", "series-c+", "venture"]


def build(save=True):
    comp = pd.read_csv(os.path.join(SRC, "crunchbase-companies.csv"), encoding="latin1", low_memory=False)
    rnd = pd.read_csv(os.path.join(SRC, "crunchbase-rounds.csv"), encoding="latin1", low_memory=False)
    inv = pd.read_csv(os.path.join(SRC, "crunchbase-investments.csv"), encoding="latin1", low_memory=False)

    rnd["funded_at"] = pd.to_datetime(rnd["funded_at"], errors="coerce")
    inv["funded_at"] = pd.to_datetime(inv["funded_at"], errors="coerce")
    rnd = rnd.dropna(subset=["funded_at"])
    inv = inv.dropna(subset=["funded_at"])
    comp["founded_at"] = pd.to_datetime(comp["founded_at"], errors="coerce")

    # ---- cohort: first event is angel ------------------------------------
    t0 = rnd.groupby("company_permalink")["funded_at"].min().rename("t0")
    rnd = rnd.join(t0, on="company_permalink")
    first = rnd[rnd["funded_at"] == rnd["t0"]]
    only_angel = first.groupby("company_permalink")["funding_round_type"].agg(lambda s: set(s) == {"angel"})
    cohort = only_angel[only_angel].index
    first = first[first["company_permalink"].isin(cohort)]
    base = (first.groupby("company_permalink")
                 .agg(t0=("t0", "first"),
                      first_amount=("raised_amount_usd", lambda s: s.sum(min_count=1)),
                      n_first_rows=("funding_round_type", "size"),
                      country_code=("company_country_code", "first"),
                      state_code=("company_state_code", "first"),
                      region=("company_region", "first"),
                      category=("company_category_code", "first"))
                 .reset_index())
    base = base[(base["t0"].dt.year >= MIN_YEAR) & (base["t0"] <= SNAPSHOT)]

    # ---- labels (future rounds only) -------------------------------------
    later = rnd.merge(base[["company_permalink", "t0"]], on="company_permalink", suffixes=("", "_c"))
    later = later[(later["funded_at"] > later["t0_c"]) &
                  (later["funded_at"] <= later["t0_c"] + pd.Timedelta(days=HORIZON_DAYS))]
    a = set(later.loc[later["funding_round_type"] == "series-a", "company_permalink"])
    f = set(later.loc[later["funding_round_type"].isin(FOLLOWON), "company_permalink"])
    base["target_series_a_24m"] = base["company_permalink"].isin(a).astype(int)
    base["target_followon_24m"] = base["company_permalink"].isin(f).astype(int)

    # ---- censoring --------------------------------------------------------
    base["followup_days"] = (SNAPSHOT - base["t0"]).dt.days
    base["is_mature"] = (base["followup_days"] >= HORIZON_DAYS).astype(int)
    base["t0_year"] = base["t0"].dt.year

    # ---- point-in-time features ------------------------------------------
    base = base.merge(comp[["permalink", "founded_at"]].rename(columns={"permalink": "company_permalink"}),
                      on="company_permalink", how="left")
    age = (base["t0"] - base["founded_at"]).dt.days / 30.44
    base["months_since_founding"] = age.where((age >= 0) & (age <= 240))
    base["founding_date_missing"] = base["months_since_founding"].isna().astype(int)
    base["amount_missing"] = base["first_amount"].isna().astype(int)
    base["log_first_amount"] = np.log1p(base["first_amount"])

    base["is_top_hub"] = base["region"].astype(str).str.contains(HUB, case=False, na=False).astype(int)
    base["country_tier"] = base["country_code"].where(base["country_code"].isin(TOP_COUNTRIES), "OTHER").fillna("OTHER")
    top_cats = base["category"].value_counts().head(12).index
    base["category_group"] = base["category"].where(base["category"].isin(top_cats), "other").fillna("other")

    # market heat: angel rounds in the 12 months before T0 (all companies)
    ang_dates = np.sort(rnd.loc[rnd["funding_round_type"] == "angel", "funded_at"].values)
    lo = np.searchsorted(ang_dates, (base["t0"] - pd.Timedelta(days=365)).values, side="left")
    hi = np.searchsorted(ang_dates, base["t0"].values, side="left")
    base["market_heat_12m"] = np.log1p(hi - lo)

    # investors in the first round and their prior experience (deals strictly before T0)
    inv_first = inv.merge(base[["company_permalink", "t0"]], on="company_permalink")
    inv_first = inv_first[inv_first["funded_at"] == inv_first["t0"]]
    inv_first = inv_first.drop_duplicates(["company_permalink", "investor_permalink"])
    deals = inv[["investor_permalink", "company_permalink", "funded_at"]].drop_duplicates()
    deals_by_inv = {k: np.sort(v.values) for k, v in deals.groupby("investor_permalink")["funded_at"]}

    def prior(row):
        d = deals_by_inv.get(row.investor_permalink)
        return 0 if d is None else int(np.searchsorted(d, row.t0.to_datetime64(), side="left"))
    inv_first["prior_deals"] = [prior(r) for r in inv_first.itertuples()]
    g = inv_first.groupby("company_permalink")["prior_deals"]
    invf = pd.DataFrame({"n_investors": g.size(), "max_prior_deals": g.max(), "mean_prior_deals": g.mean(),
                         "n_experienced_investors": g.apply(lambda s: int((s >= 10).sum()))}).reset_index()
    base = base.merge(invf, on="company_permalink", how="left")
    base["investors_listed"] = base["n_investors"].notna().astype(int)
    for c in ["n_investors", "max_prior_deals", "mean_prior_deals", "n_experienced_investors"]:
        base[c] = base[c].fillna(0)
    base["log_max_prior_deals"] = np.log1p(base["max_prior_deals"])
    base["log_mean_prior_deals"] = np.log1p(base["mean_prior_deals"])

    keep = ["company_permalink", "t0", "t0_year", "followup_days", "is_mature",
            "log_first_amount", "amount_missing", "months_since_founding", "founding_date_missing",
            "is_top_hub", "country_tier", "category_group", "market_heat_12m",
            "n_investors", "investors_listed", "log_max_prior_deals", "log_mean_prior_deals",
            "n_experienced_investors", "target_series_a_24m", "target_followon_24m"]
    out = base[keep].reset_index(drop=True)
    if save:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        out.to_parquet(OUT, index=False)
    m = out[out.is_mature == 1]
    print(f"angel-first cohort: {len(out)} | mature (24m window complete): {len(m)}")
    print(m.groupby("t0_year")[["target_series_a_24m", "target_followon_24m"]].agg(["mean", "count"]).round(3).iloc[:, [0, 1, 2]].to_string())
    print("investor info available for", f"{m.investors_listed.mean():.0%}", "of mature rows")
    return out


if __name__ == "__main__":
    build()
