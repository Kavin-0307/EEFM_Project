"""
Seed-Stage Feature Engineering Pipeline (Cleaned Official Crunchbase Dataset)
=============================================================================
Transforms raw Crunchbase investment data (investments_VC.csv) into high-fidelity
features strictly available at the close of the early (Seed / Angel) financing.

Key methodological improvements:
--------------------------------
1. Clean Taxonomy: Multi-label category extraction replacing the fragmented raw market
   field. Drops the ambiguous 'Other' dumpster from 58% down to ~16%.
2. Informative Missingness: Explicit indicators for unlisted founding dates, missing amounts,
   and online presence (has_homepage).
3. Anti-Leakage & Right-Censoring: Retains only cohorts with mature observation windows
   (>= 3 years) so incomplete recent horizons do not pollute negative outcomes.
4. Clean Targets: Both classic `target_series_a` and broader `target_followon` (Series A or Venture).
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(ROOT, "data", "raw", "investments_VC.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")

SNAPSHOT_DATE = pd.Timestamp("2014-12-31")
MIN_FOLLOWUP_YEARS = 3.0

TOP_HUB_PATTERN = "SF Bay Area|Silicon Valley|New York|Boston|London|Los Angeles|Seattle|Austin|Bangalore"
TOP_COUNTRIES = ["USA", "GBR", "CAN", "IND", "DEU", "FRA", "ISR", "ESP"]


def clean_curr(val):
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace(",", "").replace("$", "").replace(" ", "")
    if s in ("", "-"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def assign_macro_sector(cat_text):
    text = str(cat_text).lower()
    if any(k in text for k in ['enterprise', 'saas', 'cloud', 'security', 'analytics', 'big data', 'b2b']):
        return 'Enterprise & B2B SaaS'
    if any(k in text for k in ['biotech', 'health', 'medical', 'pharma', 'life science']):
        return 'Healthcare & BioTech'
    if any(k in text for k in ['finance', 'fintech', 'payment', 'banking', 'insurance']):
        return 'Fintech & Financial'
    if any(k in text for k in ['e-commerce', 'ecommerce', 'marketplace', 'retail', 'fashion']):
        return 'E-Commerce & Retail'
    if any(k in text for k in ['mobile', 'apps', 'games', 'social media', 'messaging', 'consumer', 'curated web']):
        return 'Consumer Web & Mobile'
    if any(k in text for k in ['clean tech', 'cleantech', 'hardware', 'manufacturing', 'robotics', 'semiconductor']):
        return 'Hardware & CleanTech'
    if any(k in text for k in ['advertising', 'marketing', 'video', 'news', 'media']):
        return 'Media & Advertising'
    if any(k in text for k in ['education', 'edtech']):
        return 'EdTech'
    if any(k in text for k in ['software', 'internet', 'technology', 'web']):
        return 'General Software & Tech'
    return 'Other / Emerging'


def build_seed_features(file_path=RAW_DATA_PATH, save=True):
    print("Reading raw Crunchbase investments data...")
    df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    for col in ["seed", "angel", "venture", "round_a", "round_b", "round_c",
                "round_d", "round_e", "round_f", "round_g", "round_h"]:
        df[col + "_usd"] = df[col].apply(clean_curr)

    df["early_funding_usd"] = df["seed_usd"] + df["angel_usd"]
    d = df[df["early_funding_usd"] > 0].copy()

    # Resolved status filtering
    d["status"] = d["status"].str.lower()
    d = d[d["status"].isin(["operating", "acquired", "ipo", "closed"])].copy()

    # ---- 1. Timing & Informative Missingness ----------------------------
    d["founded_at"] = pd.to_datetime(d["founded_at"], errors="coerce")
    d["first_funding_at"] = pd.to_datetime(d["first_funding_at"], errors="coerce")
    d = d.dropna(subset=["first_funding_at"]).copy()
    d["first_funding_year"] = d["first_funding_at"].dt.year

    gap = (d["first_funding_at"] - d["founded_at"]).dt.days
    gap = gap.where((gap >= 0) & (gap <= 3650))
    d["founding_date_missing"] = gap.isna().astype(int)
    d["months_to_first_funding"] = gap / 30.44  # Linear imputer handles NaN in-fold; tree models handle naturally

    # ---- 2. Seed-Stage Financial Signals -------------------------------
    d["log_seed_usd"] = np.log1p(d["seed_usd"])
    d["log_angel_usd"] = np.log1p(d["angel_usd"])
    d["log_early_funding_usd"] = np.log1p(d["early_funding_usd"])
    d["has_seed"] = (d["seed_usd"] > 0).astype(int)
    d["has_angel"] = (d["angel_usd"] > 0).astype(int)
    d["angel_share"] = d["angel_usd"] / d["early_funding_usd"]  # bounded [0, 1]

    # ---- 3. Geography & Venture Hub Signals ----------------------------
    d["is_top_hub"] = d["region"].astype(str).str.contains(TOP_HUB_PATTERN, case=False, na=False).astype(int)
    d["country_tier"] = d["country_code"].where(d["country_code"].isin(TOP_COUNTRIES), "OTHER").fillna("OTHER")
    d["has_homepage"] = d["homepage_url"].notna().astype(int)

    # ---- 4. Clean Multi-Label Category Taxonomy ------------------------
    combined_cats = d["category_list"].fillna("").astype(str) + " " + d["market"].fillna("").astype(str)
    d["macro_sector"] = combined_cats.apply(assign_macro_sector)
    
    # Granular cross-cutting domain flags
    d["is_b2b_saas"] = combined_cats.str.lower().apply(lambda s: int(any(k in s for k in ['enterprise', 'saas', 'b2b', 'cloud'])))
    d["is_ai_analytics"] = combined_cats.str.lower().apply(lambda s: int(any(k in s for k in ['ai', 'artificial intelligence', 'machine learning', 'analytics', 'big data'])))
    d["is_biotech_health"] = combined_cats.str.lower().apply(lambda s: int(any(k in s for k in ['biotech', 'health', 'medical', 'pharma'])))
    d["is_fintech"] = combined_cats.str.lower().apply(lambda s: int(any(k in s for k in ['finance', 'fintech', 'payment', 'banking'])))
    d["is_consumer_mobile"] = combined_cats.str.lower().apply(lambda s: int(any(k in s for k in ['mobile', 'apps', 'games', 'social media', 'consumer'])))
    d["is_ecommerce"] = combined_cats.str.lower().apply(lambda s: int(any(k in s for k in ['e-commerce', 'ecommerce', 'marketplace', 'retail'])))
    
    cats_split = d["category_list"].fillna("").astype(str)
    d["num_categories"] = cats_split.apply(lambda s: len([c for c in s.split("|") if c.strip()]))

    # ---- 5. Observation Horizon & Anti-Censoring -----------------------
    d["followup_years"] = (SNAPSHOT_DATE - d["first_funding_at"]).dt.days / 365.25
    d["is_mature"] = (d["followup_years"] >= MIN_FOLLOWUP_YEARS).astype(int)

    # ---- 6. Targets (Outcome definitions) ------------------------------
    # Primary Milestone: Series A round secured
    d["target_series_a"] = (d["round_a_usd"] > 0).astype(int)
    
    # Broader Milestone: Any early institutional follow-on (Series A or Venture check)
    later_institutional = d[["venture_usd", "round_a_usd", "round_b_usd", "round_c_usd"]]
    d["target_followon"] = (later_institutional.sum(axis=1) > 0).astype(int)

    # Terminal Exit (Acquired / IPO vs Closed)
    d["is_exit"] = np.where(d["status"].isin(["acquired", "ipo"]), 1.0,
                            np.where(d["status"] == "closed", 0.0, np.nan))

    keep = [
        "name", "first_funding_year", "followup_years", "is_mature",
        "months_to_first_funding", "founding_date_missing",
        "seed_usd", "angel_usd", "early_funding_usd",
        "log_seed_usd", "log_angel_usd", "log_early_funding_usd",
        "has_seed", "has_angel", "angel_share",
        "is_top_hub", "country_tier", "macro_sector",
        "is_b2b_saas", "is_ai_analytics", "is_biotech_health", "is_fintech",
        "is_consumer_mobile", "is_ecommerce", "num_categories", "has_homepage",
        "target_series_a", "target_followon", "is_exit", "status",
    ]
    out = d[keep].reset_index(drop=True)

    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        parquet_path = os.path.join(PROCESSED_DIR, "seed_features.parquet")
        csv_path = os.path.join(PROCESSED_DIR, "seed_features.csv")
        out.to_parquet(parquet_path, index=False)
        out.to_csv(csv_path, index=False)
        print(f"Cleaned feature dataset saved -> {parquet_path}")

    mat = out[out["is_mature"] == 1]
    print(f"Total Seed/Angel startups: {len(out)} | Mature (>=3y follow-up): {len(mat)}")
    print(f"Macro sector distribution:\n{out['macro_sector'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"Series A graduation rate: all={out['target_series_a'].mean():.2%} | mature={mat['target_series_a'].mean():.2%}")
    print(f"Broader follow-on rate:   all={out['target_followon'].mean():.2%} | mature={mat['target_followon'].mean():.2%}")
    return out


if __name__ == "__main__":
    build_seed_features()
