"""
Explainability & Heuristic Checklist (Cleaned Official Model)
=============================================================
* Calibrated Series A / Follow-on probability for early-stage deals.
* SHAP attributions aggregated back to the original semantic input features.
* Transparent rule-based checklist based on early-stage investment heuristics.
"""

import os

import joblib
import numpy as np
import pandas as pd
import shap

from src.models.train_models import apply_platt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(ROOT, "artifacts", "best_seed_model_target_series_a.joblib")

DEFAULTS = {
    "seed_usd": 1_000_000.0,
    "angel_usd": 0.0,
    "months_to_first_funding": np.nan,
    "is_top_hub": 0,
    "country_tier": "USA",
    "macro_sector": "Enterprise & B2B SaaS",
    "is_b2b_saas": 1,
    "is_ai_analytics": 0,
    "is_biotech_health": 0,
    "is_fintech": 0,
    "is_consumer_mobile": 0,
    "is_ecommerce": 0,
    "num_categories": 2,
    "has_homepage": 1,
}


def deal_to_features(deal: dict) -> pd.DataFrame:
    """Raw deal inputs -> the exact cleaned feature frame used in training."""
    d = {**DEFAULTS, **deal}
    seed, angel = float(d["seed_usd"]), float(d["angel_usd"])
    early = seed + angel
    months = d["months_to_first_funding"]
    
    # Auto-derive domain flags if macro_sector provided and flags not explicitly set
    sec = str(d.get("macro_sector", "")).lower()
    is_b2b = int(d.get("is_b2b_saas", 1 if any(k in sec for k in ['enterprise', 'saas', 'b2b']) else 0))
    is_ai = int(d.get("is_ai_analytics", 1 if any(k in sec for k in ['ai', 'analytics']) else 0))
    is_bio = int(d.get("is_biotech_health", 1 if any(k in sec for k in ['biotech', 'health']) else 0))
    is_fin = int(d.get("is_fintech", 1 if any(k in sec for k in ['fintech', 'finance']) else 0))
    is_mob = int(d.get("is_consumer_mobile", 1 if any(k in sec for k in ['mobile', 'consumer']) else 0))
    is_ecom = int(d.get("is_ecommerce", 1 if any(k in sec for k in ['e-commerce', 'ecommerce', 'retail']) else 0))

    row = {
        "log_seed_usd": np.log1p(seed),
        "log_angel_usd": np.log1p(angel),
        "log_early_funding_usd": np.log1p(early),
        "has_seed": int(seed > 0),
        "has_angel": int(angel > 0),
        "angel_share": angel / early if early > 0 else 0.0,
        "months_to_first_funding": np.nan if months is None else float(months),
        "founding_date_missing": int(months is None or (isinstance(months, float) and np.isnan(months))),
        "is_top_hub": int(d["is_top_hub"]),
        "num_categories": int(d.get("num_categories", 2)),
        "has_homepage": int(d.get("has_homepage", 1)),
        "is_b2b_saas": is_b2b,
        "is_ai_analytics": is_ai,
        "is_biotech_health": is_bio,
        "is_fintech": is_fin,
        "is_consumer_mobile": is_mob,
        "is_ecommerce": is_ecom,
        "country_tier": d["country_tier"],
        "macro_sector": d["macro_sector"],
    }
    return pd.DataFrame([row])


class VentureExplainer:
    def __init__(self, model_bundle_path=MODEL_PATH):
        if not os.path.exists(model_bundle_path):
            raise FileNotFoundError(f"Model bundle not found at {model_bundle_path}. Run src/models/train_models.py first.")
        b = joblib.load(model_bundle_path)
        self.bundle = b
        self.pipe = b["pipeline"]
        self.calibrator = b["calibrator"]
        self.num_cols, self.cat_cols = b["num_cols"], b["cat_cols"]
        self.prep = self.pipe.named_steps["prep"]
        self.clf = self.pipe.named_steps["clf"]
        self.feature_names = b["feature_names"]
        self.train_base_rate = b["train_base_rate"]

        bg = self.prep.transform(b["background_sample"][self.num_cols + self.cat_cols])
        if b["model_name"] == "LogisticRegression":
            self.explainer = shap.LinearExplainer(self.clf, bg)
        else:
            self.explainer = shap.TreeExplainer(self.clf)

    def _original_name(self, transformed_name: str) -> str:
        n = transformed_name.split("__", 1)[-1]
        for c in self.cat_cols:
            if n.startswith(c + "_"):
                return c
        return n

    def _local_shap(self, X_trans):
        sv = self.explainer.shap_values(X_trans)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., -1]
        return sv[0]

    def evaluate_deal(self, deal: dict) -> dict:
        X = deal_to_features(deal)[self.num_cols + self.cat_cols]
        X_trans = self.prep.transform(X)
        raw = float(self.clf.predict_proba(X_trans)[0, 1])
        prob = float(apply_platt(self.calibrator, np.array([raw]))[0])

        shap_df = (pd.DataFrame({
            "Feature": [self._original_name(n) for n in self.feature_names],
            "Attribution": self._local_shap(X_trans),
        }).groupby("Feature", as_index=False)["Attribution"].sum()
          .sort_values("Attribution", key=abs, ascending=False).reset_index(drop=True))

        lift = prob / self.train_base_rate
        if lift >= 1.25:
            tier = "High Conviction (Tier 1)"
        elif lift >= 0.85:
            tier = "Moderate Watchlist (Tier 2)"
        else:
            tier = "Low / Pass (Tier 3)"

        return {
            "probability_series_a": prob,
            "base_rate": self.train_base_rate,
            "lift_vs_base": lift,
            "conviction_tier": tier,
            "top_positive_factors": shap_df[shap_df.Attribution > 0].head(4).to_dict("records"),
            "top_risk_factors": shap_df[shap_df.Attribution < 0].head(4).to_dict("records"),
            "checklist": self.heuristic_checklist(deal),
            "shap_dataframe": shap_df,
        }

    @staticmethod
    def heuristic_checklist(deal: dict) -> dict:
        """Transparent rule-based screen independent of the ML model."""
        d = {**DEFAULTS, **deal}
        items = []

        def add(pillar, text, ok):
            items.append({"Pillar": pillar, "Check": text, "Pass": bool(ok)})

        total = float(d["seed_usd"]) + float(d["angel_usd"])
        add("Financing", "Early capital between $500k and $5M", 5e5 <= total <= 5e6)
        add("Financing", "Seed round (not angel-only) present", float(d["seed_usd"]) > 0)
        m = d["months_to_first_funding"]
        known = m is not None and not (isinstance(m, float) and np.isnan(m))
        add("Momentum", "First check within 24 months of founding", known and float(m) <= 24)
        add("Ecosystem", "Located in top venture hub (SV/NY/Boston/London/BLR)", int(d["is_top_hub"]) == 1)
        add("Ecosystem", "Located in Tier-1 venture country", d["country_tier"] != "OTHER")
        add("Online Presence", "Verified homepage/domain documented", int(d.get("has_homepage", 1)) == 1)

        passed = sum(i["Pass"] for i in items)
        return {"score": passed, "max_score": len(items), "items": items}


if __name__ == "__main__":
    ex = VentureExplainer()
    test_deal = {
        "seed_usd": 2_500_000,
        "angel_usd": 250_000,
        "months_to_first_funding": 8.0,
        "is_top_hub": 1,
        "country_tier": "USA",
        "macro_sector": "Enterprise & B2B SaaS",
        "is_b2b_saas": 1,
        "is_ai_analytics": 1
    }
    res = ex.evaluate_deal(test_deal)
    print(f"Calibrated P(Follow-on): {res['probability_series_a']:.1%} "
          f"(base {res['base_rate']:.1%}, lift {res['lift_vs_base']:.2f}x) -> {res['conviction_tier']}")
    print(f"Checklist: {res['checklist']['score']}/{res['checklist']['max_score']}")
    print(res["shap_dataframe"].head(8).to_string(index=False))
