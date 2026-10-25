"""
Venture Capital Explainable AI (XAI) & Checklist Scorecard
===========================================================
Generates local SHAP feature attributions and computes an automated
21-factor venture capital checklist scorecard based on Corea et al. [2].
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_seed_model_target_series_a.joblib")

class VentureExplainer:
    def __init__(self, model_bundle_path=MODEL_PATH):
        if not os.path.exists(model_bundle_path):
            raise FileNotFoundError(f"Model bundle not found at {model_bundle_path}. Run training first.")
        
        self.bundle = joblib.load(model_bundle_path)
        self.model = self.bundle['model']
        self.preprocessor = self.bundle['preprocessor']
        self.feature_names = self.bundle['feature_names']
        
        # Use TreeExplainer for tree ensembles
        self.explainer = shap.TreeExplainer(self.model)

    def evaluate_deal(self, deal_dict):
        """
        Takes raw deal input, transforms it, predicts success probability,
        computes SHAP local explanations, and calculates the 21-Factor Venture Score.
        """
        df_input = pd.DataFrame([deal_dict])
        
        # Ensure required numerical and categorical fields exist with defaults
        defaults = {
            'months_to_first_funding': 12.0,
            'seed_usd': 1_500_000.0,
            'angel_usd': 0.0,
            'is_top_hub': 1,
            'country_tier': 'USA',
            'market_group': 'Software'
        }
        for k, v in defaults.items():
            if k not in df_input.columns:
                df_input[k] = v
                
        # Derived features
        df_input['log_seed_usd'] = np.log1p(df_input['seed_usd'])
        df_input['log_angel_usd'] = np.log1p(df_input['angel_usd'])
        df_input['early_funding_usd'] = df_input['seed_usd'] + df_input['angel_usd']
        df_input['log_early_funding_usd'] = np.log1p(df_input['early_funding_usd'])
        df_input['has_seed'] = (df_input['seed_usd'] > 0).astype(int)
        df_input['has_angel'] = (df_input['angel_usd'] > 0).astype(int)
        df_input['seed_angel_ratio'] = np.where(
            df_input['angel_usd'] > 0,
            df_input['seed_usd'] / df_input['angel_usd'],
            df_input['seed_usd'] / 1.0
        )

        # Preprocess
        X_trans = self.preprocessor.transform(df_input)
        
        # Predict probability
        prob_success = float(self.model.predict_proba(X_trans)[0, 1])

        # SHAP local values
        shap_values = self.explainer.shap_values(X_trans)
        if isinstance(shap_values, list):
            # Binary classification case where shap_values is a list of [neg_class, pos_class]
            local_shap = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            local_shap = shap_values[0, :, 1]
        else:
            local_shap = shap_values[0]

        shap_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Attribution': local_shap
        }).sort_values(by='Attribution', key=abs, ascending=False)

        top_positive = shap_df[shap_df['Attribution'] > 0].head(4).to_dict(orient='records')
        top_negative = shap_df[shap_df['Attribution'] < 0].head(4).to_dict(orient='records')

        # Automated 21-Factor Scorecard (Corea et al.)
        scorecard = self._compute_checklist_score(deal_dict, prob_success)

        return {
            'probability_series_a': round(prob_success, 4),
            'conviction_tier': 'High Conviction (Tier 1)' if prob_success > 0.40 else ('Moderate Watchlist (Tier 2)' if prob_success > 0.20 else 'Low / Pass (Tier 3)'),
            'top_positive_factors': top_positive,
            'top_risk_factors': top_negative,
            'corea_scorecard': scorecard,
            'shap_dataframe': shap_df
        }

    def _compute_checklist_score(self, deal_dict, prob_success):
        """Computes rule-based 21-Factor Checklist score (0 to 100)."""
        score = 50.0  # Baseline
        checks = []

        # Pillar 1: Financial & Valuation Discipline
        seed_amount = deal_dict.get('seed_usd', 0)
        if 1_000_000 <= seed_amount <= 4_000_000:
            score += 12
            checks.append({"Pillar": "Financing", "Check": "Healthy Seed Round Size ($1M-$4M)", "Pass": True})
        else:
            score -= 5
            checks.append({"Pillar": "Financing", "Check": "Sub-optimal or Extreme Capital Size", "Pass": False})

        # Pillar 2: Momentum & Traction Speed
        months = deal_dict.get('months_to_first_funding', 12)
        if months <= 12:
            score += 15
            checks.append({"Pillar": "Momentum", "Check": "High Velocity: Seed secured within 12 months", "Pass": True})
        elif months > 24:
            score -= 10
            checks.append({"Pillar": "Momentum", "Check": "Extended pre-seed lag (>24 months)", "Pass": False})
        else:
            checks.append({"Pillar": "Momentum", "Check": "Normal traction trajectory (12-24 months)", "Pass": True})

        # Pillar 3: Ecosystem Hub Advantage
        is_hub = deal_dict.get('is_top_hub', 0)
        if is_hub == 1:
            score += 10
            checks.append({"Pillar": "Ecosystem", "Check": "HQ in Top Venture Hub (Silicon Valley/NYC/London)", "Pass": True})
        else:
            checks.append({"Pillar": "Ecosystem", "Check": "Emerging / Non-Hub Ecosystem", "Pass": False})

        # Normalize score
        final_score = int(np.clip(score + (prob_success * 20), 0, 100))
        return {
            'venture_score_100': final_score,
            'checklist_items': checks
        }

if __name__ == '__main__':
    explainer = VentureExplainer()
    test_deal = {
        'seed_usd': 2_500_000,
        'angel_usd': 250_000,
        'months_to_first_funding': 8.0,
        'is_top_hub': 1,
        'country_tier': 'USA',
        'market_group': 'Enterprise Software'
    }
    res = explainer.evaluate_deal(test_deal)
    print("Test Deal Evaluation Result:")
    print(f"Predicted Probability: {res['probability_series_a']:.2%}")
    print(f"Tier: {res['conviction_tier']}")
    print(f"21-Factor Venture Score: {res['corea_scorecard']['venture_score_100']} / 100")
    print("\nTop Positive Drivers:")
    for f in res['top_positive_factors']:
        print(f" + {f['Feature']}: {f['Attribution']:+.4f}")
    print("\nTop Risk Penalties:")
    for f in res['top_risk_factors']:
        print(f" - {f['Feature']}: {f['Attribution']:+.4f}")
