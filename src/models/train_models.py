"""
Model Training & Evaluation Pipeline
=====================================
Trains baseline and gradient-boosted tree ensembles (XGBoost, LightGBM)
under strict chronological temporal splitting and explicit class-imbalance weighting.
Evaluates using investor-oriented metrics (PR-AUC, Precision@Top-10%, ROC-AUC).
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report
)
import xgboost as xgb
import lightgbm as lgb

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "seed_features.parquet")
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")

def precision_at_k(y_true, y_prob, k_percent=0.10):
    """Calculates precision among top K% highest confidence deals (VC Deal Screening)."""
    n = len(y_prob)
    k = max(1, int(n * k_percent))
    top_k_indices = np.argsort(y_prob)[-k:]
    return np.mean(y_true.iloc[top_k_indices] == 1)

def train_and_evaluate(target_col='target_series_a'):
    print(f"\n==================================================")
    print(f"Training Seed-Stage Models for Target: '{target_col}'")
    print(f"==================================================")
    
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    # Filter to records where target is non-null
    df = df.dropna(subset=[target_col, 'first_funding_year']).copy()
    df[target_col] = df[target_col].astype(int)

    # 1. Temporal Out-of-Time Split (Anti-Leakage)
    # Train: Startups first funded <= 2011; Test: Startups first funded >= 2012
    train_mask = df['first_funding_year'] <= 2011
    test_mask = df['first_funding_year'] >= 2012
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    
    print(f"Temporal Split: Train (<=2011): {len(train_df)} | Test (>=2012): {len(test_df)}")
    print(f"Train positive rate: {train_df[target_col].mean():.2%} | Test positive rate: {test_df[target_col].mean():.2%}")

    feature_cols_num = [
        'months_to_first_funding', 'seed_usd', 'angel_usd',
        'log_seed_usd', 'log_angel_usd', 'log_early_funding_usd',
        'has_seed', 'has_angel', 'seed_angel_ratio', 'is_top_hub'
    ]
    feature_cols_cat = ['country_tier', 'market_group']
    
    X_train = train_df[feature_cols_num + feature_cols_cat]
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols_num + feature_cols_cat]
    y_test = test_df[target_col]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_cols_num),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), feature_cols_cat)
        ]
    )

    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # Feature names
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(feature_cols_cat).tolist()
    all_feature_names = feature_cols_num + cat_names

    # Compute positive class weight for imbalanced learning
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos = n_neg / max(1, n_pos)
    print(f"Calculated class-imbalance weight (scale_pos_weight): {scale_pos:.2f}")

    models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=42, verbose=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, scale_pos_weight=scale_pos, eval_metric='logloss', random_state=42)
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_trans, y_train)
        y_prob = model.predict_proba(X_test_trans)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        roc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        prec_10 = precision_at_k(y_test, y_prob, k_percent=0.10)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "PR-AUC (Primary)": round(pr_auc, 4),
            "Precision@Top-10%": round(prec_10, 4),
            "ROC-AUC": round(roc, 4),
            "F1-Score": round(f1, 4)
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results).sort_values(by="PR-AUC (Primary)", ascending=False)
    print("\nBenchmark Evaluation Results on Out-of-Time Test Set:")
    print(results_df.to_string(index=False))

    # Save best model (LightGBM / XGBoost) & preprocessor
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    
    bundle = {
        'model_name': best_model_name,
        'model': best_model,
        'preprocessor': preprocessor,
        'feature_names': all_feature_names,
        'metrics': results_df.to_dict(orient='records'),
        'target_col': target_col
    }
    
    save_path = os.path.join(ARTIFACTS_DIR, f"best_seed_model_{target_col}.joblib")
    joblib.dump(bundle, save_path)
    print(f"\nBest model ({best_model_name}) saved successfully to: {save_path}")
    return results_df

if __name__ == '__main__':
    train_and_evaluate('target_series_a')
