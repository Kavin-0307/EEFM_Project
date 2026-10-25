"""
Seed-Stage Feature Engineering Pipeline
=======================================
Transforms raw Crunchbase investment data into point-in-time features strictly
available at the close of the Seed / Angel round.
"""

import os
import re
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "investments_VC.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")

def clean_curr(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip().replace(',', '').replace('$', '').replace(' ', '')
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def build_seed_features(file_path=RAW_DATA_PATH):
    print("Loading raw Crunchbase investments data...")
    df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # Clean early funding amounts
    df['seed_usd'] = df['seed'].apply(clean_curr)
    df['angel_usd'] = df['angel'].apply(clean_curr)
    df['round_a_usd'] = df['round_a'].apply(clean_curr)
    df['venture_usd'] = df['venture'].apply(clean_curr)
    
    # Filter to startups with early-stage capital (Seed or Angel)
    df['early_funding_usd'] = df['seed_usd'] + df['angel_usd']
    seed_df = df[df['early_funding_usd'] > 0].copy()
    print(f"Total Seed/Angel startups extracted: {len(seed_df)}")

    # 1. Temporal signals (Point-in-Time)
    seed_df['founded_at'] = pd.to_datetime(seed_df['founded_at'], errors='coerce')
    seed_df['first_funding_at'] = pd.to_datetime(seed_df['first_funding_at'], errors='coerce')
    
    seed_df['founded_year'] = seed_df['founded_at'].dt.year
    seed_df['first_funding_year'] = seed_df['first_funding_at'].dt.year

    # Days and months from founding to first institutional check
    seed_df['days_to_first_funding'] = (seed_df['first_funding_at'] - seed_df['founded_at']).dt.days
    # Clean anomalies (e.g. negative days or extreme outliers > 10 years)
    seed_df['days_to_first_funding'] = seed_df['days_to_first_funding'].apply(lambda d: d if (0 <= d <= 3650) else np.nan)
    median_days = seed_df['days_to_first_funding'].median()
    seed_df['days_to_first_funding'] = seed_df['days_to_first_funding'].fillna(median_days)
    seed_df['months_to_first_funding'] = seed_df['days_to_first_funding'] / 30.44

    # 2. Financial signals at Seed stage
    seed_df['log_seed_usd'] = np.log1p(seed_df['seed_usd'])
    seed_df['log_angel_usd'] = np.log1p(seed_df['angel_usd'])
    seed_df['has_angel'] = (seed_df['angel_usd'] > 0).astype(int)
    seed_df['has_seed'] = (seed_df['seed_usd'] > 0).astype(int)
    seed_df['seed_angel_ratio'] = np.where(
        seed_df['angel_usd'] > 0,
        seed_df['seed_usd'] / seed_df['angel_usd'],
        seed_df['seed_usd'] / 1.0
    )
    seed_df['log_early_funding_usd'] = np.log1p(seed_df['early_funding_usd'])

    # 3. Geographic Hub & Location Signals
    top_hubs = ['SF Bay Area', 'Silicon Valley', 'New York', 'Boston', 'London', 'Bangalore', 'Los Angeles', 'Seattle', 'Austin']
    seed_df['is_top_hub'] = seed_df['region'].astype(str).str.contains('|'.join(top_hubs), case=False, na=False).astype(int)
    
    # Major countries
    top_countries = ['USA', 'GBR', 'CAN', 'IND', 'DEU', 'FRA', 'ISR', 'ESP']
    seed_df['country_tier'] = seed_df['country_code'].apply(lambda c: c if c in top_countries else 'OTHER')
    
    # 4. Industry Sector Categorization
    seed_df['market'] = seed_df['market'].fillna('Other').astype(str).str.strip()
    top_markets = [
        'Software', 'Biotechnology', 'Mobile', 'E-Commerce', 'Enterprise Software',
        'Curated Web', 'Health Care', 'Social Media', 'Games', 'Advertising',
        'Clean Technology', 'Finance', 'Security', 'Education', 'Analytics'
    ]
    seed_df['market_group'] = seed_df['market'].apply(lambda m: m if m in top_markets else 'Other')

    # 5. Core Targets
    # Target A: Graduation to Series A / Follow-on Venture Round
    seed_df['target_series_a'] = ((seed_df['round_a_usd'] > 0) | (seed_df['venture_usd'] > 0)).astype(int)

    # Target B: Definitive Exit (Acquired or IPO = 1, Closed or Stagnant Mature = 0)
    # Startups founded <= 2011 have had 10+ years to achieve liquidity
    seed_df['is_exit'] = seed_df['status'].str.lower().apply(lambda s: 1 if s in ['acquired', 'ipo'] else (0 if s == 'closed' else (0 if s == 'operating' else np.nan)))

    # Save feature-engineered dataset
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "seed_features.parquet")
    
    # Select clean feature columns
    keep_cols = [
        'name', 'founded_year', 'first_funding_year', 'months_to_first_funding',
        'seed_usd', 'angel_usd', 'early_funding_usd', 'log_seed_usd', 'log_angel_usd',
        'log_early_funding_usd', 'has_seed', 'has_angel', 'seed_angel_ratio',
        'is_top_hub', 'country_tier', 'market_group',
        'target_series_a', 'is_exit', 'status'
    ]
    
    final_df = seed_df[keep_cols].copy()
    final_df.to_parquet(out_path, index=False)
    csv_out = os.path.join(PROCESSED_DIR, "seed_features.csv")
    final_df.to_csv(csv_out, index=False)
    
    print(f"\nFeature engineering complete! Saved {len(final_df)} records.")
    print(f"Target 'target_series_a' distribution: {final_df['target_series_a'].value_counts(normalize=True).to_dict()}")
    print(f"Target 'is_exit' distribution: {final_df['is_exit'].value_counts(normalize=True).to_dict()}")
    return final_df

if __name__ == '__main__':
    build_seed_features()
