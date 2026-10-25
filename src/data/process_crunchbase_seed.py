"""
Crunchbase Seed-Stage Data Preprocessor
=======================================
Processes the public Crunchbase investment dataset (investments_VC.csv)
specifically for early-stage (Seed/Angel) prediction with strict anti-leakage guards.
"""

import os
import sys
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "investments_VC.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed")

def clean_currency(val):
    """Clean string currency formats to float."""
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip().replace(',', '').replace('$', '').replace(' ', '')
    try:
        return float(val_str)
    except ValueError:
        return 0.0

def load_and_preprocess_seed(file_path=RAW_DATA_PATH):
    """
    Filters and formats public Crunchbase data for Seed-stage evaluation.
    """
    if not os.path.exists(file_path):
        # Check alternate names in data/raw
        raw_dir = os.path.dirname(file_path)
        candidates = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.csv')]
        if not candidates:
            raise FileNotFoundError(f"No public CSV dataset found in {raw_dir}. Please place 'investments_VC.csv' or 'startup_investments.csv' into data/raw/.")
        file_path = candidates[0]
        print(f"Using found raw dataset: {file_path}")

    # Load with low_memory=False and handle encoding
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

    print(f"Raw dataset shape: {df.shape}")
    # Strip column whitespace
    df.columns = [c.strip().lower() for c in df.columns]

    # Target variable: IPO or Acquired = 1, Closed = 0
    # Filter out companies with unknown or undefined status
    valid_statuses = ['acquired', 'ipo', 'closed', 'operating']
    df = df[df['status'].str.lower().isin(valid_statuses)].copy()
    
    # In VC literature: Success is defined as Acquired or IPO (Corea [2], Gautam [4], Thirupathi [10])
    df['target_success'] = df['status'].str.lower().apply(lambda s: 1 if s in ['acquired', 'ipo'] else (0 if s == 'closed' else np.nan))
    
    # Parse seed and angel funding
    if 'seed' in df.columns:
        df['seed_usd'] = df['seed'].apply(clean_currency)
    else:
        df['seed_usd'] = 0.0
        
    if 'angel' in df.columns:
        df['angel_usd'] = df['angel'].apply(clean_currency)
    else:
        df['angel_usd'] = 0.0
        
    df['early_funding_usd'] = df['seed_usd'] + df['angel_usd']

    # Filter to startups with early-stage (Seed / Angel) activity
    seed_df = df[df['early_funding_usd'] > 0].copy()
    print(f"Startups with Seed/Angel funding: {len(seed_df)}")

    # Time signals: Founding date to first funding date
    seed_df['founded_at'] = pd.to_datetime(seed_df['founded_at'], errors='coerce')
    seed_df['first_funding_at'] = pd.to_datetime(seed_df['first_funding_at'], errors='coerce')
    
    seed_df['founded_year'] = seed_df['founded_at'].dt.year
    seed_df['first_funding_year'] = seed_df['first_funding_at'].dt.year
    
    # Months from founding to first round (speed of initial traction)
    seed_df['months_to_first_funding'] = (seed_df['first_funding_at'] - seed_df['founded_at']).dt.days / 30.44
    seed_df['months_to_first_funding'] = seed_df['months_to_first_funding'].apply(lambda x: x if (0 <= x <= 120) else np.nan)
    
    # Hub detection
    top_hubs = ['SF Bay Area', 'Silicon Valley', 'New York', 'Boston', 'London', 'Bangalore', 'Los Angeles']
    seed_df['is_top_hub'] = seed_df['region'].astype(str).str.contains('|'.join(top_hubs), case=False, na=False).astype(int)
    
    # Market clean
    seed_df['market'] = seed_df['market'].fillna('Other').str.strip()

    # Drop target-leakage columns (Total funding, future Series rounds, total rounds)
    leakage_cols = [c for c in seed_df.columns if any(k in c for k in ['total', 'round_a', 'round_b', 'round_c', 'round_d', 'round_e', 'round_f', 'funding_rounds', 'last_funding_at', 'venture'])]
    print(f"Dropped {len(leakage_cols)} potential leakage columns to enforce strict Seed-stage boundary.")
    
    # Only keep completed outcomes (Exit vs Closed) for supervised benchmark or retain operating for semi-supervised
    benchmark_df = seed_df.dropna(subset=['target_success']).copy()
    benchmark_df['target_success'] = benchmark_df['target_success'].astype(int)
    
    print(f"Final Seed-stage benchmark set size: {len(benchmark_df)}")
    print(f"Exit (1s): {benchmark_df['target_success'].sum()} ({benchmark_df['target_success'].mean():.2%}) | Closed (0s): {(benchmark_df['target_success'] == 0).sum()}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_file = os.path.join(PROCESSED_DIR, "seed_crunchbase_cleaned.csv")
    benchmark_df.to_csv(out_file, index=False)
    print(f"Saved processed Seed dataset to: {out_file}")
    return benchmark_df

if __name__ == '__main__':
    load_and_preprocess_seed()
