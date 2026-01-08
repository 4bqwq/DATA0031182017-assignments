import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from datetime import timezone
from .config import DATA_RAW, DATA_PROCESSED

def parse_metric(value):
    """
    Parse string metrics with K/M suffixes to float/int.
    Examples: '1.7K' -> 1700, '2M' -> 2000000, '125' -> 125, '0' -> 0.
    Returns NaN on failure or missing.
    """
    if pd.isna(value) or value == '':
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
        
    value = str(value).upper().strip()
    
    try:
        if value.endswith('K'):
            return float(value[:-1]) * 1_000
        elif value.endswith('M'):
            return float(value[:-1]) * 1_000_000
        elif value.endswith('B'): # Billion just in case
            return float(value[:-1]) * 1_000_000_000
        else:
            return float(value)
    except ValueError:
        return np.nan

def normalize_text_column(text):
    """
    Normalize text:
    - Lowercase
    - URL -> <URL>
    - @user -> <USER>
    - #tag -> <HASHTAG>
    - Compress whitespace
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # URL (simple regex)
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)
    # Mentions
    text = re.sub(r'@\w+', '<USER>', text)
    # Hashtags
    text = re.sub(r'#\w+', '<HASHTAG>', text)
    # Compress whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(df):
    """
    Generate boolean flags for keywords.
    """
    # Pre-compile regex patterns
    # \b for word boundaries where appropriate, though simple contains might suffice for some
    # covid-19/coronavirus/vaccine/vaccination/wfh/work from home/remote work
    
    # COVID
    covid_pattern = r'covid|coronavirus|sars-cov-2|pandemic'
    df['kw_covid'] = df['text_norm'].str.contains(covid_pattern, regex=True, na=False)
    
    # Vaccine
    vaccine_pattern = r'vaccin|vax|jab|shot|pfizer|moderna|astrazeneca'
    df['kw_vaccine'] = df['text_norm'].str.contains(vaccine_pattern, regex=True, na=False)
    
    # Remote Work
    remote_pattern = r'remote\s*work|work\s*from\s*home|wfh|telework|home\s*office'
    df['kw_remote_work'] = df['text_norm'].str.contains(remote_pattern, regex=True, na=False)
    
    return df

def clean_data():
    print("--- Starting Data Cleaning ---")
    
    # 1. Load Data
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Raw data not found at {DATA_RAW}")
    
    df = pd.read_csv(DATA_RAW)
    raw_count = len(df)
    print(f"Loaded {raw_count} raw rows.")

    # 2. Time Parsing
    # Ensure UTC
    df['publication_time'] = pd.to_datetime(df['publication_time'], utc=True, errors='coerce')
    
    # Drop rows without valid time? Or keep? Usually drop for time-series analysis.
    # We will keep but they won't have year/month. 
    # Actually, let's filter out if critical. For now, just derive.
    df['year'] = df['publication_time'].dt.year
    df['month'] = df['publication_time'].dt.to_period('M')

    # 3. Numeric Parsing
    numeric_cols = ['replies', 'retweets', 'likes', 'views']
    for col in numeric_cols:
        df[f'{col}_n'] = df[col].apply(parse_metric)
    
    # 4. Views Handling
    # views_is_zero: True if parsed value is 0
    df['views_is_zero'] = (df['views_n'] == 0)
    # views_adj: 0 treated as NaN
    df['views_adj'] = df['views_n'].replace(0, np.nan)

    # 5. Text Normalization
    df['text_norm'] = df['text'].apply(normalize_text_column)
    
    # 6. Feature Generation
    # Structural features
    df['has_url'] = df['text_norm'].str.contains('<url>')
    df['has_mention'] = df['text_norm'].str.contains('<user>')
    df['has_hashtag'] = df['text_norm'].str.contains('<hashtag>')
    
    df['char_len'] = df['text'].fillna('').str.len()
    df['word_len'] = df['text'].fillna('').str.split().str.len()
    
    # Keywords
    df = extract_keywords(df)
    
    # Engagement
    # total_eng_n = likes + retweets + replies (only if all present)
    # If any is NaN, result is NaN (default pandas behavior for +)
    df['total_eng_n'] = df['likes_n'] + df['retweets_n'] + df['replies_n']

    # 7. Deduplication
    # Strategy: Group by (author_handle, publication_time, text_norm)
    # Keep the one with highest likes_n, then non-nulls.
    # We can sort by likes_n descending and keep first.
    
    df_sorted = df.sort_values(by=['likes_n', 'total_eng_n'], ascending=[False, False])
    dedup_subset = ['author_handle', 'publication_time', 'text_norm']
    # Handle NaN in subset columns for drop_duplicates? Pandas handles NaN as equal usually.
    # author_handle might be NaN.
    
    df_clean = df_sorted.drop_duplicates(subset=dedup_subset, keep='first').copy()
    dedup_count = len(df_clean)
    
    print(f"Deduplication: {raw_count} -> {dedup_count} rows (removed {raw_count - dedup_count}).")

    # 8. Audit & Save
    audit_stats = {
        'raw_rows': raw_count,
        'clean_rows': dedup_count,
        'missing_rates': {
            col: df_clean[col].isna().mean() for col in ['replies_n', 'retweets_n', 'likes_n', 'views_n', 'views_adj']
        },
        'quantiles': {
            'likes_n': df_clean['likes_n'].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).to_dict(),
            'total_eng_n': df_clean['total_eng_n'].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).to_dict()
        }
    }
    
    # Convert Timestamp/Period to str for JSON serialization if needed, or handle in dump
    # Missing rates are float, quantiles are float. 
    
    # Save Parquet
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_parquet = DATA_PROCESSED / "tweets_clean.parquet"
    # Convert period to timestamp for parquet compatibility usually, or store as string
    # PyArrow supports period but sometimes it's tricky. Converting 'month' to string is safer for broad compat.
    df_clean['month_str'] = df_clean['month'].astype(str)
    
    # Drop the period object column to avoid parquet issues if any
    df_clean = df_clean.drop(columns=['month'])
    
    df_clean.to_parquet(out_parquet, index=False)
    print(f"Saved cleaned data to {out_parquet}")
    
    # Save Audit
    out_audit = DATA_PROCESSED / "data_audit.json"
    with open(out_audit, 'w') as f:
        json.dump(audit_stats, f, indent=2)
    print(f"Saved audit stats to {out_audit}")
    
    print("\n--- Audit Summary ---")
    print(json.dumps(audit_stats, indent=2))

if __name__ == "__main__":
    clean_data()
