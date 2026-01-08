import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .config import DATA_PROCESSED, FIGURES_DIR, TABLES_DIR

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def gini_coefficient(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    x = np.array(x, dtype=np.float64)
    # Filter nan
    x = x[~np.isnan(x)]
    if len(x) == 0: return 0.0
    
    n = len(x)
    x = np.sort(x)
    
    # Gini = (2 * sum(i * x_i) - (n + 1) * sum(x_i)) / (n * sum(x_i))
    # Using the simpler formula based on Lorenz curve area
    
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * x).sum() / (n * np.sum(x))

def run_eda_analysis():
    print("--- Starting EDA ---")
    input_path = DATA_PROCESSED / "tweets_clean.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned data not found at {input_path}")
    
    df = pd.read_parquet(input_path)
    
    # Convert month_str back to datetime for plotting (start of month)
    df['date_month'] = pd.to_datetime(df['month_str'])
    
    # --- 1. Time Distribution ---
    print("Generating Time Distribution plots...")
    monthly_counts = df.groupby('date_month').size()
    
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_counts.index, monthly_counts.values, marker='o', linestyle='-')
    plt.title('Monthly Tweet Volume')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "time_distribution.png")
    plt.close()

    # Keyword Trends
    # Calculate proportion of tweets containing keywords per month
    kw_trends = df.groupby('date_month')[['kw_vaccine', 'kw_remote_work']].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(kw_trends.index, kw_trends['kw_vaccine'], label='Vaccine', marker='.')
    plt.plot(kw_trends.index, kw_trends['kw_remote_work'], label='Remote Work', marker='.')
    plt.title('Monthly Proportion of Keywords')
    plt.xlabel('Date')
    plt.ylabel('Proportion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "keyword_trends.png")
    plt.close()

    # --- 2. Interaction Distribution ---
    print("Generating Interaction plots...")
    # Histogram of log1p likes
    likes_log = np.log1p(df['likes_n'].dropna())
    
    plt.figure(figsize=(8, 6))
    plt.hist(likes_log, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Likes (Log1p)')
    plt.xlabel('log(Likes + 1)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "likes_hist.png")
    plt.close()

    # Boxplot by Year
    # Filter out NaNs for this plot
    data_by_year = []
    years = sorted(df['year'].dropna().unique())
    labels = []
    
    for y in years:
        subset = df[df['year'] == y]['likes_n'].dropna()
        if len(subset) > 0:
            data_by_year.append(np.log1p(subset)) # Using log scale for visibility
            labels.append(int(y))
    
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_by_year, labels=labels, patch_artist=True)
    plt.title('Likes Distribution by Year (Log Scale)')
    plt.xlabel('Year')
    plt.ylabel('log(Likes + 1)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "likes_boxplot.png")
    plt.close()

    # --- 3. Missingness & Quality ---
    print("Generating Quality plots...")
    metrics = ['replies_n', 'retweets_n', 'likes_n', 'views_n']
    missing_rates = [df[m].isna().mean() for m in metrics]
    
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, missing_rates, color='salmon')
    plt.title('Missing Rates by Metric')
    plt.ylabel('Missing Proportion')
    plt.ylim(0, 1.0)
    for i, v in enumerate(missing_rates):
        plt.text(i, v + 0.01, f"{v:.2%}", ha='center')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "missing_rates.png")
    plt.close()

    # Views == 0 Trend
    views_zero_trend = df.groupby('date_month')['views_is_zero'].mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(views_zero_trend.index, views_zero_trend.values, marker='o', color='orange')
    plt.title('Proportion of Tweets with 0 Views over Time')
    plt.xlabel('Date')
    plt.ylabel('Proportion (Views == 0)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "views_zero_trend.png")
    plt.close()

    # --- 4. Author Concentration (Lorenz Curve) ---
    print("Generating Author Concentration plots...")
    # Agg by author
    author_likes = df.groupby('author_handle')['likes_n'].sum().fillna(0).sort_values(ascending=False)
    total_likes = author_likes.sum()
    
    # Cumulative stats
    cum_authors = np.arange(1, len(author_likes) + 1) / len(author_likes)
    cum_likes = author_likes.cumsum() / total_likes
    
    # Perfect equality line
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Equality')
    plt.plot(cum_authors, cum_likes, label='Actual Distribution', linewidth=2)
    plt.title('Lorenz Curve of Author Likes')
    plt.xlabel('Cumulative Proportion of Authors')
    plt.ylabel('Cumulative Proportion of Likes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "author_lorenz.png")
    plt.close()
    
    # Calculate Gini
    gini_val = gini_coefficient(author_likes.values)
    print(f"Author Gini Coefficient: {gini_val:.4f}")

    # --- 5. Key Numbers Table ---
    print("Generating Key Numbers Table...")
    
    # Top 1% Authors share
    n_top_1pct = int(len(author_likes) * 0.01)
    if n_top_1pct < 1: n_top_1pct = 1
    top_1pct_share = author_likes.iloc[:n_top_1pct].sum() / total_likes
    
    key_stats = {
        'total_tweets': len(df),
        'unique_authors': df['author_handle'].nunique(),
        'avg_likes': df['likes_n'].mean(),
        'median_likes': df['likes_n'].median(),
        'missing_views_rate': df['views_n'].isna().mean(),
        'zero_views_rate': df['views_is_zero'].mean(),
        'author_gini': gini_val,
        'top_1pct_author_share': top_1pct_share,
        'kw_covid_pct': df['kw_covid'].mean(),
        'kw_vaccine_pct': df['kw_vaccine'].mean(),
        'kw_remote_work_pct': df['kw_remote_work'].mean()
    }
    
    stats_df = pd.DataFrame([key_stats]).T.reset_index()
    stats_df.columns = ['Metric', 'Value']
    stats_path = TABLES_DIR / "eda_key_numbers.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved key numbers to {stats_path}")

    print("EDA Complete.")

if __name__ == "__main__":
    run_eda_analysis()
