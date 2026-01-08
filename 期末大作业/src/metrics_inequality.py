import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from .config import DATA_PROCESSED, TABLES_DIR, FIGURES_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)

def calculate_inequality_metrics(likes_array):
    """
    Calculate Gini, Top Shares, and HHI for an array of likes.
    """
    likes = np.array(likes_array, dtype=np.float64)
    likes = likes[likes >= 0] # Filter negatives if any
    total_likes = np.sum(likes)
    n = len(likes)
    
    if n == 0 or total_likes == 0:
        return {
            'gini': np.nan,
            'top1_share': np.nan,
            'top5_share': np.nan,
            'hhi': np.nan,
            'effective_authors': np.nan,
            'n_authors': n,
            'total_likes': total_likes
        }
        
    # Sort
    likes_sorted = np.sort(likes)
    
    # Gini
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * likes_sorted).sum() / (n * total_likes)
    
    # Top Shares
    likes_desc = likes_sorted[::-1]
    
    k1 = int(np.ceil(0.01 * n))
    top1_share = likes_desc[:k1].sum() / total_likes
    
    k5 = int(np.ceil(0.05 * n))
    top5_share = likes_desc[:k5].sum() / total_likes
    
    # HHI
    shares = likes / total_likes
    hhi = np.sum(shares ** 2)
    effective_authors = 1.0 / hhi if hhi > 0 else 0
    
    return {
        'gini': gini,
        'top1_share': top1_share,
        'top5_share': top5_share,
        'hhi': hhi,
        'effective_authors': effective_authors,
        'n_authors': n,
        'total_likes': total_likes
    }

def run_inequality_analysis():
    print("--- Starting Inequality Analysis ---")
    
    input_path = DATA_PROCESSED / "tweets_labeled.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Data not found at {input_path}")
        
    df = pd.read_parquet(input_path)
    
    # Ensure month
    if 'month_str' not in df.columns:
         df['month_str'] = pd.to_datetime(df['publication_time']).dt.to_period('M').astype(str)
    
    # Pre-aggregate: Author Likes per (Topic, Month)
    # We need Author Likes per Topic (Overall) and Author Likes per (Topic, Month)
    
    # --- 1. Topic-Month Level ---
    print("Calculating Topic-Month metrics...")
    tm_metrics = []
    
    # Group by Topic, Month
    grouped = df.groupby(['topic_id', 'topic_label', 'high_level_category', 'month_str'])
    
    for (tid, tlabel, cat, m), group in tqdm(grouped, desc="Topic-Months"):
        # Agg likes by author
        author_likes = group.groupby('author_handle')['likes_n'].sum().fillna(0).values
        
        metrics = calculate_inequality_metrics(author_likes)
        metrics.update({
            'topic_id': tid,
            'topic_label': tlabel,
            'high_level_category': cat,
            'month_str': m
        })
        tm_metrics.append(metrics)
        
    tm_df = pd.DataFrame(tm_metrics)
    # Mark unstable
    tm_df['is_stable'] = tm_df['n_authors'] >= 10
    
    tm_out = TABLES_DIR / "attention_inequality_topic_month.csv"
    tm_df.to_csv(tm_out, index=False)
    print(f"Saved monthly inequality to {tm_out}")
    
    # --- 2. Topic Overall Level ---
    print("Calculating Topic Overall metrics...")
    to_metrics = []
    
    grouped_overall = df.groupby(['topic_id', 'topic_label', 'high_level_category'])
    
    for (tid, tlabel, cat), group in tqdm(grouped_overall, desc="Topics Overall"):
        author_likes = group.groupby('author_handle')['likes_n'].sum().fillna(0).values
        metrics = calculate_inequality_metrics(author_likes)
        metrics.update({
            'topic_id': tid,
            'topic_label': tlabel,
            'high_level_category': cat
        })
        to_metrics.append(metrics)
        
    to_df = pd.DataFrame(to_metrics)
    to_out = TABLES_DIR / "attention_inequality_topic_overall.csv"
    to_df.to_csv(to_out, index=False)
    print(f"Saved overall inequality to {to_out}")
    
    # --- 3. Visualizations ---
    
    # A. Time Series (Aggregated by Category)
    # Filter stable months only for plotting? Usually yes.
    plot_df = tm_df[tm_df['is_stable']].copy()
    plot_df['date'] = pd.to_datetime(plot_df['month_str'])
    
    # Group by Category and Month -> Mean Gini
    cat_time = plot_df.groupby(['high_level_category', 'date'])['gini'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    for cat in cat_time['high_level_category'].unique():
        subset = cat_time[cat_time['high_level_category'] == cat]
        plt.plot(subset['date'], subset['gini'], marker='o', label=cat)
        
    plt.title('Average Attention Inequality (Gini) Over Time')
    plt.xlabel('Date')
    plt.ylabel('Gini Coefficient')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ineq_over_time_vaccine_vs_remote.png")
    plt.close()
    
    # B. Overall Gini Bar Chart
    to_df_sorted = to_df.sort_values('gini', ascending=True)
    
    plt.figure(figsize=(10, 8))
    # Color bars by category
    colors = to_df_sorted['high_level_category'].map({
        'vaccine': 'salmon', 
        'remote_work': 'skyblue', 
        'other': 'gray'
    }).fillna('gray')
    
    bars = plt.barh(to_df_sorted['topic_label'], to_df_sorted['gini'], color=colors)
    plt.title('Overall Attention Inequality (Gini) by Topic')
    plt.xlabel('Gini Coefficient')
    
    # Legend manually
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='salmon', lw=4),
        Line2D([0], [0], color='skyblue', lw=4),
        Line2D([0], [0], color='gray', lw=4)
    ]
    plt.legend(custom_lines, ['Vaccine', 'Remote Work', 'Other'])
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "gini_by_topic.png")
    plt.close()
    
    # --- 4. Statistical Test (Permutation) ---
    print("Running Permutation Test...")
    
    # Data: Overall Gini per Topic
    # We want to compare Mean(Gini | Vaccine) vs Mean(Gini | Remote Work)
    # Unit of observation: Topic
    
    subset = to_df[to_df['high_level_category'].isin(['vaccine', 'remote_work'])].copy()
    
    group_v = subset[subset['high_level_category'] == 'vaccine']['gini'].values
    group_r = subset[subset['high_level_category'] == 'remote_work']['gini'].values
    
    if len(group_v) < 2 or len(group_r) < 2:
        print("Not enough topics per category for test.")
        return

    obs_diff = np.mean(group_v) - np.mean(group_r)
    
    # Permutation
    n_perms = 10000
    pool = subset['gini'].values
    n_v = len(group_v)
    
    diffs = []
    for _ in range(n_perms):
        np.random.shuffle(pool)
        perm_v = pool[:n_v]
        perm_r = pool[n_v:]
        diffs.append(np.mean(perm_v) - np.mean(perm_r))
        
    diffs = np.array(diffs)
    p_value = np.mean(np.abs(diffs) >= np.abs(obs_diff))
    
    print(f"\nPermutation Test Results (Topic Level, N={len(subset)} topics):")
    print(f"Mean Gini (Vaccine): {np.mean(group_v):.4f}")
    print(f"Mean Gini (Remote Work): {np.mean(group_r):.4f}")
    print(f"Observed Difference: {obs_diff:.4f}")
    print(f"P-value (Two-sided): {p_value:.4f}")
    
    # Save test result
    with open(TABLES_DIR / "inequality_test_result.txt", "w") as f:
        f.write("Permutation Test: Difference in Mean Gini (Vaccine vs Remote Work)\n")
        f.write(f"N_topics_vaccine: {len(group_v)}\n")
        f.write(f"N_topics_remote: {len(group_r)}\n")
        f.write(f"Mean Gini (Vaccine): {np.mean(group_v):.4f}\n")
        f.write(f"Mean Gini (Remote Work): {np.mean(group_r):.4f}\n")
        f.write(f"Observed Diff: {obs_diff:.4f}\n")
        f.write(f"P-value (10k perms): {p_value:.4f}\n")

if __name__ == "__main__":
    run_inequality_analysis()
