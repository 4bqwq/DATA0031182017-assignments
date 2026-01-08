import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from .config import DATA_PROCESSED, TABLES_DIR, FIGURES_DIR

def calculate_lifecycle_metrics(monthly_df, topic_info):
    """
    Calculate lifecycle metrics for each topic series.
    monthly_df: DataFrame with 'date_month', 'topic_label', 'share'
    """
    metrics = []
    
    # Ensure sorted by date
    monthly_df = monthly_df.sort_values('date_month')
    
    for label, group in monthly_df.groupby('topic_label'):
        # Fill missing months with 0 share for continuity
        # Create full date range
        min_date = monthly_df['date_month'].min()
        max_date = monthly_df['date_month'].max()
        all_months = pd.date_range(min_date, max_date, freq='MS')
        
        group = group.set_index('date_month').reindex(all_months, fill_value=0).reset_index()
        group = group.rename(columns={'index': 'date_month'})
        # 'share' might be NaN after reindex, fill with 0
        group['share'] = group['share'].fillna(0)
        
        shares = group['share'].values
        dates = group['date_month'].values
        
        # 1. Peak
        peak_idx = np.argmax(shares)
        peak_share = shares[peak_idx]
        peak_date = dates[peak_idx]
        
        # 2. Rise Time (Months to reach 0.8 * peak from first non-zero)
        # Threshold
        thresh_rise = 0.8 * peak_share
        # Find first non-zero (or very small threshold)
        non_zero_indices = np.where(shares > 0)[0]
        if len(non_zero_indices) == 0:
            # Should not happen if topic exists
            continue
            
        start_idx = non_zero_indices[0]
        
        # Find first index >= thresh_rise
        # search only from start_idx to peak_idx (inclusive)
        # Argmax on boolean gives first True
        rise_candidates = np.where(shares[:peak_idx+1] >= thresh_rise)[0]
        if len(rise_candidates) > 0:
            reach_idx = rise_candidates[0]
            rise_time_months = reach_idx - start_idx
        else:
            rise_time_months = 0 # Peak is at start
            
        # 3. Half Life (Months from peak to drop below 0.5 * peak)
        thresh_decay = 0.5 * peak_share
        # Search after peak
        half_life_months = np.nan
        if peak_idx < len(shares) - 1:
            decay_candidates = np.where(shares[peak_idx+1:] < thresh_decay)[0]
            if len(decay_candidates) > 0:
                # decay_candidates indices are relative to peak_idx+1
                first_decay_offset = decay_candidates[0]
                half_life_months = (first_decay_offset + 1)
        
        # 4. Revival Count (Months after peak > 0.7 * peak)
        revival_count = 0
        if peak_idx < len(shares) - 1:
            thresh_revival = 0.7 * peak_share
            # We strictly count months. 
            revival_months = np.sum(shares[peak_idx+1:] > thresh_revival)
            revival_count = revival_months
            
        # Get category info
        cat = topic_info.get(label, "other")
        
        metrics.append({
            'topic_label': label,
            'high_level_category': cat,
            'peak_month': peak_date,
            'peak_share': peak_share,
            'rise_time_months': rise_time_months,
            'half_life_months': half_life_months,
            'revival_count': revival_count
        })
        
    return pd.DataFrame(metrics)

def plot_heatmap(pivot_df):
    """
    X: Month, Y: Topic, Color: Share
    """
    print("Generating Heatmap...")
    plt.figure(figsize=(14, 8))
    
    # Prepare data
    data = pivot_df.values
    x_dates = pivot_df.columns
    y_labels = pivot_df.index
    
    # Plot
    # Using pcolormesh needs coordinates for edges. 
    # imshow is easier for grid data.
    plt.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Ticks
    plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)
    
    # X-ticks formatted
    # Show every 3rd month to avoid crowding
    x_indices = np.arange(len(x_dates))
    plt.xticks(ticks=x_indices[::3], labels=[d.strftime('%Y-%m') for d in x_dates][::3], rotation=45)
    
    plt.colorbar(label='Topic Share')
    plt.title('Topic Lifecycle Heatmap (Share of Monthly Tweets)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "topic_share_heatmap.png")
    plt.close()

def plot_streamgraph(pivot_df):
    """
    Stacked area chart (Streamgraph-like).
    """
    print("Generating Streamgraph...")
    plt.figure(figsize=(14, 8))
    
    x = pivot_df.columns
    y = pivot_df.values
    labels = pivot_df.index
    
    # Matplotlib stackplot
    plt.stackplot(x, y, labels=labels, alpha=0.8)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Topics')
    plt.title('Topic Evolution (Stacked Share)')
    plt.xlabel('Month')
    plt.ylabel('Share of Discussion')
    plt.xlim(x.min(), x.max())
    plt.ylim(0, 1.0) # shares sum to 1
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "topic_stream.png")
    plt.close()

def run_lifecycle_analysis():
    print("--- Starting Lifecycle Analysis ---")
    
    input_path = DATA_PROCESSED / "tweets_labeled.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Labeled data not found at {input_path}")
    
    df = pd.read_parquet(input_path)
    
    # Ensure time
    if 'month_str' not in df.columns:
        # Fallback if missing
        df['month_str'] = pd.to_datetime(df['publication_time']).dt.to_period('M').astype(str)
    
    # 1. Aggregate Monthly
    # Convert month_str to datetime for sorting
    df['date_month'] = pd.to_datetime(df['month_str'])
    
    # Count per month per topic
    monthly_counts = df.groupby(['date_month', 'topic_label']).size().reset_index(name='count')
    
    # Total per month
    month_totals = df.groupby('date_month').size().reset_index(name='total')
    
    # Merge
    merged = monthly_counts.merge(month_totals, on='date_month')
    merged['share'] = merged['count'] / merged['total']
    
    # Save intermediate
    monthly_out = DATA_PROCESSED / "topic_monthly.csv"
    merged.to_csv(monthly_out, index=False)
    print(f"Saved monthly stats to {monthly_out}")
    
    # 2. Calculate Metrics
    # Map topic label to category
    topic_cat_map = df[['topic_label', 'high_level_category']].drop_duplicates().set_index('topic_label')['high_level_category'].to_dict()
    
    lifecycle_df = calculate_lifecycle_metrics(merged, topic_cat_map)
    
    # Save Metrics
    metrics_out = TABLES_DIR / "topic_lifecycle.csv"
    lifecycle_df.to_csv(metrics_out, index=False)
    print(f"Saved lifecycle metrics to {metrics_out}")
    
    # Summary by Category
    # Group by category, calc mean/median
    # Columns to summarize: rise_time, half_life, revival, peak_share
    # Note: peak_month is datetime, can't mean easily (could do median date).
    
    cat_summary = lifecycle_df.groupby('high_level_category')[['rise_time_months', 'half_life_months', 'revival_count', 'peak_share']].agg(['mean', 'median'])
    cat_out = TABLES_DIR / "topic_lifecycle_by_category.csv"
    cat_summary.to_csv(cat_out)
    print(f"Saved category summary to {cat_out}")
    
    # 3. Visualizations
    # Pivot for plotting: Index=Label, Col=Date, Val=Share
    pivot_df = merged.pivot(index='topic_label', columns='date_month', values='share').fillna(0)
    
    # Sort topics by total share or peak time for better visuals?
    # Sorting by Total Volume (sum of shares or counts) often looks best in stackplot
    # Actually, stackplot order matters. Let's sort by overall volume.
    topic_vols = merged.groupby('topic_label')['count'].sum()
    sorted_topics = topic_vols.sort_values(ascending=False).index
    pivot_df = pivot_df.loc[sorted_topics]
    
    plot_heatmap(pivot_df)
    plot_streamgraph(pivot_df)
    
    print("Lifecycle Analysis Complete.")

if __name__ == "__main__":
    run_lifecycle_analysis()
