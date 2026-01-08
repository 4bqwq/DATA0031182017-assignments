import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path
from .config import DATA_PROCESSED, TABLES_DIR

def format_summary(model_res, model_name):
    """
    Extract coefficients table from statsmodels result.
    """
    # Extract robust results
    summary = model_res.summary2().tables[1]
    summary['model'] = model_name
    summary = summary.reset_index().rename(columns={'index': 'term'})
    return summary

def generate_interpretation(res_eng, res_ineq):
    """
    Generate natural language interpretation based on model results.
    """
    lines = ["# Statistical Modeling Results & Interpretation", ""]
    
    lines.append("## Disclaimer")
    lines.append("These models represent associative relationships (correlations) controlled for covariates, not causal effects.\n")
    
    # --- Model 1: Engagement ---
    lines.append("## Model 1: Tweet Engagement (Log Likes)")
    lines.append(f"- **R-squared**: {res_eng.rsquared:.3f}")
    
    # Check key vars
    pvals = res_eng.pvalues
    coefs = res_eng.params
    
    # URL effect
    if 'has_url[T.True]' in pvals and pvals['has_url[T.True]'] < 0.05:
        direction = "lower" if coefs['has_url[T.True]'] < 0 else "higher"
        lines.append(f"- Tweets containing URLs are associated with significantly **{direction}** engagement (coef={coefs['has_url[T.True]']:.2f}, p<.05).")
    
    # Hashtag effect
    if 'has_hashtag[T.True]' in pvals and pvals['has_hashtag[T.True]'] < 0.05:
        direction = "lower" if coefs['has_hashtag[T.True]'] < 0 else "higher"
        lines.append(f"- Using hashtags is linked to **{direction}** engagement (coef={coefs['has_hashtag[T.True]']:.2f}).")
        
    # Mention effect
    if 'has_mention[T.True]' in pvals and pvals['has_mention[T.True]'] < 0.05:
        direction = "lower" if coefs['has_mention[T.True]'] < 0 else "higher"
        lines.append(f"- Mentions (@user) are associated with **{direction}** likes.")

    # Character length
    if 'char_len' in pvals and pvals['char_len'] < 0.05:
         direction = "increases" if coefs['char_len'] > 0 else "decreases"
         lines.append(f"- Longer tweets (by character count) generally see **{direction}** engagement.")

    lines.append("")
    
    # --- Model 2: Inequality ---
    lines.append("## Model 2: Attention Inequality (Gini) Over Time")
    lines.append(f"- **R-squared**: {res_ineq.rsquared:.3f}")
    
    # Category effect
    coefs_2 = res_ineq.params
    pvals_2 = res_ineq.pvalues
    
    # Look for the category term, ignoring interaction first
    term_cat = [c for c in coefs_2.index if 'high_level_category' in c and ':' not in c]
    term_inter = [c for c in coefs_2.index if 'high_level_category' in c and ':' in c]
    
    if term_cat:
        t = term_cat[0]
        if pvals_2[t] < 0.05:
            cat_name = t.split('[T.')[1].split(']')[0]
            comp = "higher" if coefs_2[t] > 0 else "lower"
            lines.append(f"- The '{cat_name}' category shows significantly **{comp}** baseline inequality compared to the reference category (coef={coefs_2[t]:.2f}).")
            
    if term_inter:
        t = term_inter[0]
        if pvals_2[t] < 0.05:
            lines.append(f"- There is a significant interaction effect over time ({t}), suggesting the gap in inequality between categories is changing.")
        else:
            lines.append("- No significant interaction with time was found; the inequality gap between categories remains relatively stable over time.")
            
    # Sample size control
    if 'np.log(n_authors)' in pvals_2 and pvals_2['np.log(n_authors)'] < 0.05:
         lines.append("- Topics with larger author pools (log n_authors) tend to have different inequality structures (control variable significant).")

    return "\n".join(lines)

def run_statistical_modeling():
    print("--- Starting Statistical Modeling ---")
    
    # --- Model 1: Engagement ---
    print("Fitting Model 1: Engagement OLS...")
    df_tweets = pd.read_parquet(DATA_PROCESSED / "tweets_labeled.parquet")
    
    df_tweets['log_likes'] = np.log1p(df_tweets['likes_n'])
    
    # Explicitly select cols and dropna to align groups
    cols_used = ['log_likes', 'topic_label', 'year', 'char_len', 'has_url', 'has_hashtag', 'has_mention', 'author_handle']
    df_model_1 = df_tweets[cols_used].dropna().copy()
    
    # Create groups from author_handle
    groups = pd.factorize(df_model_1['author_handle'])[0]
    
    formula_1 = "log_likes ~ C(topic_label) + C(year) + char_len + has_url + has_hashtag + has_mention"
    
    model_1 = smf.ols(formula=formula_1, data=df_model_1)
    res_1 = model_1.fit(cov_type='cluster', cov_kwds={'groups': groups})
    
    # Save results
    summ_1 = format_summary(res_1, "OLS_Engagement_LogLikes")
    path_1 = TABLES_DIR / "model_engagement_ols.csv"
    summ_1.to_csv(path_1, index=False)
    print(f"Saved Model 1 results to {path_1}")
    
    # --- Model 2: Inequality ---
    print("Fitting Model 2: Inequality OLS...")
    df_ineq = pd.read_csv(TABLES_DIR / "attention_inequality_topic_month.csv")
    
    # Filter
    df_ineq = df_ineq[df_ineq['is_stable'] == True].copy()
    
    # Month Index
    df_ineq['date'] = pd.to_datetime(df_ineq['month_str'])
    min_date = df_ineq['date'].min()
    df_ineq['month_index'] = ((df_ineq['date'].dt.year - min_date.year) * 12 + 
                              (df_ineq['date'].dt.month - min_date.month))
    
    formula_2 = "gini ~ C(high_level_category) * month_index + np.log(n_authors)"
    
    model_2 = smf.wls(formula=formula_2, data=df_ineq, weights=df_ineq['n_authors'])
    res_2 = model_2.fit()
    
    # Save results
    summ_2 = format_summary(res_2, "WLS_Inequality_Gini")
    path_2 = TABLES_DIR / "model_inequality_time.csv"
    summ_2.to_csv(path_2, index=False)
    print(f"Saved Model 2 results to {path_2}")
    
    # --- Interpretation ---
    text = generate_interpretation(res_1, res_2)
    path_text = TABLES_DIR / "model_interpretation.txt"
    with open(path_text, "w") as f:
        f.write(text)
    print(f"Saved interpretation to {path_text}")
    print("\n--- Interpretation Preview ---")
    print(text)

if __name__ == "__main__":
    run_statistical_modeling()