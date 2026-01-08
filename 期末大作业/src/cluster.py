import os
import json
import time
import hashlib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from .config import DATA_PROCESSED, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, RANDOM_SEED

# Load environment variables
load_dotenv()

CACHE_PATH = DATA_PROCESSED / "embed_cache.parquet"
API_KEY = os.getenv("API_KEY")
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "text-embedding-v3"

def get_text_hash(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_cache():
    if CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return pd.DataFrame(columns=['text_hash', 'embedding'])

def save_cache(new_cache_df):
    if CACHE_PATH.exists():
        existing = pd.read_parquet(CACHE_PATH)
        # Append only new
        combined = pd.concat([existing, new_cache_df]).drop_duplicates(subset='text_hash', keep='last')
        combined.to_parquet(CACHE_PATH, index=False)
    else:
        new_cache_df.to_parquet(CACHE_PATH, index=False)

def call_embedding_api(texts, batch_size=16, retries=3):
    """
    Call the API in batches.
    Returns a list of embeddings (lists of floats).
    """
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
        batch = texts[i:i+batch_size]
        payload = {
            "model": MODEL_NAME,
            "input": batch
        }
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(retries):
            try:
                response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                # Ensure order is preserved
                batch_embeddings = [x['embedding'] for x in data['data']]
                embeddings.extend(batch_embeddings)
                break
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Failed batch {i} after {retries} attempts: {e}")
                    # Fill with None or zeros to keep alignment? 
                    # Better to raise error or skip. 
                    # For this pipeline, we'll append None and filter later to avoid crashing everything.
                    embeddings.extend([None] * len(batch))
                else:
                    time.sleep(2 * (attempt + 1))
                    
    return embeddings

def get_embeddings(df):
    """
    Main embedding orchestration.
    """
    print("--- Starting Embedding Process ---")
    
    # 1. Prepare Data
    df['text_hash'] = df['text_norm'].apply(get_text_hash)
    unique_texts = df[['text_norm', 'text_hash']].drop_duplicates()
    
    # 2. Check Cache
    cache = load_cache()
    cached_hashes = set(cache['text_hash'])
    
    missing_mask = ~unique_texts['text_hash'].isin(cached_hashes)
    missing_texts = unique_texts[missing_mask]
    
    print(f"Total unique texts: {len(unique_texts)}")
    print(f"Cached: {len(cached_hashes)}")
    print(f"To embed: {len(missing_texts)}")
    
    # 3. Fetch Missing
    if not missing_texts.empty:
        texts_to_fetch = missing_texts['text_norm'].tolist()
        new_embeddings = call_embedding_api(texts_to_fetch)
        
        # Create DataFrame for new entries
        # Filter out failed calls (None)
        valid_indices = [i for i, x in enumerate(new_embeddings) if x is not None]
        
        if len(valid_indices) < len(new_embeddings):
            print(f"Warning: {len(new_embeddings) - len(valid_indices)} texts failed embedding.")
            
        new_cache_data = {
            'text_hash': [missing_texts.iloc[i]['text_hash'] for i in valid_indices],
            'embedding': [new_embeddings[i] for i in valid_indices]
        }
        new_cache_df = pd.DataFrame(new_cache_data)
        
        # Save to disk
        save_cache(new_cache_df)
        
        # Reload full cache
        cache = load_cache()
    
    # 4. Merge back to main DF
    # We merge on text_hash
    # Embeddings in parquet are often stored as arrays/lists.
    df_merged = df.merge(cache, on='text_hash', how='left')
    
    # Drop rows where embedding is null (failed API)
    df_clean = df_merged.dropna(subset=['embedding'])
    
    print(f"Rows with embeddings: {len(df_clean)}")
    return df_clean

def perform_clustering(df, min_k=8, max_k=20):
    print("--- Starting Clustering ---")
    
    # Stack embeddings into a matrix
    # Ensure they are list of floats, convert to numpy array
    matrix = np.stack(df['embedding'].values)
    
    best_k = min_k
    best_score = -1
    best_model = None
    
    # Search for optimal K
    print(f"Searching K from {min_k} to {max_k}...")
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(matrix)
        score = silhouette_score(matrix, labels, sample_size=1000, random_state=RANDOM_SEED)
        
        print(f"K={k}, Silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
            
    print(f"Best K: {best_k} (Score: {best_score:.4f})")
    
    # Assign labels
    df['topic_id'] = best_model.labels_
    
    # Calculate distance to centroid
    # transform() returns distance to all centroids. We take the one for the assigned label.
    dists = best_model.transform(matrix)
    # dists[i, label] is the distance
    df['dist_to_centroid'] = [dists[i, label] for i, label in enumerate(best_model.labels_)]
    
    return df

def extract_keywords(df):
    print("--- Extracting Keywords ---")
    # Group text by topic
    topics = sorted(df['topic_id'].unique())
    
    # Concatenate all texts in a topic
    # Use text_norm
    topic_docs = df.groupby('topic_id')['text_norm'].apply(lambda x: " ".join(x)).tolist()
    
    # TF-IDF
    # Use standard english stop words + custom if needed
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = tfidf.fit_transform(topic_docs)
    feature_names = np.array(tfidf.get_feature_names_out())
    
    topic_keywords = []
    
    for i, topic_id in enumerate(topics):
        # Get top 10 words for this topic
        # Sort indices of the row
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[-10:][::-1]
        top_words = feature_names[top_indices]
        keywords_str = ", ".join(top_words)
        
        topic_keywords.append({
            'topic_id': topic_id,
            'keywords': keywords_str
        })
        
    kw_df = pd.DataFrame(topic_keywords)
    out_path = TABLES_DIR / "topic_keywords.csv"
    kw_df.to_csv(out_path, index=False)
    print(f"Saved keywords to {out_path}")
    return kw_df

def visualize_topics(df):
    print("--- Visualizing Topics ---")
    matrix = np.stack(df['embedding'].values)
    
    # PCA to 2D
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(matrix)
    
    df['emb_2d_x'] = coords[:, 0]
    df['emb_2d_y'] = coords[:, 1]
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Size logic: log1p(likes_n). Fillna with median.
    median_likes = df['likes_n'].median()
    sizes = np.log1p(df['likes_n'].fillna(median_likes)) * 10
    
    scatter = plt.scatter(
        df['emb_2d_x'], 
        df['emb_2d_y'], 
        c=df['topic_id'], 
        cmap='tab20', 
        s=sizes, 
        alpha=0.6,
        edgecolors='none'
    )
    
    plt.colorbar(scatter, label='Topic ID')
    plt.title(f"Topic Clusters (PCA) - K={df['topic_id'].nunique()}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.2)
    
    out_path = FIGURES_DIR / "topics_scatter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot to {out_path}")
    
    return df

def run_clustering_pipeline():
    input_path = DATA_PROCESSED / "tweets_clean.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned data not found at {input_path}")
    
    df = pd.read_parquet(input_path)
    
    # 1. Embeddings
    df = get_embeddings(df)
    
    # 2. Clustering
    df = perform_clustering(df, min_k=8, max_k=20)
    
    # 3. Keywords
    extract_keywords(df)
    
    # 4. Visualization & 2D coords
    df = visualize_topics(df)
    
    # 5. Summary Stats
    summary = df.groupby('topic_id').agg({
        'author_handle': 'nunique',
        'text': 'count',
        'likes_n': 'mean'
    }).rename(columns={'text': 'tweet_count', 'author_handle': 'author_count', 'likes_n': 'avg_likes'})
    
    summary_path = TABLES_DIR / "topic_summary.csv"
    summary.to_csv(summary_path)
    print(f"Saved topic summary to {summary_path}")
    
    # 6. Save final parquet
    # Remove 'embedding' column to save space if 1024 dims is too large? 
    # User asked for "tweets_with_topics.parquet". Keeping embedding might be heavy but useful.
    # But usually separate is better. I will keep it as user implicitly asked for "emb_2d_x", "emb_2d_y". 
    # If the user wants to reuse embeddings later, they are in cache.
    # I'll drop the high-dim embedding column from the final output to keep it lightweight, 
    # relying on cache for the vector itself.
    
    df_out = df.drop(columns=['embedding'])
    out_parquet = DATA_PROCESSED / "tweets_with_topics.parquet"
    df_out.to_parquet(out_parquet, index=False)
    print(f"Saved final data to {out_parquet}")

if __name__ == "__main__":
    run_clustering_pipeline()
