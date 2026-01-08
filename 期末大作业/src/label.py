import os
import json
import time
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from .config import DATA_PROCESSED, OUTPUTS_DIR, TABLES_DIR

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL_NAME = "qwen3-max"

SYSTEM_PROMPT = """You are an expert social media analyst specializing in public discourse analysis during the COVID-19 pandemic. 
Your task is to analyze a cluster of tweets and extract the coherent theme, framing, and stance.
You must output strict, valid JSON only. Do not add any markdown formatting (like ```json), commentary, or extra text."""

def get_llm_json(messages, retries=3):
    """
    Call the LLM and parse the response as JSON.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "temperature": 0.3 # Lower temperature for consistent formatting
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            
            # Clean potential markdown
            content_clean = content.strip()
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
            
            return json.loads(content_clean)
            
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt == retries - 1:
                print(f"LLM Error after {retries} attempts: {e}")
                print(f"Raw content was: {content if 'content' in locals() else 'No content'}")
                return None
            time.sleep(2)
    return None

def construct_user_prompt(topic_id, samples, top_keywords):
    """
    Construct the user prompt with tweet samples and keywords.
    """
    tweets_str = "\n".join([f"- {t}" for t in samples])
    
    return f"""Analyze the following cluster of tweets (Topic ID: {topic_id}) and the provided TF-IDF keywords. 
    
    Top Keywords: {top_keywords}
    
    Sample Tweets:
    {tweets_str}
    
    Task:
    1. Identify a concise "topic_label" (3-5 words).
    2. Write a "one_sentence_summary".
    3. Categorize into "high_level_category": ["vaccine", "remote_work", "covid_general", "other"].
    4. Extract 3-6 "frame_tags" (e.g., "economic impact", "health mandate", "personal struggle").
    5. Estimate "stance_profile" (fractions summing to ~1.0).
    
    Output strictly in this JSON format:
    {{
        "topic_id": {topic_id},
        "topic_label": "string",
        "one_sentence_summary": "string",
        "high_level_category": "string",
        "frame_tags": ["tag1", "tag2"],
        "stance_profile": {{
            "mostly_informational": 0.0,
            "mostly_personal_experience": 0.0,
            "mostly_policy_or_advocacy": 0.0,
            "mostly_argument_or_conflict": 0.0
        }},
        "representative_keywords": ["kw1", "kw2", "kw3"]
    }}
    """

def run_labeling_pipeline():
    print("--- Starting LLM Labeling Pipeline ---")
    
    # 1. Load Data
    tweets_path = DATA_PROCESSED / "tweets_with_topics.parquet"
    keywords_path = TABLES_DIR / "topic_keywords.csv"
    
    if not tweets_path.exists():
        raise FileNotFoundError(f"File not found: {tweets_path}")
        
    df = pd.read_parquet(tweets_path)
    
    # Load keywords if available
    topic_kw_map = {}
    if keywords_path.exists():
        kw_df = pd.read_csv(keywords_path)
        for _, row in kw_df.iterrows():
            topic_kw_map[row['topic_id']] = row['keywords']
    
    # 2. Select Samples per Topic
    topic_ids = sorted(df['topic_id'].unique())
    results = []
    
    print(f"Labeling {len(topic_ids)} topics...")
    
    for tid in tqdm(topic_ids):
        # Filter for this topic
        subset = df[df['topic_id'] == tid]
        
        # Strategy 1: Closest to centroid (N=20)
        # Sort by dist_to_centroid ascending
        representative = subset.sort_values('dist_to_centroid', ascending=True).head(20)
        
        # Strategy 2: High Engagement (M=5)
        # Sort by likes_n descending
        influential = subset.sort_values('likes_n', ascending=False).head(5)
        
        # Combine and deduplicate
        combined_samples = pd.concat([representative, influential]).drop_duplicates(subset='text_norm')
        
        # Get text list
        sample_texts = combined_samples['text_norm'].tolist()
        keywords = topic_kw_map.get(tid, "")
        
        # 3. Call LLM
        user_content = construct_user_prompt(tid, sample_texts, keywords)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        label_data = get_llm_json(messages)
        
        if label_data:
            # Enforce topic_id consistency
            label_data['topic_id'] = int(tid)
            results.append(label_data)
        else:
            print(f"Failed to label topic {tid}")
            # Append a fallback to keep pipeline running?
            results.append({
                "topic_id": int(tid),
                "topic_label": f"Topic {tid} (Label Failed)",
                "high_level_category": "other",
                "frame_tags": [],
                "stance_profile": {},
                "representative_keywords": []
            })
    
    # 4. Save Results
    # JSON
    json_path = DATA_PROCESSED / "topic_labels.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON labels to {json_path}")
    
    # CSV Table
    labels_df = pd.DataFrame(results)
    # Flatten stance profile for CSV
    stance_df = pd.json_normalize(labels_df['stance_profile'])
    csv_df = labels_df.drop(columns=['stance_profile']).join(stance_df)
    
    csv_path = TABLES_DIR / "topic_labels.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved CSV labels to {csv_path}")
    
    # 5. Merge back to Parquet
    # We mainly want the label and category on the main dataframe
    merge_df = labels_df[['topic_id', 'topic_label', 'high_level_category']]
    final_df = df.merge(merge_df, on='topic_id', how='left')
    
    final_parquet_path = DATA_PROCESSED / "tweets_labeled.parquet"
    final_df.to_parquet(final_parquet_path, index=False)
    print(f"Saved labeled dataframe to {final_parquet_path}")

if __name__ == "__main__":
    run_labeling_pipeline()
