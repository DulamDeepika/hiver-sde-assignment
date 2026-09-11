"""
Brand Selection Analysis Script for Hiver SDE Assignment.
Analyzes the Kaggle Twitter Customer Support dataset (twcs.csv) in chunks
to identify top candidate brands, conversation structures, and response quality.
"""

import os
import json
from collections import Counter, defaultdict
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "twcs.csv")

def analyze_brands(chunksize=100000, max_chunks=30):
    print(f"Reading {DATA_PATH} in chunks of {chunksize}...")
    
    brand_outbound_counts = Counter()
    brand_first_contact_counts = Counter()
    brand_substantive_replies = Counter()
    brand_dm_redirect_replies = Counter()
    brand_sample_replies = defaultdict(list)
    brand_sample_pairs = defaultdict(list)
    
    chunks_processed = 0
    total_rows = 0
    
    # Process chunks
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize, dtype={'tweet_id': str, 'response_tweet_id': str, 'in_response_to_tweet_id': str}):
        chunks_processed += 1
        total_rows += len(chunk)
        
        # Identify company tweets (inbound == False)
        outbound = chunk[chunk['inbound'] == False]
        for _, row in outbound.iterrows():
            brand = row['author_id']
            text = str(row['text'])
            brand_outbound_counts[brand] += 1
            
            # Check for generic DM redirects vs substantive answers
            text_lower = text.lower()
            if any(term in text_lower for term in ['send us a dm', 'dm us', 'direct message', 'pm us', 'private message']):
                brand_dm_redirect_replies[brand] += 1
            else:
                brand_substantive_replies[brand] += 1
                
            if len(brand_sample_replies[brand]) < 5:
                brand_sample_replies[brand].append(text)
                
        if max_chunks and chunks_processed >= max_chunks:
            break
            
    print(f"Processed {chunks_processed} chunks ({total_rows:,} rows).")
    top_brands = brand_outbound_counts.most_common(12)
    
    results = []
    for brand, count in top_brands:
        dm_count = brand_dm_redirect_replies[brand]
        substantive = brand_substantive_replies[brand]
        substantive_ratio = substantive / count if count > 0 else 0
        results.append({
            "brand": brand,
            "total_outbound_tweets": count,
            "substantive_replies": substantive,
            "dm_redirect_replies": dm_count,
            "substantive_ratio": round(substantive_ratio, 3),
            "samples": brand_sample_replies[brand][:3]
        })
        
    return results

if __name__ == "__main__":
    results = analyze_brands(chunksize=100000, max_chunks=30)
    print("\n--- TOP CANDIDATE BRANDS ---")
    for r in results:
        print(f"Brand: {r['brand']:<15} | Outbound: {r['total_outbound_tweets']:<7} | Substantive %: {r['substantive_ratio']*100:.1f}%")
        
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "brand_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed analysis to {output_path}")
