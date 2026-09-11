"""
Data Preparation & Conversation Reconstruction Module for AmazonHelp Customer Support.
Processes the Kaggle twcs.csv dataset in memory-efficient chunks, reconstructs
inbound customer inquiry -> outbound brand resolution pairs, cleans text,
prevents conversation-level data leakage, and outputs train/evaluation splits.
"""

import os
import re
import sys
import json
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional

# Ensure standard UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_PATH = os.path.join(DATA_DIR, "twcs.csv")

# Intent definitions and keyword markers derived from historical AmazonHelp data
INTENT_PATTERNS = {
    "order_status_delivery": [
        r"\b(where('?s| is) my (order|package|item|delivery))\b",
        r"\b(track(ing)?|transit|delayed|shipment|shipped|carrier|courier|late delivery)\b",
        r"\b(marked delivered|says delivered|never arrived|not delivered|haven't received)\b",
        r"\b(delivery date|expected delivery|dispatch(ed)?|parcel)\b"
    ],
    "returns_and_refunds": [
        r"\b(return(ing)?|return label|send back|drop off|exchange)\b",
        r"\b(refund|money back|refund status|credited back|reimbursement)\b",
        r"\b(return window|return policy|initiate (a )?return)\b"
    ],
    "damaged_defective_item": [
        r"\b(damaged|broken|defective|faulty|smashed|shattered|cracked|scratched)\b",
        r"\b(wrong item|missing (parts|pieces|item)|spilled|leaking|poor condition)\b",
        r"\b(opened package|expired|unusable)\b"
    ],
    "billing_and_payment": [
        r"\b(charged twice|double charged|unauthorized charge|overcharged|billing)\b",
        r"\b(credit card|debit card|payment method|declined|bank account|gift card balance)\b",
        r"\b(transaction fee|invoice|payment issue|charge on my card)\b"
    ],
    "account_and_security": [
        r"\b(password|reset password|forgot password|login|sign in|locked out)\b",
        r"\b(otp|2fa|verification code|two-factor|security code)\b",
        r"\b(hacked|compromised|unauthorized access|suspended account|close (my )?account)\b"
    ],
    "subscription_and_prime": [
        r"\b(prime membership|prime subscription|amazon prime|cancel prime)\b",
        r"\b(auto-renew(al)?|annual fee|monthly prime|prime charge|prime trial|student prime)\b",
        r"\b(prime benefits|prime shipping)\b"
    ],
    "digital_and_devices": [
        r"\b(kindle|firestick|fire tv|echo|alexa|echo dot|echo show|fire tablet)\b",
        r"\b(prime video|streaming issue|kindle app|ebook|audiobook|audible)\b",
        r"\b(device won't turn on|firmware|app crash|frozen screen|alexa command)\b"
    ]
}

def clean_customer_text(text: str) -> str:
    """Cleans customer tweet by stripping handle mentions and excessive whitespace."""
    if not isinstance(text, str):
        return ""
    # Strip twitter handles e.g. @AmazonHelp, @115820
    cleaned = re.sub(r"@\w+", "", text)
    # Normalize URLs
    cleaned = re.sub(r"https?://\S+", "", cleaned)
    # Remove HTML entities
    cleaned = re.sub(r"&[a-z]+;", " ", cleaned)
    # Clean whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def clean_reply_text(text: str) -> str:
    """Cleans company reply by stripping handle mentions and agent signature initials (e.g. ^AG)."""
    if not isinstance(text, str):
        return ""
    # Strip user mention at beginning e.g. @115820
    cleaned = re.sub(r"^(@\w+\s*)+", "", text)
    # Strip trailing agent initials e.g. ^AG, ^KL, /CB
    cleaned = re.sub(r"[\^/][A-Z]{2,3}\s*$", "", cleaned)
    # Clean whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def is_valid_english(text: str) -> bool:
    """Ensures tweet text is readable English without requiring heavy external dependencies."""
    if len(text) < 15 or len(text) > 500:
        return False
    # Check ASCII ratio to filter out Japanese, Arabic, etc. (which AmazonHelp supports)
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars / len(text) < 0.85:
        return False
    return True

def assign_intent(text: str) -> str:
    """Matches text against the 7 customer support intent patterns."""
    text_lower = text.lower()
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        match_count = sum(1 for p in patterns if re.search(p, text_lower))
        if match_count > 0:
            scores[intent] = match_count
            
    if scores:
        # Return intent with highest pattern match count
        return max(scores.items(), key=lambda x: x[1])[0]
    return "order_status_delivery"  # Default modal intent if ambiguous

def extract_amazon_conversations(
    raw_path: str = RAW_DATA_PATH,
    target_brand: str = "AmazonHelp",
    chunksize: int = 100000,
    max_rows_to_scan: Optional[int] = 1200000
) -> pd.DataFrame:
    """
    Scans twcs.csv in chunks, collects AmazonHelp replies and customer inquiries,
    and stitches them into clean conversation pairs.
    """
    logger.info(f"Scanning {raw_path} for brand '{target_brand}'...")
    
    brand_replies = []
    needed_customer_tweet_ids = set()
    rows_scanned = 0
    
    # First pass: collect AmazonHelp outbound replies and the customer tweet IDs they respond to
    for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype=str):
        rows_scanned += len(chunk)
        amazon_chunk = chunk[(chunk["author_id"] == target_brand) & (chunk["inbound"] == "False")]
        
        for _, row in amazon_chunk.iterrows():
            parent_id = row["in_response_to_tweet_id"]
            if pd.notna(parent_id) and str(parent_id).strip():
                brand_replies.append({
                    "reply_tweet_id": str(row["tweet_id"]),
                    "reply_text": str(row["text"]),
                    "in_response_to_tweet_id": str(parent_id),
                    "created_at": str(row["created_at"])
                })
                needed_customer_tweet_ids.add(str(parent_id))
                
        if max_rows_to_scan and rows_scanned >= max_rows_to_scan:
            break
            
    logger.info(f"Collected {len(brand_replies)} {target_brand} replies. Finding {len(needed_customer_tweet_ids)} customer parent tweets...")
    
    # Second pass: extract parent customer tweets
    customer_tweets = {}
    rows_scanned = 0
    for chunk in pd.read_csv(raw_path, chunksize=chunksize, dtype=str):
        rows_scanned += len(chunk)
        # Find rows whose tweet_id is in needed_customer_tweet_ids
        matches = chunk[chunk["tweet_id"].isin(needed_customer_tweet_ids)]
        for _, row in matches.iterrows():
            tid = str(row["tweet_id"])
            customer_tweets[tid] = {
                "customer_tweet_id": tid,
                "author_id": str(row["author_id"]),
                "customer_text": str(row["text"])
            }
        if len(customer_tweets) >= len(needed_customer_tweet_ids):
            break
        if max_rows_to_scan and rows_scanned >= max_rows_to_scan:
            break
            
    logger.info(f"Matched {len(customer_tweets)} customer parent tweets. Constructing conversation pairs...")
    
    pairs = []
    seen_customer_texts = set()
    
    for r in brand_replies:
        p_id = r["in_response_to_tweet_id"]
        if p_id in customer_tweets:
            c = customer_tweets[p_id]
            clean_cust = clean_customer_text(c["customer_text"])
            clean_rep = clean_reply_text(r["reply_text"])
            
            # Quality filters
            if not is_valid_english(clean_cust) or not is_valid_english(clean_rep):
                continue
            if clean_cust in seen_customer_texts:
                continue
                
            seen_customer_texts.add(clean_cust)
            intent = assign_intent(clean_cust)
            
            pairs.append({
                "customer_tweet_id": c["customer_tweet_id"],
                "customer_id": c["author_id"],
                "customer_message": clean_cust,
                "reference_reply": clean_rep,
                "reply_tweet_id": r["reply_tweet_id"],
                "intent": intent,
                "created_at": r["created_at"]
            })
            
    df_pairs = pd.DataFrame(pairs)
    logger.info(f"Successfully constructed {len(df_pairs)} clean conversation pairs across 7 intents.")
    return df_pairs

def split_and_prevent_leakage(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits conversation pairs by customer_id to guarantee that no customer's multi-turn
    conversations or duplicate contexts appear in both train/retrieval and evaluation pools.
    """
    unique_customers = df["customer_id"].drop_duplicates().sample(frac=1.0, random_state=random_state)
    n_train = int(len(unique_customers) * train_ratio)
    train_cust_ids = set(unique_customers.iloc[:n_train])
    
    train_df = df[df["customer_id"].isin(train_cust_ids)].copy()
    eval_df = df[~df["customer_id"].isin(train_cust_ids)].copy()
    
    # Assert strict disjointness
    overlap = set(train_df["customer_id"]).intersection(set(eval_df["customer_id"]))
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED: {len(overlap)} customers overlap between train and eval!"
    
    logger.info(f"Leakage check passed: Train customers={len(train_cust_ids)}, Eval customers={len(unique_customers) - len(train_cust_ids)}, Overlap=0.")
    logger.info(f"Split results: Train/Retrieval rows={len(train_df)}, Eval candidate rows={len(eval_df)}")
    return train_df, eval_df

def run_pipeline(output_dir: str = PROCESSED_DIR):
    """Executes the complete preprocessing and leakage-free dataset creation."""
    os.makedirs(output_dir, exist_ok=True)
    
    df_pairs = extract_amazon_conversations()
    
    train_df, eval_df = split_and_prevent_leakage(df_pairs)
    
    train_path = os.path.join(output_dir, "train_corpus.csv")
    eval_path = os.path.join(output_dir, "eval_candidates.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    eval_df.to_csv(eval_path, index=False, encoding="utf-8")
    
    stats = {
        "total_clean_pairs": len(df_pairs),
        "train_corpus_size": len(train_df),
        "eval_candidates_size": len(eval_df),
        "intent_distribution_train": train_df["intent"].value_counts().to_dict(),
        "intent_distribution_eval": eval_df["intent"].value_counts().to_dict(),
        "leakage_prevention": "Strict customer_id disjoint partitioning"
    }
    
    stats_path = os.path.join(output_dir, "preprocessing_summary.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    logger.info(f"Preprocessing completed successfully. Summary saved to {stats_path}")
    print("\n--- PREPROCESSING SUMMARY ---")
    print(f"Total Clean Pairs: {len(df_pairs):,}")
    print(f"Train/Retrieval Corpus: {len(train_df):,} rows -> {train_path}")
    print(f"Evaluation Pool: {len(eval_df):,} rows -> {eval_path}")
    print("\nTrain Intent Distribution:")
    for intent, cnt in train_df["intent"].value_counts().items():
        print(f"  - {intent:<25}: {cnt}")

if __name__ == "__main__":
    run_pipeline()
