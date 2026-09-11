# Data Directory: Customer Support on Twitter (AmazonHelp)

This directory contains the dataset assets, preprocessed conversation pairs, and evaluation sets for the Hiver SDE Take-Home assignment.

## Dataset Origin
* **Source**: Kaggle `thoughtvector/customer-support-on-twitter` (`twcs.csv`)
* **Total Records**: 2,811,774 tweets across multiple global brands.
* **Target Brand**: `AmazonHelp` (169,840 outbound tweets; largest volume, substantive responses).

## Structure
```
data/
├── twcs.csv                     # Raw Kaggle dataset (not committed to git)
├── brand_analysis.json          # Quantitative brand comparison data across top 12 brands
├── brand_samples.json           # Inspected conversation samples for top candidate brands
└── processed/
    ├── preprocessing_summary.json  # Ingestion and splitting statistics
    ├── train_corpus.csv            # 57,604 clean conversation pairs for training & retrieval
    └── eval_candidates.csv         # 14,495 clean conversation pairs for evaluation pool
```

## Leakage Prevention
To prevent data leakage:
* Multi-turn tweets are linked via `in_response_to_tweet_id` to form complete conversation pairs `(customer_message, reference_reply)`.
* Splitting between `train_corpus.csv` and `eval_candidates.csv` is performed strictly at the unique **`customer_id`** level (80% train, 20% eval).
* Overlap between train and eval customer IDs is mathematically verified to be **0**. No customer's multi-turn context or near-duplicate messages appear in both splits.
