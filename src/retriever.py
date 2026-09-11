"""
Historical Resolution Retriever Module for AmazonHelp Customer Support.
Retrieves the most semantically relevant historical customer inquiry -> agent resolution pairs
from the training corpus, computes an evidence support score, and provides grounded context for RAG.
"""

import os
import re
import joblib
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "train_corpus.csv")
INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "retriever_index.joblib")

class HistoricalResolutionRetriever:
    """
    Indexes historical resolution pairs from the training corpus and retrieves
    evidence based on cosine similarity and intent relevance.
    """
    def __init__(self, corpus_path: str = CORPUS_PATH, index_path: str = INDEX_PATH):
        self.corpus_path = corpus_path
        self.index_path = index_path
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.df_corpus: Optional[pd.DataFrame] = None
        
    def _preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
        
    def build_index(self, max_samples: int = 25000):
        """
        Builds the retrieval index from the training corpus.
        """
        logger.info(f"Building retrieval index from {self.corpus_path}...")
        df = pd.read_csv(self.corpus_path)
        
        # Sample representative examples across intents if corpus is large
        sub_dfs = []
        for intent in df["intent"].unique():
            sub = df[df["intent"] == intent]
            if len(sub) > 4000:
                sub = sub.sample(n=4000, random_state=42)
            sub_dfs.append(sub)
        self.df_corpus = pd.concat(sub_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        # We index the customer message for matching the inquiry
        corpus_texts = self.df_corpus["customer_message"].apply(self._preprocess)
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=30000,
            stop_words="english",
            min_df=2
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)
        
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        joblib.dump({
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "df_corpus": self.df_corpus
        }, self.index_path)
        
        logger.info(f"Retriever index built with {self.tfidf_matrix.shape[0]} documents and saved to {self.index_path}")
        
    def load_index(self):
        """Loads pre-built retrieval index from disk."""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found at {self.index_path}. Call build_index() first.")
        data = joblib.load(self.index_path)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.df_corpus = data["df_corpus"]
        logger.info(f"Loaded retriever index with {len(self.df_corpus)} items from {self.index_path}")
        
    def retrieve(
        self,
        query: str,
        predicted_intent: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieves top-k historical resolution pairs for a given customer query.
        Calculates an aggregate evidence_score based on similarity and intent alignment.
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            if os.path.exists(self.index_path):
                self.load_index()
            else:
                self.build_index()
                
        cleaned_query = self._preprocess(query)
        if not cleaned_query:
            return {
                "retrieved_evidence": [],
                "evidence_score": 0.0
            }
            
        q_vec = self.vectorizer.transform([cleaned_query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        
        # Apply intent boost if intent is provided
        if predicted_intent:
            intent_mask = (self.df_corpus["intent"] == predicted_intent).values
            # Boost matches that share the same intent by 15%
            sims = np.where(intent_mask, sims * 1.15, sims * 0.90)
            
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        evidence_items = []
        for idx in top_indices:
            score = float(sims[idx])
            row = self.df_corpus.iloc[idx]
            evidence_items.append({
                "historical_inquiry": row["customer_message"],
                "historical_reply": row["reference_reply"],
                "intent": row["intent"],
                "similarity_score": round(min(1.0, max(0.0, score)), 4)
            })
            
        # Compute aggregate evidence score
        if evidence_items:
            # Weighted combination of top-1 and top-3 average
            top1 = evidence_items[0]["similarity_score"]
            avg_top = sum(e["similarity_score"] for e in evidence_items) / len(evidence_items)
            aggregate_score = round(0.7 * top1 + 0.3 * avg_top, 4)
        else:
            aggregate_score = 0.0
            
        return {
            "retrieved_evidence": evidence_items,
            "evidence_score": aggregate_score
        }

if __name__ == "__main__":
    retriever = HistoricalResolutionRetriever()
    retriever.build_index()
    
    test_query = "My package says delivered but I checked everywhere and it's not here"
    res = retriever.retrieve(test_query, predicted_intent="order_status_delivery", top_k=3)
    
    print(f"\nQuery: \"{test_query}\"")
    print(f"Evidence Score: {res['evidence_score']}")
    print("Top Evidence:")
    for i, ev in enumerate(res["retrieved_evidence"]):
        print(f"  [{i+1}] (Sim: {ev['similarity_score']}) Historical Reply: {ev['historical_reply']}")
