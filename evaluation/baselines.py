"""
Baselines Module for Hiver Customer Support Agent Evaluation.
Implements:
1. Baseline 1: Majority Class Predictor (trivial baseline)
2. Baseline 2: Simple TF-IDF + Logistic Regression (standard classical ML baseline)
"""

import os
import sys
import pandas as pd
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRAIN_CORPUS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train_corpus.csv")
GOLDEN_SET_PATH = os.path.join(PROJECT_ROOT, "evaluation", "golden_set.csv")

class MajorityClassBaseline:
    """
    Baseline 1: Always predicts the most frequent class in the training corpus.
    """
    def __init__(self, corpus_path: str = TRAIN_CORPUS_PATH):
        self.corpus_path = corpus_path
        self.majority_intent = "order_status_delivery"
        self.majority_handling = "AUTO_HANDLE"
        
    def fit(self):
        if os.path.exists(self.corpus_path):
            df = pd.read_csv(self.corpus_path)
            self.majority_intent = df["intent"].mode()[0]
        return self
        
    def predict(self, texts: List[str]) -> Dict[str, List[Any]]:
        return {
            "predicted_intent": [self.majority_intent] * len(texts),
            "confidence": [1.0] * len(texts),
            "handling_decision": [self.majority_handling] * len(texts)
        }

class SimpleLogisticRegressionBaseline:
    """
    Baseline 2: Simple unigram TF-IDF + unweighted standard Logistic Regression.
    """
    def __init__(self, corpus_path: str = TRAIN_CORPUS_PATH):
        self.corpus_path = corpus_path
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.clf = LogisticRegression(max_iter=500, random_state=42)
        
    def fit(self, max_samples: int = 20000):
        df = pd.read_csv(self.corpus_path)
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        X = self.vectorizer.fit_transform(df["customer_message"].fillna(""))
        y = df["intent"]
        self.clf.fit(X, y)
        return self
        
    def predict(self, texts: List[str]) -> Dict[str, List[Any]]:
        X_test = self.vectorizer.transform(texts)
        preds = self.clf.predict(X_test)
        probs = self.clf.predict_proba(X_test).max(axis=1)
        # Default simple heuristic for handling: if confidence < 0.5 escalate, else auto_handle
        handling = ["ESCALATE" if p < 0.5 else "AUTO_HANDLE" for p in probs]
        return {
            "predicted_intent": list(preds),
            "confidence": [round(float(p), 4) for p in probs],
            "handling_decision": handling
        }

if __name__ == "__main__":
    df_golden = pd.read_csv(GOLDEN_SET_PATH)
    texts = df_golden["customer_message"].tolist()
    
    b1 = MajorityClassBaseline().fit()
    p1 = b1.predict(texts)
    print(f"Baseline 1 (Majority Class) predicted {len(p1['predicted_intent'])} examples.")
    
    b2 = SimpleLogisticRegressionBaseline().fit()
    p2 = b2.predict(texts)
    print(f"Baseline 2 (Simple TF-IDF + Logistic Regression) predicted {len(p2['predicted_intent'])} examples.")
