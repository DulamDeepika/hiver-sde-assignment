"""
Intent Classification Module for AmazonHelp Customer Support.
Provides a calibrated, balanced multi-class classifier to predict
customer intent from text inquiries with confidence scores.
"""

import os
import re
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "intent_classifier.joblib")
TRAIN_CORPUS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed", "train_corpus.csv")

TAXONOMY_INTENTS = [
    "order_status_delivery",
    "returns_and_refunds",
    "damaged_defective_item",
    "billing_and_payment",
    "account_and_security",
    "subscription_and_prime",
    "digital_and_devices"
]

class IntentClassifier:
    """
    Calibrated intent classifier using n-gram TF-IDF and class-balanced Logistic Regression.
    """
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None
        
    def _preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        # Remove mentions and urls
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
        
    def train(self, corpus_path: str = TRAIN_CORPUS_PATH, max_samples: int = 50000) -> Dict[str, Any]:
        """
        Trains the classifier on the training corpus with stratified sampling across classes.
        """
        logger.info(f"Loading training data from {corpus_path}...")
        df = pd.read_csv(corpus_path)
        
        # Balance classes to avoid extreme majority dominance while maintaining representativeness
        # Cap majority class (order_status_delivery) so other classes get fair representation
        sampled_dfs = []
        for intent in df["intent"].unique():
            sub = df[df["intent"] == intent]
            if len(sub) > 6000:
                sub = sub.sample(n=6000, random_state=42)
            sampled_dfs.append(sub)
        balanced_df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        X = balanced_df["customer_message"].apply(self._preprocess)
        y = balanced_df["intent"]
        
        logger.info(f"Training on {len(balanced_df)} samples across {len(y.unique())} intents...")
        
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=25000,
                stop_words="english",
                min_df=2
            )),
            ("clf", LogisticRegression(
                class_weight="balanced",
                C=2.0,
                max_iter=1000,
                solver="lbfgs"
            ))
        ])
        
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        logger.info(f"Trained intent classifier successfully saved to {self.model_path}")
        
        preds = self.pipeline.predict(X)
        acc = accuracy_score(y, preds)
        macro_f1 = f1_score(y, preds, average="macro")
        
        logger.info(f"Training metrics - Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")
        return {"accuracy": acc, "macro_f1": macro_f1, "classes": list(self.classes_)}
        
    def load(self):
        """Loads a pre-trained pipeline from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train the model first.")
        self.pipeline = joblib.load(self.model_path)
        self.classes_ = self.pipeline.named_steps["clf"].classes_
        logger.info(f"Loaded intent classifier from {self.model_path}")
        
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predicts the intent and confidence score for a given customer message.
        """
        if self.pipeline is None:
            if os.path.exists(self.model_path):
                self.load()
            else:
                self.train()
                
        cleaned = self._preprocess(text)
        if not cleaned:
            return {
                "predicted_intent": "order_status_delivery",
                "confidence": 0.30,
                "probabilities": {}
            }
            
        probs = self.pipeline.predict_proba([cleaned])[0]
        max_idx = np.argmax(probs)
        predicted_intent = self.classes_[max_idx]
        confidence = float(probs[max_idx])
        
        all_probs = {cls: round(float(p), 4) for cls, p in zip(self.classes_, probs)}
        
        return {
            "predicted_intent": predicted_intent,
            "confidence": round(confidence, 4),
            "probabilities": all_probs
        }

if __name__ == "__main__":
    clf = IntentClassifier()
    metrics = clf.train()
    print("\n--- INTENT CLASSIFIER TRAINED ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    # Test a few sample inputs
    test_cases = [
        "Where is my package? It was supposed to be delivered yesterday.",
        "I need to return this broken laptop screen, how do I get a refund?",
        "My account got locked and I cannot reset my password",
        "Why was my credit card charged twice for Amazon Prime?",
        "My Kindle fire screen is completely black and won't turn on"
    ]
    print("\n--- SAMPLE PREDICTIONS ---")
    for t in test_cases:
        res = clf.predict(t)
        print(f"Input: \"{t}\"")
        print(f" -> Predicted: {res['predicted_intent']} (Conf: {res['confidence']:.2f})\n")
