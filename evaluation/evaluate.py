"""
Comprehensive Evaluation Harness for Hiver AI Customer Support Agent.
Runs comparative benchmarks on the 200-example Golden Evaluation Set across:
1. Baseline 1: Majority Class Predictor
2. Baseline 2: Simple TF-IDF + Logistic Regression
3. Proposed System: End-to-End Grounded Customer Support Agent

Calculates Accuracy, Macro F1, Precision, Recall, Confusion Matrix,
Escalation Precision/Recall, and Evidence Grounding Scores.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# Ensure project root in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agent import CustomerSupportAgent
from evaluation.baselines import MajorityClassBaseline, SimpleLogisticRegressionBaseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = os.path.join(PROJECT_ROOT, "evaluation", "golden_set.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "evaluation_results.json")
SAMPLE_PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "examples", "sample_predictions.json")

def compute_metrics(y_true_intent: List[str], y_pred_intent: List[str],
                    y_true_handling: List[str], y_pred_handling: List[str],
                    labels: List[str]) -> Dict[str, Any]:
    """
    Computes intent classification metrics and escalation decision metrics.
    """
    # 1. Intent Metrics
    intent_acc = accuracy_score(y_true_intent, y_pred_intent)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true_intent, y_pred_intent, labels=labels).tolist()
    
    # Per-intent metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true_intent, y_pred_intent, labels=labels, zero_division=0
    )
    per_intent = {}
    for i, label in enumerate(labels):
        per_intent[label] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(sup_per[i])
        }
        
    # 2. Handling / Escalation Metrics
    handling_acc = accuracy_score(y_true_handling, y_pred_handling)
    # ESCALATE class metrics (pos_label="ESCALATE")
    p_esc, r_esc, f1_esc, _ = precision_recall_fscore_support(
        y_true_handling, y_pred_handling, labels=["ESCALATE"], average="macro", zero_division=0
    )
    p_auto, r_auto, f1_auto, _ = precision_recall_fscore_support(
        y_true_handling, y_pred_handling, labels=["AUTO_HANDLE"], average="macro", zero_division=0
    )
    
    return {
        "intent_accuracy": round(float(intent_acc), 4),
        "intent_macro_f1": round(float(f1_macro), 4),
        "intent_macro_precision": round(float(p_macro), 4),
        "intent_macro_recall": round(float(r_macro), 4),
        "intent_weighted_f1": round(float(f1_weighted), 4),
        "confusion_matrix": cm,
        "per_intent_metrics": per_intent,
        "handling_accuracy": round(float(handling_acc), 4),
        "escalation_precision": round(float(p_esc), 4),
        "escalation_recall": round(float(r_esc), 4),
        "escalation_f1": round(float(f1_esc), 4),
        "auto_handle_precision": round(float(p_auto), 4),
        "auto_handle_recall": round(float(r_auto), 4),
        "auto_handle_f1": round(float(f1_auto), 4)
    }

def run_evaluation():
    logger.info(f"Loading golden evaluation set from {GOLDEN_SET_PATH}...")
    df_golden = pd.read_csv(GOLDEN_SET_PATH)
    
    y_true_intent = df_golden["intent"].tolist()
    y_true_handling = df_golden["expected_handling"].tolist()
    texts = df_golden["customer_message"].tolist()
    
    labels = sorted(list(set(y_true_intent)))
    logger.info(f"Total evaluation examples: {len(texts)} across {len(labels)} intents.")
    
    # -------------------------------------------------------------
    # 1. Evaluate Baseline 1: Majority Class Predictor
    # -------------------------------------------------------------
    logger.info("Evaluating Baseline 1 (Majority Class)...")
    b1 = MajorityClassBaseline().fit()
    p1 = b1.predict(texts)
    metrics_b1 = compute_metrics(
        y_true_intent, p1["predicted_intent"],
        y_true_handling, p1["handling_decision"],
        labels
    )
    
    # -------------------------------------------------------------
    # 2. Evaluate Baseline 2: Simple TF-IDF + Logistic Regression
    # -------------------------------------------------------------
    logger.info("Evaluating Baseline 2 (Simple TF-IDF + Logistic Regression)...")
    b2 = SimpleLogisticRegressionBaseline().fit()
    p2 = b2.predict(texts)
    metrics_b2 = compute_metrics(
        y_true_intent, p2["predicted_intent"],
        y_true_handling, p2["handling_decision"],
        labels
    )
    
    # -------------------------------------------------------------
    # 3. Evaluate Proposed System: Full Customer Support Agent
    # -------------------------------------------------------------
    logger.info("Evaluating Proposed System (Full Customer Support Agent)...")
    agent = CustomerSupportAgent()
    
    agent_preds = []
    evidence_scores = []
    
    for i, row in df_golden.iterrows():
        msg = row["customer_message"]
        res = agent.process_message(msg)
        # Add metadata for audit
        res["example_id"] = row["example_id"]
        res["ground_truth_intent"] = row["intent"]
        res["ground_truth_handling"] = row["expected_handling"]
        res["reference_reply"] = row["reference_reply"]
        
        agent_preds.append(res)
        evidence_scores.append(res["evidence_score"])
        
    y_agent_intent = [p["predicted_intent"] for p in agent_preds]
    y_agent_handling = [p["handling_decision"] for p in agent_preds]
    
    metrics_agent = compute_metrics(
        y_true_intent, y_agent_intent,
        y_true_handling, y_agent_handling,
        labels
    )
    
    # Evidence scores
    mean_ev_score = float(np.mean(evidence_scores))
    median_ev_score = float(np.median(evidence_scores))
    metrics_agent["evidence_score_mean"] = round(mean_ev_score, 4)
    metrics_agent["evidence_score_median"] = round(median_ev_score, 4)
    
    # Save full sample predictions for failure analysis and audit
    os.makedirs(os.path.dirname(SAMPLE_PREDICTIONS_PATH), exist_ok=True)
    with open(SAMPLE_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(agent_preds, f, indent=2)
    logger.info(f"Saved {len(agent_preds)} detailed predictions to {SAMPLE_PREDICTIONS_PATH}")
    
    # Assemble comprehensive results document
    results = {
        "evaluation_dataset": {
            "path": GOLDEN_SET_PATH,
            "total_examples": len(df_golden),
            "intents": labels,
            "intent_distribution": df_golden["intent"].value_counts().to_dict(),
            "handling_distribution": df_golden["expected_handling"].value_counts().to_dict()
        },
        "metrics_summary": {
            "baseline_1_majority": metrics_b1,
            "baseline_2_tfidf_lr": metrics_b2,
            "proposed_system": metrics_agent
        }
    }
    
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved complete evaluation results to {RESULTS_PATH}")
    
    # Print formatted comparison table
    print("\n" + "="*85)
    print("                      HEADLINE BENCHMARK RESULTS")
    print("="*85)
    print(f"{'Metric':<30} | {'Baseline 1 (Majority)':<22} | {'Baseline 2 (Simple LR)':<22} | {'Proposed System':<16}")
    print("-"*85)
    print(f"{'Intent Accuracy':<30} | {metrics_b1['intent_accuracy']*100:<21.1f}% | {metrics_b2['intent_accuracy']*100:<21.1f}% | {metrics_agent['intent_accuracy']*100:<15.1f}%")
    print(f"{'Intent Macro F1':<30} | {metrics_b1['intent_macro_f1']*100:<21.1f}% | {metrics_b2['intent_macro_f1']*100:<21.1f}% | {metrics_agent['intent_macro_f1']*100:<15.1f}%")
    print(f"{'Intent Weighted F1':<30} | {metrics_b1['intent_weighted_f1']*100:<21.1f}% | {metrics_b2['intent_weighted_f1']*100:<21.1f}% | {metrics_agent['intent_weighted_f1']*100:<15.1f}%")
    print(f"{'Escalation Precision':<30} | {metrics_b1['escalation_precision']*100:<21.1f}% | {metrics_b2['escalation_precision']*100:<21.1f}% | {metrics_agent['escalation_precision']*100:<15.1f}%")
    print(f"{'Escalation Recall':<30} | {metrics_b1['escalation_recall']*100:<21.1f}% | {metrics_b2['escalation_recall']*100:<21.1f}% | {metrics_agent['escalation_recall']*100:<15.1f}%")
    print(f"{'Escalation F1':<30} | {metrics_b1['escalation_f1']*100:<21.1f}% | {metrics_b2['escalation_f1']*100:<21.1f}% | {metrics_agent['escalation_f1']*100:<15.1f}%")
    print(f"{'Overall Handling Accuracy':<30} | {metrics_b1['handling_accuracy']*100:<21.1f}% | {metrics_b2['handling_accuracy']*100:<21.1f}% | {metrics_agent['handling_accuracy']*100:<15.1f}%")
    print(f"{'Mean Evidence Grounding':<30} | {'N/A':<22} | {'N/A':<22} | {metrics_agent['evidence_score_mean']:<16.3f}")
    print("="*85)
    
    return results

if __name__ == "__main__":
    run_evaluation()
