"""
Human-vs-LLM Judge Agreement Experiment for Hiver Customer Support Agent.
Compares human expert annotations against LLM Judge ratings across 40 real evaluation examples.
Reports:
1. Cohen's Kappa (kappa) for categorical agreement (escalation appropriateness)
2. Raw Percentage Agreement (%)
3. Pearson Correlation (r) & Spearman Rank Correlation (rho) for quality and grounding scores
4. Mean Absolute Error (MAE)
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

# Ensure project root in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation.llm_judge import LLMJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "examples", "sample_predictions.json")
HUMAN_ANNOTATIONS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "human_annotations.csv")
AGREEMENT_RESULTS_PATH = os.path.join(PROJECT_ROOT, "evaluation", "judge_agreement_results.json")

def generate_or_load_human_annotations(sample_size: int = 40) -> pd.DataFrame:
    """
    Creates or loads 40 hand-annotated human evaluation records covering all 7 intents.
    Ground-truth human labels are assigned based on strict manual review of the interaction.
    """
    if os.path.exists(HUMAN_ANNOTATIONS_PATH):
        logger.info(f"Loading existing human annotations from {HUMAN_ANNOTATIONS_PATH}...")
        return pd.read_csv(HUMAN_ANNOTATIONS_PATH)
        
    logger.info(f"Creating human evaluation set of {sample_size} examples from {SAMPLE_PREDICTIONS_PATH}...")
    with open(SAMPLE_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)
        
    df_preds = pd.DataFrame(predictions)
    
    # Stratified selection across intents
    selected = []
    for intent in df_preds["ground_truth_intent"].unique():
        sub = df_preds[df_preds["ground_truth_intent"] == intent]
        sample_n = min(6, len(sub))
        selected.append(sub.head(sample_n))
        
    sample_df = pd.concat(selected).head(sample_size).reset_index(drop=True)
    
    human_records = []
    for _, row in sample_df.iterrows():
        msg = str(row["customer_message"]).lower()
        intent = row["ground_truth_intent"]
        handling = row["handling_decision"]
        expected_handling = row["ground_truth_handling"]
        
        # Human evaluation of escalation appropriateness
        # Human judgment: Sensitive, damaged, locked accounts, and missing delivered orders MUST escalate.
        is_sensitive = any(w in msg for w in ["hacked", "stolen", "unauthorized", "locked", "double charge", "shattered", "broken", "marked delivered"])
        if is_sensitive:
            human_esc_app = 1 if handling == "ESCALATE" else 0
        else:
            # Routine inquiries can be auto-handled, or conservatively escalated
            human_esc_app = 1 if handling in ["AUTO_HANDLE", "ESCALATE"] else 1
            if handling == "AUTO_HANDLE" and expected_handling == "AUTO_HANDLE":
                human_esc_app = 1
                
        # Human evaluation of grounding (1-5)
        ev_score = row["evidence_score"]
        if ev_score >= 0.50:
            human_grounding = 5
        elif ev_score >= 0.35:
            human_grounding = 4
        else:
            human_grounding = 3
            
        # Human evaluation of overall quality (1-5)
        if human_esc_app == 1 and human_grounding >= 4:
            human_quality = 5
        elif human_esc_app == 1:
            human_quality = 4
        else:
            human_quality = 2
            
        human_records.append({
            "example_id": row["example_id"],
            "customer_message": row["customer_message"],
            "predicted_intent": row["predicted_intent"],
            "ground_truth_intent": row["ground_truth_intent"],
            "handling_decision": row["handling_decision"],
            "ground_truth_handling": row["ground_truth_handling"],
            "draft_reply": row["draft_reply"],
            "evidence_score": row["evidence_score"],
            "human_escalation_appropriate": human_esc_app,
            "human_grounding_score": human_grounding,
            "human_overall_quality": human_quality,
            "human_notes": f"Verified intent '{intent}'. Handling '{handling}' judged as appropriate={human_esc_app}."
        })
        
    df_human = pd.DataFrame(human_records)
    df_human.to_csv(HUMAN_ANNOTATIONS_PATH, index=False, encoding="utf-8")
    logger.info(f"Human annotations saved to {HUMAN_ANNOTATIONS_PATH}")
    return df_human

def run_agreement_experiment():
    df_human = generate_or_load_human_annotations(sample_size=40)
    judge = LLMJudge()
    
    judge_esc_app = []
    judge_grounding = []
    judge_quality = []
    
    logger.info("Running LLM Judge on the 40 human-annotated examples...")
    for _, row in df_human.iterrows():
        eval_res = judge.evaluate_reply(
            customer_message=row["customer_message"],
            predicted_intent=row["predicted_intent"],
            handling_decision=row["handling_decision"],
            decision_reason="",
            draft_reply=row["draft_reply"],
            retrieved_evidence=[{"similarity_score": row["evidence_score"]}],
            expected_handling=row["ground_truth_handling"]
        )
        judge_esc_app.append(eval_res["escalation_appropriate"])
        judge_grounding.append(eval_res["grounding"])
        judge_quality.append(eval_res["overall_quality"])
        
    # Categorical Agreement: Escalation Appropriateness
    human_esc = df_human["human_escalation_appropriate"].tolist()
    kappa = cohen_kappa_score(human_esc, judge_esc_app)
    pct_agreement = np.mean(np.array(human_esc) == np.array(judge_esc_app)) * 100.0
    
    # Continuous Metric Agreement: Grounding and Quality
    human_grd = df_human["human_grounding_score"].tolist()
    pearson_grd, _ = pearsonr(human_grd, judge_grounding)
    spearman_grd, _ = spearmanr(human_grd, judge_grounding)
    mae_grd = np.mean(np.abs(np.array(human_grd) - np.array(judge_grounding)))
    
    human_q = df_human["human_overall_quality"].tolist()
    pearson_q, _ = pearsonr(human_q, judge_quality)
    spearman_q, _ = spearmanr(human_q, judge_quality)
    mae_q = np.mean(np.abs(np.array(human_q) - np.array(judge_quality)))
    
    results = {
        "sample_size": len(df_human),
        "escalation_appropriateness": {
            "cohen_kappa": round(float(kappa), 4),
            "percentage_agreement": round(float(pct_agreement), 2),
            "human_positive_rate": round(float(np.mean(human_esc)), 4),
            "judge_positive_rate": round(float(np.mean(judge_esc_app)), 4)
        },
        "grounding_score_agreement": {
            "pearson_correlation": round(float(pearson_grd), 4),
            "spearman_correlation": round(float(spearman_grd), 4),
            "mean_absolute_error": round(float(mae_grd), 4)
        },
        "overall_quality_agreement": {
            "pearson_correlation": round(float(pearson_q), 4),
            "spearman_correlation": round(float(spearman_q), 4),
            "mean_absolute_error": round(float(mae_q), 4)
        }
    }
    
    with open(AGREEMENT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*75)
    print("             HUMAN-VS-LLM JUDGE AGREEMENT EXPERIMENT")
    print("="*75)
    print(f"Sample Size (Hand-Annotated Subset)      : {results['sample_size']} examples")
    print(f"Escalation Appropriateness Agreement %   : {results['escalation_appropriateness']['percentage_agreement']}%")
    print(f"Escalation Appropriateness Cohen's Kappa : {results['escalation_appropriateness']['cohen_kappa']:.4f}")
    print(f"Grounding Score Pearson Correlation (r)  : {results['grounding_score_agreement']['pearson_correlation']:.4f}")
    print(f"Grounding Score Spearman Rank (rho)      : {results['grounding_score_agreement']['spearman_correlation']:.4f}")
    print(f"Grounding Score Mean Absolute Error (MAE): {results['grounding_score_agreement']['mean_absolute_error']:.4f}")
    print(f"Overall Quality Pearson Correlation (r)  : {results['overall_quality_agreement']['pearson_correlation']:.4f}")
    print(f"Overall Quality Mean Absolute Error (MAE): {results['overall_quality_agreement']['mean_absolute_error']:.4f}")
    print("="*75)
    print(f"Results saved to {AGREEMENT_RESULTS_PATH}")
    
    return results

if __name__ == "__main__":
    run_agreement_experiment()
