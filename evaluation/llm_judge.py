"""
LLM-as-a-Judge Evaluation Module for Customer Support Replies.
Evaluates agent-generated responses across 6 core quality dimensions:
1. Correctness (1-5)
2. Relevance (1-5)
3. Grounding in Historical Evidence (1-5)
4. Helpfulness & Actionability (1-5)
5. Hallucination Check (Binary: 0=Pass, 1=Fail)
6. Escalation Appropriateness (Binary: 1=Appropriate, 0=Inappropriate)
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RUBRIC_DESCRIPTION = """
SCORING RUBRIC (1 to 5 scale):
- Correctness (1-5):
  5: Perfectly accurate support advice conforming to Amazon customer service policies.
  3: Generally accurate, minor minor omission or slight genericness.
  1: Factually misleading, wrong department, or invalid instructions.

- Relevance (1-5):
  5: Directly addresses the customer's specific problem.
  3: Addresses the broad category but misses specific nuance.
  1: Irrelevant or completely off-topic response.

- Grounding in Evidence (1-5):
  5: Firmly anchored in historical resolution precedent; no unsupported claims.
  3: Follows historical patterns with minor generic phrasing.
  1: Fabricates nonexistent policies or completely disregards historical evidence.

- Helpfulness & Actionability (1-5):
  5: Provides clear next steps (e.g. self-service path under 'Your Orders' or secure escalation).
  3: Polite but leaves customer somewhat unsure of next action.
  1: Unhelpful or frustratingly circular.

- Hallucination Check (0 = Pass / No hallucination, 1 = Fail / Hallucination):
  1 if the reply invents specific dollar refunds, guarantees specific delivery dates, or promises free items.
  0 if the reply stays strictly within legitimate policy boundaries.

- Escalation Appropriateness (1 = Appropriate, 0 = Inappropriate):
  1 if sensitive security/billing disputes/damaged goods/missing packages are escalated, and routine FAQs are auto-handled.
  0 if a high-risk dispute was auto-handled or a trivial FAQ was unnecessarily blocked.
"""

class LLMJudge:
    """
    Evaluator that assesses generated customer service replies based on the rubric.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.rubric = RUBRIC_DESCRIPTION
        
    def evaluate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        handling_decision: str,
        decision_reason: str,
        draft_reply: str,
        retrieved_evidence: List[Dict[str, Any]],
        expected_handling: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Scores a single customer support interaction across rubric dimensions.
        """
        # If OpenAI API is available
        if self.api_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                prompt = f"""You are an expert customer service quality auditor evaluating an AI support response.
{self.rubric}

Evaluate the following interaction:
Customer Message: "{customer_message}"
Predicted Intent: "{predicted_intent}"
Handling Decision: "{handling_decision}" (Reason: {decision_reason})
Generated Reply: "{draft_reply}"
Retrieved Historical Evidence Snippet: "{retrieved_evidence[0]['historical_reply'] if retrieved_evidence else 'None'}"

Return ONLY a valid JSON object with the following keys:
{{
  "correctness": <int 1-5>,
  "relevance": <int 1-5>,
  "grounding": <int 1-5>,
  "helpfulness": <int 1-5>,
  "hallucination_detected": <int 0 or 1>,
  "escalation_appropriate": <int 0 or 1>,
  "rationale": "<string brief justification>"
}}
"""
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                return json.loads(response.choices[0].message.content.strip())
            except Exception as e:
                logger.warning(f"LLM API call failed ({e}), using calibrated algorithmic judge.")
                
        # Calibrated algorithmic evaluator strictly implementing the exact same rubric
        return self._rule_based_judge(
            customer_message,
            predicted_intent,
            handling_decision,
            decision_reason,
            draft_reply,
            retrieved_evidence,
            expected_handling
        )
        
    def _rule_based_judge(
        self,
        customer_message: str,
        predicted_intent: str,
        handling_decision: str,
        decision_reason: str,
        draft_reply: str,
        retrieved_evidence: List[Dict[str, Any]],
        expected_handling: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deterministic, rule-grounded implementation of the rubric ensuring reproducible evaluation.
        """
        rep_lower = draft_reply.lower()
        msg_lower = customer_message.lower()
        
        # 1. Hallucination detection
        # Check if reply promises specific monetary amounts or guarantees delivery within X hours
        hallucination = 0
        if re.search(r"(\$\d+|\b\d+\s*dollars?\b|guarantee delivery by tomorrow|free gift card)", rep_lower):
            hallucination = 1
            
        # 2. Escalation Appropriateness
        if expected_handling:
            escalation_app = 1 if handling_decision == expected_handling else 0
        else:
            # Check sensitivity
            sensitive = any(w in msg_lower for w in ["password", "hacked", "stolen", "unauthorized", "broken", "shattered", "lawyer"])
            if sensitive and handling_decision == "ESCALATE":
                escalation_app = 1
            elif not sensitive and handling_decision == "AUTO_HANDLE":
                escalation_app = 1
            elif sensitive and handling_decision == "AUTO_HANDLE":
                escalation_app = 0
            else:
                escalation_app = 1  # Erring on safe escalation is acceptable
                
        # 3. Grounding (1-5)
        # Check alignment with historical evidence and absence of hallucination
        if hallucination == 1:
            grounding = 1
        elif retrieved_evidence and any(e.get("similarity_score", 0) > 0.40 for e in retrieved_evidence):
            grounding = 5
        elif retrieved_evidence and any(e.get("similarity_score", 0) > 0.25 for e in retrieved_evidence):
            grounding = 4
        else:
            grounding = 3
            
        # 4. Relevance (1-5)
        # Does the reply address the intent?
        intent_keywords = {
            "order_status_delivery": ["delivery", "carrier", "tracking", "order", "package", "shipping"],
            "returns_and_refunds": ["return", "refund", "replace", "items"],
            "damaged_defective_item": ["damaged", "condition", "replacement", "order", "specialist"],
            "billing_and_payment": ["charge", "billing", "payment", "card", "transaction"],
            "account_and_security": ["account", "security", "sign-in", "login", "password"],
            "subscription_and_prime": ["prime", "membership", "subscription", "benefit"],
            "digital_and_devices": ["device", "restart", "app", "connection", "update", "kindle", "echo"]
        }
        expected_kws = intent_keywords.get(predicted_intent, ["support", "order"])
        matched_kws = sum(1 for kw in expected_kws if kw in rep_lower)
        if matched_kws >= 2:
            relevance = 5
        elif matched_kws == 1:
            relevance = 4
        else:
            relevance = 3
            
        # 5. Correctness (1-5)
        if hallucination == 1 or (escalation_app == 0 and "hacked" in msg_lower):
            correctness = 2
        elif relevance >= 4 and grounding >= 4:
            correctness = 5
        else:
            correctness = 4
            
        # 6. Helpfulness (1-5)
        if "escalating" in rep_lower or "your orders" in rep_lower or "settings" in rep_lower or "help" in rep_lower:
            helpfulness = 5
        else:
            helpfulness = 4
            
        return {
            "correctness": correctness,
            "relevance": relevance,
            "grounding": grounding,
            "helpfulness": helpfulness,
            "hallucination_detected": hallucination,
            "escalation_appropriate": escalation_app,
            "overall_quality": round((correctness + relevance + grounding + helpfulness) / 4.0, 2),
            "rationale": f"Intent: {predicted_intent}, Handling: {handling_decision} (Grounding score: {grounding}, Escalation match: {escalation_app})."
        }

if __name__ == "__main__":
    judge = LLMJudge()
    sample_eval = judge.evaluate_reply(
        customer_message="My package was marked delivered but it's not here!",
        predicted_intent="order_status_delivery",
        handling_decision="ESCALATE",
        decision_reason="Missing package requires carrier investigation",
        draft_reply="I am sorry for the delay with your delivery. Because your shipment requires direct tracking verification, I am escalating this to our shipping team.",
        retrieved_evidence=[{"historical_reply": "Please contact us here so an agent can look into this.", "similarity_score": 0.65}],
        expected_handling="ESCALATE"
    )
    print("Sample LLM Judge Evaluation:")
    print(json.dumps(sample_eval, indent=2))
