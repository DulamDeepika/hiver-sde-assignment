"""
Escalation Policy Engine for AmazonHelp Customer Support.
Evaluates intent confidence, historical evidence strength, sensitive domain risk,
and customer sentiment to make a deterministic, auditable AUTO_HANDLE vs ESCALATE decision.
"""

import re
import logging
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configurable thresholds
INTENT_CONFIDENCE_THRESHOLD = 0.65
EVIDENCE_SCORE_THRESHOLD = 0.40

# Risk keyword triggers
SECURITY_KEYWORDS = [
    r"\b(locked account|can'?t log ?in|hacked|compromised|unauthorized access|stolen account)\b",
    r"\b(reset password|change password|forgot password|otp|2fa|verification code)\b",
    r"\b(phishing|identity theft|fraudulent)\b"
]

BILLING_DISPUTE_KEYWORDS = [
    r"\b(unauthorized charge|charged twice|double charged|scam|stole my money|dispute charge)\b",
    r"\b(fraudulent charge|bank statement|deducted without permission|credit card fraud)\b"
]

DAMAGED_ITEM_KEYWORDS = [
    r"\b(broken|shattered|smashed|cracked screen|damaged|faulty|missing parts|leaking)\b"
]

DELIVERY_DISPUTE_KEYWORDS = [
    r"\b(marked delivered|says delivered|handed to resident|stolen package|porch pirate)\b",
    r"\b(never arrived|where the hell|days late|lost package)\b"
]

URGENT_OR_HOSTILE_KEYWORDS = [
    r"\b(lawsuit|lawyer|legal action|sue you|better business bureau|bbb|attorney|police)\b",
    r"\b(incompetent|disgusting service|worst company|useless|scammers)\b"
]

class EscalationPolicyEngine:
    """
    Evaluates multi-signal criteria to determine if a ticket can be safely AUTO-HANDLED
    or must be ESCALATED TO HUMAN customer support specialists.
    """
    def __init__(
        self,
        intent_conf_threshold: float = INTENT_CONFIDENCE_THRESHOLD,
        evidence_score_threshold: float = EVIDENCE_SCORE_THRESHOLD
    ):
        self.intent_conf_threshold = intent_conf_threshold
        self.evidence_score_threshold = evidence_score_threshold
        
    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence_score: float,
        retrieved_evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates the message and returns the decision, rationale, and confidence score.
        """
        text_lower = customer_message.lower()
        
        # 1. High-risk urgent / legal / hostile customer language
        for pattern in URGENT_OR_HOSTILE_KEYWORDS:
            if re.search(pattern, text_lower):
                return {
                    "handling_decision": "ESCALATE",
                    "decision_reason": "High-urgency/hostility: Escalated due to severe customer distress or legal/reputational risk.",
                    "confidence": 0.95
                }
                
        # 2. Account Security & Privacy Concerns
        if predicted_intent == "account_and_security":
            return {
                "handling_decision": "ESCALATE",
                "decision_reason": "Security protocol: Account credentials, locked accounts, and authentication cannot be handled in public channels.",
                "confidence": 0.98
            }
        for pattern in SECURITY_KEYWORDS:
            if re.search(pattern, text_lower):
                return {
                    "handling_decision": "ESCALATE",
                    "decision_reason": "Security-sensitive keywords detected: Involves account access or authentication.",
                    "confidence": 0.95
                }
                
        # 3. Financial & Billing Disputes
        if predicted_intent == "billing_and_payment":
            for pattern in BILLING_DISPUTE_KEYWORDS:
                if re.search(pattern, text_lower):
                    return {
                        "handling_decision": "ESCALATE",
                        "decision_reason": "Billing dispute: Unauthorized or duplicate debit requires secure transaction lookup and payment verification.",
                        "confidence": 0.92
                    }
                    
        # 4. Damaged / Defective Physical Merchandise
        if predicted_intent == "damaged_defective_item":
            for pattern in DAMAGED_ITEM_KEYWORDS:
                if re.search(pattern, text_lower):
                    return {
                        "handling_decision": "ESCALATE",
                        "decision_reason": "Physical defect: Damaged goods require photo review and agent replacement authorization.",
                        "confidence": 0.90
                    }
            return {
                "handling_decision": "ESCALATE",
                "decision_reason": "Product quality complaint: Requires specialist review for return/exchange.",
                "confidence": 0.85
            }
            
        # 5. Delivery Dispute (Marked Delivered but Missing)
        if predicted_intent == "order_status_delivery":
            for pattern in DELIVERY_DISPUTE_KEYWORDS:
                if re.search(pattern, text_lower):
                    return {
                        "handling_decision": "ESCALATE",
                        "decision_reason": "Fulfillment failure: Package marked delivered but missing requires carrier tracer investigation.",
                        "confidence": 0.91
                    }
                    
        # 6. Low Intent Confidence Check
        if intent_confidence < self.intent_conf_threshold:
            return {
                "handling_decision": "ESCALATE",
                "decision_reason": f"Low intent confidence ({intent_confidence:.2f} < {self.intent_conf_threshold:.2f}): Customer inquiry is ambiguous.",
                "confidence": round(1.0 - intent_confidence, 2)
            }
            
        # 7. Low Evidence Score / Novel Issue Check
        if evidence_score < self.evidence_score_threshold:
            return {
                "handling_decision": "ESCALATE",
                "decision_reason": f"Insufficient historical precedent ({evidence_score:.2f} < {self.evidence_score_threshold:.2f}): No close match in support knowledge base.",
                "confidence": round(1.0 - evidence_score, 2)
            }
            
        # 8. Otherwise, safely AUTO-HANDLE with standard resolution guidance
        return {
            "handling_decision": "AUTO_HANDLE",
            "decision_reason": (
                f"Standard routine inquiry for '{predicted_intent}': High intent confidence ({intent_confidence:.2f}) "
                f"and strong historical resolution precedent ({evidence_score:.2f})."
            ),
            "confidence": round((intent_confidence + evidence_score) / 2.0, 2)
        }

if __name__ == "__main__":
    policy = EscalationPolicyEngine()
    
    # Test case 1: Standard tracking
    res1 = policy.evaluate(
        "Can you tell me how to track my order #123?",
        "order_status_delivery", 0.92, 0.78, []
    )
    print("Test 1 (Routine tracking):", res1)
    
    # Test case 2: Stolen / missing delivered package
    res2 = policy.evaluate(
        "Tracking says marked delivered to resident but my package was stolen off my porch!",
        "order_status_delivery", 0.88, 0.82, []
    )
    print("Test 2 (Missing package):", res2)
    
    # Test case 3: Ambiguous message
    res3 = policy.evaluate(
        "Hey",
        "order_status_delivery", 0.35, 0.20, []
    )
    print("Test 3 (Ambiguous):", res3)
