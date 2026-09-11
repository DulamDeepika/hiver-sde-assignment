"""
Customer Support Agent Orchestrator for AmazonHelp.
Glues together Intent Classifier, Historical Resolution Retriever,
Escalation Policy Engine, and Grounded Reply Generator into an end-to-end support system.
Returns standard structured JSON output.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.intent_classifier import IntentClassifier
from src.retriever import HistoricalResolutionRetriever
from src.reply_generator import GroundedReplyGenerator
from src.escalation import EscalationPolicyEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class CustomerSupportAgent:
    """
    End-to-End AI Customer Support Agent for Amazon customer inquiries.
    """
    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        retriever: Optional[HistoricalResolutionRetriever] = None,
        reply_generator: Optional[GroundedReplyGenerator] = None,
        escalation_engine: Optional[EscalationPolicyEngine] = None
    ):
        self.classifier = classifier or IntentClassifier()
        self.retriever = retriever or HistoricalResolutionRetriever()
        self.reply_generator = reply_generator or GroundedReplyGenerator()
        self.escalation_engine = escalation_engine or EscalationPolicyEngine()
        
        # Pre-warm models
        self._initialize()
        
    def _initialize(self):
        """Loads models into memory for fast inference."""
        logger.info("Initializing Agent models...")
        try:
            self.classifier.load()
        except Exception:
            logger.info("Classifier not found on disk, training...")
            self.classifier.train()
            
        try:
            self.retriever.load_index()
        except Exception:
            logger.info("Retriever index not found, building...")
            self.retriever.build_index()
            
        logger.info("Customer Support Agent successfully initialized.")
        
    def process_message(self, customer_message: str) -> Dict[str, Any]:
        """
        Executes the full pipeline for an incoming customer message:
        1. Classify intent & compute confidence
        2. Retrieve historical resolution evidence & evidence score
        3. Evaluate escalation policy
        4. Generate grounded reply
        """
        # Step 1: Intent Classification
        clf_res = self.classifier.predict(customer_message)
        predicted_intent = clf_res["predicted_intent"]
        intent_confidence = clf_res["confidence"]
        
        # Step 2: Historical Evidence Retrieval
        ret_res = self.retriever.retrieve(
            query=customer_message,
            predicted_intent=predicted_intent,
            top_k=3
        )
        retrieved_evidence = ret_res["retrieved_evidence"]
        evidence_score = ret_res["evidence_score"]
        
        # Step 3: Escalation Decision
        esc_res = self.escalation_engine.evaluate(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            intent_confidence=intent_confidence,
            evidence_score=evidence_score,
            retrieved_evidence=retrieved_evidence
        )
        handling_decision = esc_res["handling_decision"]
        decision_reason = esc_res["decision_reason"]
        
        # Step 4: Grounded Reply Generation
        draft_reply = self.reply_generator.generate_reply(
            customer_message=customer_message,
            predicted_intent=predicted_intent,
            retrieved_evidence=retrieved_evidence,
            is_escalated=(handling_decision == "ESCALATE"),
            escalation_reason=decision_reason
        )
        
        # Structured output schema matching assignment specification
        return {
            "customer_message": customer_message,
            "predicted_intent": predicted_intent,
            "intent_confidence": intent_confidence,
            "retrieved_evidence": retrieved_evidence,
            "draft_reply": draft_reply,
            "handling_decision": handling_decision,
            "decision_reason": decision_reason,
            "evidence_score": evidence_score
        }

if __name__ == "__main__":
    agent = CustomerSupportAgent()
    
    samples = [
        "Where is my package? The tracking says it was out for delivery 3 days ago.",
        "I was charged twice on my credit card for my Prime subscription!",
        "Can I return an opened box of headphones within the 30-day window?",
        "My Kindle paperwhite is frozen on the startup screen, how do I reboot it?",
        "My account got hacked and the password was changed, please help!"
    ]
    
    print("\n--- RUNNING AGENT ON SAMPLE INQUIRIES ---")
    for s in samples:
        res = agent.process_message(s)
        print("\n" + "="*70)
        print(f"CUSTOMER : {res['customer_message']}")
        print(f"INTENT   : {res['predicted_intent']} (Conf: {res['intent_confidence']})")
        print(f"DECISION : {res['handling_decision']} | Reason: {res['decision_reason']}")
        print(f"EVIDENCE : Score={res['evidence_score']} | Top items={len(res['retrieved_evidence'])}")
        print(f"REPLY    : {res['draft_reply']}")
