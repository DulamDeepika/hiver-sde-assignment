"""
Automated Pipeline Test Suite for Hiver Customer Support Agent.
Tests preprocessing, intent classification, retrieval, escalation policy,
full agent orchestrator output format, and golden set integrity.
"""

import os
import sys
import pytest
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preparation import clean_customer_text, clean_reply_text, is_valid_english
from src.intent_classifier import IntentClassifier, TAXONOMY_INTENTS
from src.retriever import HistoricalResolutionRetriever
from src.escalation import EscalationPolicyEngine
from src.agent import CustomerSupportAgent

GOLDEN_SET_PATH = os.path.join(PROJECT_ROOT, "evaluation", "golden_set.csv")

def test_text_cleaning():
    raw_cust = "@AmazonHelp where is my order? https://t.co/abc1234 &amp; when will it arrive?"
    cleaned = clean_customer_text(raw_cust)
    assert "@AmazonHelp" not in cleaned
    assert "https://" not in cleaned
    assert "&amp;" not in cleaned
    assert "where is my order" in cleaned
    
    raw_reply = "@115820 Hi there! We'd be glad to help with your delivery. ^AG"
    cleaned_reply = clean_reply_text(raw_reply)
    assert "@115820" not in cleaned_reply
    assert not cleaned_reply.endswith("^AG")
    assert "Hi there!" in cleaned_reply

def test_intent_classifier():
    clf = IntentClassifier()
    clf.load()
    
    res = clf.predict("Where is my package? It is 3 days late.")
    assert "predicted_intent" in res
    assert res["predicted_intent"] in TAXONOMY_INTENTS
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == len(TAXONOMY_INTENTS)
    
    # Check security intent prediction
    res_sec = clf.predict("My account was locked and I need to reset my password")
    assert res_sec["predicted_intent"] == "account_and_security"
    assert res_sec["confidence"] > 0.70

def test_retriever():
    retriever = HistoricalResolutionRetriever()
    retriever.load_index()
    
    res = retriever.retrieve("My package was stolen from my doorstep", predicted_intent="order_status_delivery", top_k=3)
    assert "retrieved_evidence" in res
    assert "evidence_score" in res
    assert len(res["retrieved_evidence"]) == 3
    assert 0.0 <= res["evidence_score"] <= 1.0
    assert "historical_reply" in res["retrieved_evidence"][0]

def test_escalation_engine():
    policy = EscalationPolicyEngine()
    
    # Test high-risk security
    sec_esc = policy.evaluate(
        customer_message="My account got hacked and someone changed my email address",
        predicted_intent="account_and_security",
        intent_confidence=0.98,
        evidence_score=0.60,
        retrieved_evidence=[]
    )
    assert sec_esc["handling_decision"] == "ESCALATE"
    assert "Security" in sec_esc["decision_reason"]
    
    # Test high-risk billing
    bill_esc = policy.evaluate(
        customer_message="You charged my card twice for the same order, this is an unauthorized charge!",
        predicted_intent="billing_and_payment",
        intent_confidence=0.95,
        evidence_score=0.55,
        retrieved_evidence=[]
    )
    assert bill_esc["handling_decision"] == "ESCALATE"
    assert "Billing dispute" in bill_esc["decision_reason"]
    
    # Test routine tracking auto-handle
    track_auto = policy.evaluate(
        customer_message="How can I track my standard shipment?",
        predicted_intent="order_status_delivery",
        intent_confidence=0.92,
        evidence_score=0.75,
        retrieved_evidence=[]
    )
    assert track_auto["handling_decision"] == "AUTO_HANDLE"

def test_agent_orchestrator_schema():
    agent = CustomerSupportAgent()
    output = agent.process_message("How do I return a pair of shoes that are too small?")
    
    required_keys = [
        "customer_message",
        "predicted_intent",
        "intent_confidence",
        "retrieved_evidence",
        "draft_reply",
        "handling_decision",
        "decision_reason",
        "evidence_score"
    ]
    for key in required_keys:
        assert key in output, f"Missing required output key: {key}"
        
    assert output["predicted_intent"] == "returns_and_refunds"
    assert output["handling_decision"] in ["AUTO_HANDLE", "ESCALATE"]
    assert isinstance(output["draft_reply"], str)
    assert len(output["draft_reply"]) > 20

def test_golden_evaluation_set_integrity():
    assert os.path.exists(GOLDEN_SET_PATH), "Golden evaluation set file not found!"
    df = pd.read_csv(GOLDEN_SET_PATH)
    assert len(df) == 200, f"Golden set should have exactly 200 examples, found {len(df)}"
    
    # Verify required fields
    expected_cols = [
        "example_id", "customer_message", "intent",
        "expected_handling", "handling_reason", "reference_reply"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column in golden set: {col}"
        assert df[col].isnull().sum() == 0, f"Null values found in column: {col}"
        
    # Verify all 7 intents are present and well-represented
    intent_counts = df["intent"].value_counts()
    assert len(intent_counts) == 7
    for count in intent_counts.values:
        assert count >= 20, f"Intent underrepresented in golden set: count={count}"
        
    # Verify handling decisions
    assert set(df["expected_handling"].unique()) == {"AUTO_HANDLE", "ESCALATE"}
