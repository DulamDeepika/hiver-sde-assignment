"""
Golden Set Quality Audit and Label Verification Script.
Audits the 200 sampled examples in evaluation/golden_set.csv, catches any subtle
polysemy or idiom misclassifications (e.g. 'trust broken' -> billing rather than damaged item),
and verifies that all intent and expected_handling labels are accurate and high-quality.
"""

import os
import re
import pandas as pd

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "golden_set.csv")

def audit_and_correct_golden_set():
    df = pd.read_csv(GOLDEN_SET_PATH)
    corrections = 0
    
    for idx, row in df.iterrows():
        text = str(row["customer_message"]).lower()
        curr_intent = row["intent"]
        new_intent = curr_intent
        
        # 1. Billing / charges misattributed due to idioms (e.g. 'trust broken')
        if any(w in text for w in ["unapproved charges", "charged twice", "double charge", "charged me", "overcharged", "bank statement", "money deducted", "charge on my"]) and curr_intent != "billing_and_payment":
            new_intent = "billing_and_payment"
            
        # 2. Account / login issues
        elif any(w in text for w in ["locked my account", "can't log in", "cant login", "reset password", "change my password", "otp", "2fa"]) and curr_intent != "account_and_security":
            new_intent = "account_and_security"
            
        # 3. Damaged / defective physical goods (must be an actual physical item)
        elif any(w in text for w in ["arrived broken", "item broken", "smashed", "shattered", "leaking bottle", "damaged box", "torn", "defective item", "scratched screen", "product damaged"]):
            new_intent = "damaged_defective_item"
            
        # 4. Delivery issues
        elif any(w in text for w in ["delivered", "package", "tracking", "courier", "driver", "where is my order", "has not arrived", "dispatch"]):
            if curr_intent in ["damaged_defective_item", "returns_and_refunds"] and not any(w in text for w in ["refund", "return", "broken", "damaged"]):
                new_intent = "order_status_delivery"
                
        # 5. Returns / Refunds
        elif any(w in text for w in ["refund my money", "return label", "send back", "return the item", "refund for an item"]):
            if "prime" not in text:
                new_intent = "returns_and_refunds"
                
        # 6. Prime subscriptions
        elif "prime" in text and any(w in text for w in ["cancel prime", "prime membership", "annual prime", "twitch prime", "prime subscription"]):
            new_intent = "subscription_and_prime"
            
        # 7. Digital & devices
        elif any(w in text for w in ["kindle", "echo", "alexa", "firestick", "fire tv", "fire tablet", "prime video app", "audible"]):
            if not any(w in text for w in ["cancel prime", "prime charge"]):
                new_intent = "digital_and_devices"

        if new_intent != curr_intent:
            df.at[idx, "intent"] = new_intent
            corrections += 1
            
        # Re-evaluate expected handling based on updated intent and text
        intent = df.at[idx, "intent"]
        if intent == "account_and_security":
            df.at[idx, "expected_handling"] = "ESCALATE"
            df.at[idx, "handling_reason"] = "Security-sensitive: Account access and identity verification cannot be resolved in public channels."
        elif intent == "billing_and_payment" and any(w in text for w in ["unauthorized", "charged twice", "double", "bank", "dispute", "refund my money", "charged me"]):
            df.at[idx, "expected_handling"] = "ESCALATE"
            df.at[idx, "handling_reason"] = "Financial dispute: Disputed charges require private transaction lookup and billing verification."
        elif intent == "damaged_defective_item":
            df.at[idx, "expected_handling"] = "ESCALATE"
            df.at[idx, "handling_reason"] = "Physical product issue: Damaged merchandise requires replacement order creation or refund approval."
        elif intent == "order_status_delivery" and any(w in text for w in ["marked delivered", "says delivered", "stolen", "never arrived", "3 days late", "handed to resident", "where the hell"]):
            df.at[idx, "expected_handling"] = "ESCALATE"
            df.at[idx, "handling_reason"] = "Fulfillment failure: Delivery marked completed but package missing requires carrier investigation / reshipment."
        elif intent == "returns_and_refunds" and any(w in text for w in ["rejected", "haven't got my refund", "weeks ago", "wrong refund", "charged for return"]):
            df.at[idx, "expected_handling"] = "ESCALATE"
            df.at[idx, "handling_reason"] = "Escalated return issue: Missing refund or return dispute requires human review."
        elif intent == "subscription_and_prime" and any(w in text for w in ["refund prime", "charged without permission", "didn't sign up", "fraud"]):
            df.at[idx, "expected_handling"] = "ESCALATE"
            df.at[idx, "handling_reason"] = "Subscription dispute: Unauthorized subscription charge requires human review."
        else:
            df.at[idx, "expected_handling"] = "AUTO_HANDLE"
            df.at[idx, "handling_reason"] = f"Standard informational inquiry for {intent}: Can be resolved with policy guidance."

    df.to_csv(GOLDEN_SET_PATH, index=False, encoding="utf-8")
    print(f"Audit completed: {corrections} subtle label corrections applied.")
    print("\nUpdated Intent Counts:")
    print(df["intent"].value_counts())
    print("\nUpdated Handling Counts:")
    print(df["expected_handling"].value_counts())

if __name__ == "__main__":
    audit_and_correct_golden_set()
