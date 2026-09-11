"""
Extracts real failure modes from evaluation predictions.
"""

import json

with open("examples/sample_predictions.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

intent_errors = [p for p in preds if p["predicted_intent"] != p["ground_truth_intent"]]
handling_errors = [p for p in preds if p["handling_decision"] != p["ground_truth_handling"]]

print(f"Total Intent Errors: {len(intent_errors)} / {len(preds)} ({len(intent_errors)/len(preds)*100:.1f}%)")
print(f"Total Handling Errors: {len(handling_errors)} / {len(preds)} ({len(handling_errors)/len(preds)*100:.1f}%)")

print("\n--- SAMPLE INTENT FAILURES ---")
for p in intent_errors[:5]:
    print(f"[{p['example_id']}] True: {p['ground_truth_intent']} | Pred: {p['predicted_intent']} (Conf: {p['intent_confidence']:.2f})")
    print(f"Text: \"{p['customer_message']}\"")
    print(f"Handling: {p['handling_decision']} | Reason: {p['decision_reason']}\n")

print("\n--- SAMPLE OVER-ESCALATIONS (True: AUTO_HANDLE, Pred: ESCALATE) ---")
over_esc = [p for p in handling_errors if p["ground_truth_handling"] == "AUTO_HANDLE" and p["handling_decision"] == "ESCALATE"]
print(f"Total Over-escalations: {len(over_esc)}")
for p in over_esc[:4]:
    print(f"[{p['example_id']}] Intent: {p['predicted_intent']}")
    print(f"Text: \"{p['customer_message']}\"")
    print(f"Reason: {p['decision_reason']}\n")

print("\n--- SAMPLE UNDER-ESCALATIONS (True: ESCALATE, Pred: AUTO_HANDLE) ---")
under_esc = [p for p in handling_errors if p["ground_truth_handling"] == "ESCALATE" and p["handling_decision"] == "AUTO_HANDLE"]
print(f"Total Under-escalations: {len(under_esc)}")
for p in under_esc[:4]:
    print(f"[{p['example_id']}] Intent: {p['predicted_intent']}")
    print(f"Text: \"{p['customer_message']}\"")
    print(f"Reason: {p['decision_reason']}\n")
