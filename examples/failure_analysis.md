# Comprehensive Failure Analysis: AI Customer Support Agent

This document analyzes real failure cases discovered during the systematic evaluation of our proposed AI Customer Support Agent on the 200-example Golden Evaluation Set. All examples, error counts, and logs are drawn directly from `examples/sample_predictions.json`.

---

## Overall Evaluation Summary

| Total Evaluated Examples | Intent Classification Errors | Intent Accuracy | Under-Escalations (Missed Risk) | Over-Escalations (Excessive Caution) | Escalation Recall |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **200** | **23** | **88.5%** | **3 (1.5%)** | **84 (42.0%)** | **95.3%** |

---

## Top 5 Failure Modes

### Failure Mode 1: Multi-Intent / Entangled Inquiries Triggering Category Misattribution

* **Example ID**: `GOLDEN_005`
* **Real Customer Message**:
  > *"please tell me how to sign in so my tv knows I have prime. It wants to charge me for all the prime movies that should be free"*
* **Ground Truth Intent**: `account_and_security` (or `digital_and_devices`)
* **Predicted Intent**: `subscription_and_prime` (Confidence: `0.74`)
* **Handling Decision**: `ESCALATE` (Triggered by: `Insufficient historical precedent (0.32 < 0.40)`)
* **What Went Wrong**:
  The message contains features of three distinct domains: device sign-in (`sign in so my tv knows`), subscription benefits (`prime movies that should be free`), and billing (`wants to charge me`). The bag-of-words/n-gram classifier locked onto the dense cluster of "prime", "charge", and "free", attributing it to subscription management rather than device sign-in.
* **Likely Cause**:
  Standard single-label multiclass classifiers assume mutually exclusive intents. Real-world customer queries frequently combine symptom, device, and subscription status in one run-on sentence.
* **Production Fix**:
  Implement hierarchical or multi-label intent classification, where an interaction can carry primary (`digital_and_devices`) and secondary (`subscription_and_prime`) tags, routing to a unified cross-functional troubleshooting flow.

---

### Failure Mode 2: Polysemous Keywords and Idiomatic Expressions

* **Example ID**: `GOLDEN_019`
* **Real Customer Message**:
  > *"I just checked my bank statements because I'm having issues with another company and then I found some BS unapproved charges from YOU guys ! What the heck amazon I loved you and now, trust broken."*
* **Ground Truth Intent**: `billing_and_payment`
* **Predicted Intent**: `damaged_defective_item` (Confidence: `0.50`)
* **Handling Decision**: `ESCALATE` (Triggered by: `Physical defect: Damaged goods require photo review`)
* **What Went Wrong**:
  The phrase *"trust broken"* caused the classifier and regex filters to latch onto "broken", biasing the prediction toward physical product damage (`damaged_defective_item`) despite the inquiry clearly revolving around *"unapproved charges"* on *"bank statements"*.
* **Likely Cause**:
  Surface-level lexical token matching is susceptible to idiomatic polysemy. Words like "broken", "cracked", or "dead" often describe trust, relationships, or smartphone batteries rather than physically shattered shipments.
* **Production Fix**:
  Transition from lexical n-gram TF-IDF to contextual transformer sentence embeddings (e.g. `sentence-transformers/all-MiniLM-L6-v2` or DeBERTa) that represent the semantic composition of the entire utterance rather than individual lexical trigger words.

---

### Failure Mode 3: Under-Escalation on Latent Legal / Chargeback Threats

* **Example ID**: `GOLDEN_200`
* **Real Customer Message**:
  > *"None of these options allow me to cancel the order due to delivery failure by . I will dispute the charges via the the credit card company instead."*
* **Ground Truth Handling**: `ESCALATE` (Card dispute / Chargeback threat)
* **Actual Handling Decision**: `AUTO_HANDLE`
* **System Reason**:
  > *"Standard routine inquiry for 'billing_and_payment': High intent confidence (0.80) and strong historical resolution precedent (0.43)."*
* **What Went Wrong**:
  The system recognized the message as `billing_and_payment` with high confidence ($0.80$) and found historical resolution examples for general billing questions ($0.43$). Because the customer mentioned "credit card" without explicit profanity, the system defaulted to auto-handling, missing the critical chargeback threat (*"dispute the charges via the credit card company"*).
* **Likely Cause**:
  The escalation policy looked for explicit dispute keywords like "unauthorized charge" or "double charged", but did not have an explicit rule for external bank chargebacks or credit card disputes threatened as a consequence of delivery failure.
* **Production Fix**:
  Add an explicit financial dispute rule triggering mandatory escalation whenever credit card chargebacks, merchant disputes, or consumer protection bureau filings are mentioned.

---

### Failure Mode 4: Over-Escalation Due to Conservative Evidence Thresholding

* **Example ID**: `GOLDEN_001`
* **Real Customer Message**:
  > *"For reference, this is what my invoice was supposed to bill to my card"*
* **Ground Truth Handling**: `AUTO_HANDLE` (Standard self-service invoice guidance)
* **Actual Handling Decision**: `ESCALATE`
* **System Reason**:
  > *"Insufficient historical precedent (0.36 < 0.40): No close match in support knowledge base."*
* **What Went Wrong**:
  The customer inquired about retrieving or clarifying an invoice. While this is a standard self-service request that could be satisfied by directing the user to 'Your Payments & Orders', the cosine similarity against the retrieval corpus fell just below the strict threshold ($0.36 < 0.40$), triggering an unnecessary escalation.
* **Likely Cause**:
  The knowledge base was sampled to 11,435 pairs to optimize memory and latency. Rare phrasings of routine tasks have lower cosine similarity in a reduced corpus, triggering the safety fallback.
* **Production Fix**:
  Calibrate the evidence threshold dynamically per intent. For low-risk intents (e.g. invoice downloads, tracking links), lower the evidence threshold from $0.40$ to $0.28$; keep it strictly high ($0.50$) only for refund and account access inquiries.

---

### Failure Mode 5: Misclassified Intent Saved by Secondary Escalation Safety Guardrail

* **Example ID**: `GOLDEN_002`
* **Real Customer Message**:
  > *"hi Amazon, I have just had an e mail to say I have changed my e mail address (I haven't)now I can't log in and change my details"*
* **Ground Truth Intent**: `account_and_security`
* **Predicted Intent**: `order_status_delivery` (Confidence: `0.37`)
* **Handling Decision**: `ESCALATE`
* **System Reason**:
  > *"Security-sensitive keywords detected: Involves account access or authentication."*
* **What Went Wrong with Intent**:
  The intent classifier was confused by the phrasing and gave a very low confidence score ($0.37$) to `order_status_delivery`.
* **Why the System Succeeded Operationally**:
  Even though the primary intent classifier failed, the multi-layered architecture prevented a catastrophic customer failure. The downstream **Escalation Engine** scanned for security keywords (`can't log in`, `change my details`, `e mail address changed`), intercepted the request, and escalated it immediately to human security specialists.
* **Takeaway**:
  This exemplifies the necessity of defense-in-depth in customer-facing AI agents: an intent classifier should never be the single point of failure.
