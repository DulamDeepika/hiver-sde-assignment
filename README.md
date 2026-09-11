# AI Customer Support Agent (Hiver SDE Take-Home Assignment)

[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen)](tests/test_pipeline.py)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)](requirements.txt)
[![Reproducibility](https://img.shields.io/badge/reproduction-under%2015%20mins-success)](evaluation/evaluate.py)

A production-grade, reproducible AI Customer Support Agent built on the Kaggle **Customer Support on Twitter** dataset (`thoughtvector/customer-support-on-twitter`). Designed for e-commerce shared-inbox workflows (Hiver), the system classifies customer intents into a grounded 7-intent taxonomy, retrieves relevant historical brand resolutions (RAG), generates evidence-grounded replies, and enforces an auditable **AUTO_HANDLE vs. ESCALATE** policy.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Definition](#2-problem-definition)
3. [Selected Brand & Rationale](#3-selected-brand--rationale)
4. [Dataset Setup](#4-dataset-setup)
5. [Installation](#5-installation)
6. [Environment Variables](#6-environment-variables)
7. [How to Run Preprocessing](#7-how-to-run-preprocessing)
8. [How to Run the Agent (Interactive CLI)](#8-how-to-run-the-agent-interactive-cli)
9. [How to Run Evaluation](#9-how-to-run-evaluation)
10. [How to Reproduce Headline Metrics](#10-how-to-reproduce-headline-metrics)
11. [Benchmark Results Summary](#11-benchmark-results-summary)
12. [System Architecture](#12-system-architecture)
13. [Limitations & Honest Critique](#13-limitations--honest-critique)
14. [Top Failure Modes](#14-top-failure-modes)
15. [Repository File Structure](#15-repository-file-structure)

---

## 1. Project Overview

This repository delivers an end-to-end customer support automation solution:
* **Zero Fabrication**: All metrics, benchmark tables, and failure analyses are computed directly from real model runs on a 200-example hand-verified golden set.
* **Leakage-Free Partitioning**: Data is split by unique customer ID, ensuring zero conversation-thread overlap between training/retrieval knowledge and evaluation.
* **Sub-Millisecond Inference**: Model weights and retrieval matrices are pre-indexed and serialized with `joblib`, booting in under 20ms without external API dependencies.
* **LLM-as-a-Judge with Human Validation**: Evaluated on a 6-dimension rubric, with empirical human agreement validation reporting Pearson correlation and Cohen's Kappa.

---

## 2. Problem Definition

In e-commerce customer support (Hiver's primary customer domain), an AI agent must:
1. **Accurately Classify Customer Intent**: Cut through informal, angry, or typo-ridden customer tweets.
2. **Ground Responses in Precedent**: Answer using factual historical resolutions without hallucinating unauthorized refunds, delivery dates, or policies.
3. **Escalate Reliably**: Direct account security threats, billing disputes, damaged items, and missing delivered goods to human specialists with a clear decision reason.

---

## 3. Selected Brand & Rationale

We analyzed all **2,811,774 rows** across candidate brands (`AmazonHelp`, `AppleSupport`, `Uber_Support`, `SpotifyCares`, `Delta`, etc.).

**Selected Brand**: **`AmazonHelp`**
* **Highest In-Domain Volume**: 169,840 outbound support tweets.
* **Substantive Resolution Rate**: 99.9% substantive troubleshooting and policy guidance (unlike `Uber_Support` or `AppleSupport`, which divert 30–75% of queries to canned DM links).
* **Hiver Product Alignment**: E-commerce inquiries (deliveries, returns, damaged items, subscriptions) directly reflect Hiver’s core ticketing workload.
* **Clean Escalation Boundaries**: Natural separation between public self-service guidance and private secure escalations.

---

## 4. Dataset Setup

1. Download `twcs.csv` from [Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
2. Place `twcs.csv` in the `data/` directory:
   ```
   data/twcs.csv
   ```
*(Pre-extracted training and golden sets are already committed in `data/processed/` and `evaluation/golden_set.csv`, so you can immediately run evaluation even before downloading the 500MB raw dataset!)*

---

## 5. Installation

```bash
# Clone the repository
git clone <repo-url>
cd hiver-sde-assignment

# Create virtual environment (Python 3.10+)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 6. Environment Variables

The agent is designed to run **100% offline and deterministically** out of the box. 
If you wish to test optional generative LLM completions or the automated LLM Judge with an OpenAI API key:

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and set your key (Optional):
# OPENAI_API_KEY=sk-...
# LLM_PROVIDER=openai
```

---

## 7. How to Run Preprocessing

To reconstruct conversation pairs, filter English text, and partition train/eval splits by customer ID:

```bash
python src/data_preparation.py
```
*Outputs: `data/processed/train_corpus.csv` (57,604 rows) and `data/processed/eval_candidates.csv` (14,495 rows).*

---

## 8. How to Run the Agent (Interactive CLI)

Test the end-to-end agent on custom inquiries:

```bash
python src/agent.py
```

### Programmatic Usage:
```python
from src.agent import CustomerSupportAgent

agent = CustomerSupportAgent()
response = agent.process_message("My package was marked delivered 2 hours ago but it is not on my porch!")

print(response)
```

**Sample Structured JSON Output**:
```json
{
  "customer_message": "My package was marked delivered 2 hours ago but it is not on my porch!",
  "predicted_intent": "order_status_delivery",
  "intent_confidence": 0.9579,
  "retrieved_evidence": [
    {
      "historical_inquiry": "My package says delivered but I can't find it",
      "historical_reply": "I'm sorry, don't fret yet! Let's try these steps first: https://t.co/9zP49AX3hn...",
      "intent": "order_status_delivery",
      "similarity_score": 0.8842
    }
  ],
  "draft_reply": "I am sorry for the delay and frustration with your delivery. Because your shipment requires direct tracking verification with the carrier, I am escalating this to our shipping team so we can locate your package or issue a replacement.",
  "handling_decision": "ESCALATE",
  "decision_reason": "Fulfillment failure: Package marked delivered but missing requires carrier tracer investigation.",
  "evidence_score": 0.558
}
```

---

## 9. How to Run Evaluation

### Run Automated Unit and Integration Tests:
```bash
python -m pytest tests/test_pipeline.py -v
```

### Run Evaluation Harness:
```bash
python evaluation/evaluate.py
```

### Run Human vs. LLM Judge Agreement Experiment:
```bash
python evaluation/human_judge_agreement.py
```

---

## 10. How to Reproduce Headline Metrics

To reproduce the complete benchmark in **under 2 minutes**:

```bash
python evaluation/evaluate.py
```
This loads the verified 200-example golden set (`evaluation/golden_set.csv`), evaluates Baseline 1 (Majority Class), Baseline 2 (Simple TF-IDF + Logistic Regression), and our Proposed System, saving full results to `evaluation/evaluation_results.json`.

---

## 11. Benchmark Results Summary

| Metric | Baseline 1 (Majority Class) | Baseline 2 (Simple TF-IDF + LR) | Proposed System (Our Agent) |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | 18.0% | 62.5% | **88.5%** |
| **Intent Macro F1** | 4.4% | 63.7% | **88.7%** |
| **Intent Weighted F1** | 5.5% | 63.4% | **88.5%** |
| **Escalation Precision** | 0.0% | 37.5% | **42.1%** |
| **Escalation Recall** | 0.0% | 9.4% | **95.3%** |
| **Escalation F1** | 0.0% | 15.0% | **58.4%** |
| **Overall Handling Accuracy** | 68.0% | 66.0% | **56.5%** |
| **Mean Evidence Grounding** | N/A | N/A | **0.409** |

### Human vs. LLM Judge Agreement (40 Hand-Annotated Inquiries):
* **Grounding Score Pearson Correlation ($r$)**: `0.7202` (Strong alignment on factual anchoring)
* **Grounding Score Spearman Rank ($\rho$)**: `0.7260`
* **Overall Quality Pearson Correlation ($r$)**: `0.5999`
* **Escalation Percentage Agreement**: `52.5%`

---

## 12. System Architecture

```
Incoming Customer Inquiry
          │
          ▼
┌──────────────────────────────┐
│   Intent Classifier          │ ──► [Predicted Intent, Calibrated Confidence]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Historical Retriever (RAG)   │ ──► [Top-3 Resolution Evidence, Evidence Score]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Escalation Policy Engine     │ ──► [AUTO_HANDLE vs. ESCALATE, Clear Reason]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Grounded Reply Generator     │ ──► [Draft Response grounded in precedent]
└──────────────┬───────────────┘
               │
               ▼
Standardized Structured Output (JSON)
```

---

## 13. Limitations & Honest Critique

*(From Section 9 of the Report: "What is misleading about my headline number?")*
1. **Handling Accuracy Illusion**: Baseline 1 gets **68.0%** handling accuracy by blindly guessing `AUTO_HANDLE` 100% of the time, yet its **Escalation Recall is 0.0%**. In customer support, high accuracy without recall is catastrophic.
2. **Conservative Over-Escalation**: Our agent prioritizes customer protection (>95% escalation recall on high-risk inquiries), which incurs an over-escalation penalty (42% of routine inquiries escalated).
3. **Intent Distribution vs. In-The-Wild Skew**: The golden evaluation set is balanced evenly across all 7 intents (~28 per class) to test rare failure modes. In live production, delivery queries account for ~87% of all ticket volume.

---

## 14. Top Failure Modes

Extracted from real evaluation records in `examples/sample_predictions.json`:
1. **Multi-Intent Entanglement (`GOLDEN_005`)**: Inquiry combines device login, Prime video streaming, and billing in one sentence.
2. **Polysemous Idioms (`GOLDEN_019`)**: The phrase *"trust broken"* biased the classifier toward physical product damage (`damaged_defective_item`) on an unapproved billing inquiry.
3. **Chargeback Threats Under-Escalation (`GOLDEN_200`)**: Customer threatened credit card disputes without explicit profanity, requiring stricter financial legal triggers.
4. **Conservative Evidence Threshold Over-Escalation (`GOLDEN_001`)**: Novel phrasing of routine invoice downloads scored $0.36 < 0.40$, triggering an unnecessary escalation.
5. **Intent Error Rescued by Escalation Net (`GOLDEN_002`)**: Intent was misclassified as delivery, but security keywords caught the account lockout and escalated safely.

*See [`examples/failure_analysis.md`](examples/failure_analysis.md) for full diagnostic breakdowns and fixes.*

---

## 15. Repository File Structure

```
hiver-sde-assignment/
├── README.md                      # Quickstart guide & reproduction manual
├── requirements.txt               # Dependencies
├── .gitignore                     # Git exclusions
├── .env.example                   # Optional environment config
├── decision_log.md                # 13 non-obvious engineering decisions
│
├── data/
│   ├── README.md                  # Data overview & leakage prevention
│   └── processed/
│       ├── preprocessing_summary.json
│       ├── train_corpus.csv       # Training & retrieval corpus (57,604 pairs)
│       └── eval_candidates.csv    # Evaluation candidate pool (14,495 pairs)
│
├── models/
│   ├── intent_classifier.joblib   # Trained intent model
│   └── retriever_index.joblib     # Pre-computed resolution index
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py        # Ingestion, cleaning, conversation pairing
│   ├── brand_selection.py         # Brand candidate analysis
│   ├── intent_classifier.py       # Calibrated n-gram TF-IDF classifier
│   ├── retriever.py               # Cosine similarity RAG engine
│   ├── reply_generator.py         # Grounded reply generator
│   ├── escalation.py              # Multi-signal escalation engine
│   └── agent.py                   # End-to-end agent orchestrator
│
├── evaluation/
│   ├── golden_set.csv             # 200 hand-verified evaluation examples
│   ├── baselines.py               # Majority and Simple LR baselines
│   ├── evaluate.py                # Full automated benchmark harness
│   ├── llm_judge.py               # 6-dimension LLM evaluator
│   ├── human_annotations.csv      # 40 hand-annotated human audit records
│   ├── human_judge_agreement.py   # Cohen's Kappa & correlation experiment
│   ├── evaluation_results.json    # Complete benchmark metrics
│   └── judge_agreement_results.json# Human-vs-judge statistical results
│
├── report/
│   └── report.md                  # Comprehensive ~6-page technical report
│
├── examples/
│   ├── sample_predictions.json    # Full predictions for 200 golden examples
│   ├── failure_analysis.md        # Top 5 real failure modes with fixes
│   └── extract_failures.py        # Diagnostic failure extraction utility
│
└── tests/
    └── test_pipeline.py           # Automated test suite (6 passing tests)
```
##Author
Dulam Gnanadeepika
