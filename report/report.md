# Engineering & Evaluation Report: AI Customer Support Agent

**Author**: Senior AI/ML Engineering Candidate  
**Target Organization**: Hiver (Customer Operations & Shared Inbox Automation)  
**Dataset**: Kaggle Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`)  
**Selected Brand**: Amazon Help (`AmazonHelp`)  
**Evaluation Set**: 200 Hand-Verified Golden Examples (Zero-Leakage Split)  

---

## 1. Problem Framing

### What "Good" Means for Amazon Support Automation
In high-volume e-commerce customer support, an AI agent must optimize for two competing objectives: **containment of routine inquiries** and **safe routing of sensitive exceptions**. 
For `AmazonHelp`, "good" means:
1. **Accurate Intent Triage**: Reliably categorizing informal, messy, and emotion-laden customer inquiries into actionable domain categories.
2. **Strict Grounding in Precedent**: Formulating responses that strictly adhere to official policies and historical resolution patterns without hallucinating unauthorized refunds, delivery dates, or arbitrary commitments.
3. **Conservative, Auditable Escalation**: Ensuring that cases involving financial disputes, compromised credentials, damaged physical goods, and severe customer distress are reliably escalated to human operators, accompanied by an explicit, human-readable rationale.

### What We Chose to Build
* A modular, sub-millisecond AI agent combining a calibrated multi-class intent classifier, a historical resolution retrieval engine (RAG), a grounded reply generator, and an explicit multi-signal escalation policy engine.
* A leakage-free preprocessing and conversation-stitching pipeline that partitions data strictly by unique customer ID.
* A comprehensive automated evaluation harness comparing the proposed system against two classical baselines across accuracy, macro F1, escalation recall/precision, and evidence grounding.
* An LLM-as-a-judge evaluation module alongside a human-annotated agreement experiment reporting Cohen's Kappa, correlation, and percentage agreement.

### What We Deliberately Chose NOT to Build
* **Autonomous Financial Concession Disbursement**: The agent is explicitly prohibited from issuing refunds, adjusting bank balances, or modifying orders autonomously over public social channels without verified internal authentication.
* **Unbounded Open-Ended Generative Chatbots**: We deliberately avoided unconstrained LLM generators that attempt to invent solutions on the fly, enforcing strict retrieval-augmented constraints.
* **Black-Box End-to-End Classifiers**: We avoided single end-to-end black-box models for escalation decisions, choosing a transparent, auditable policy engine that evaluates intent confidence, risk keywords, and evidence similarity separately.

---

## 2. Dataset & Methodology

### Dataset Overview & Ingestion
The Kaggle Twitter Customer Support dataset contains approximately 2.81 million rows (`twcs.csv`). To avoid out-of-memory bottlenecks, we implemented a chunked streaming ingestion pipeline (`src/data_preparation.py`) processing 100,000 rows per batch.

### Brand Selection Justification
Analysis of all 2.81M rows identified 12 major candidate brands:
* `AmazonHelp`: 169,840 outbound tweets (99.9% substantive replies)
* `AppleSupport`: 106,860 outbound tweets (30.2% pure DM redirects)
* `Uber_Support`: 56,270 outbound tweets (heavy canned URL redirects to web forms)
* `SpotifyCares`: 43,265 outbound tweets (narrow digital music streaming focus)
* `Delta / AmericanAir`: 42,253 / 36,764 tweets (heavy reliance on live proprietary airline reservation APIs)

`AmazonHelp` was selected because:
1. **Domain Alignment**: E-commerce fulfillment, billing disputes, returns, and delivery inquiries mirror Hiver’s core enterprise customer base.
2. **Data Abundance**: With 169,840 brand replies, there is sufficient volume to build rich retrieval knowledge bases and balanced evaluation benchmarks without sparsity.
3. **Clear Auto-Handle vs. Escalation Boundary**: Distinguishes cleanly between self-service policies (tracking, return procedures, Prime settings) and private human interventions (stolen deliveries, credit card disputes, account lockouts).

### Conversation Reconstruction & Cleaning
Customer parent tweets were matched to `AmazonHelp` responses via `in_response_to_tweet_id`. The text was cleaned by:
* Stripping Twitter handle mentions (`@AmazonHelp`, `@115820`).
* Removing trailing agent signature initials (`^AG`, `^KL`, `^DW`).
* Filtering out non-English customer interactions (multilingual tweets in Japanese, German, French, etc., which Amazon handles on international domains).
* Normalizing URLs and whitespace.

### Strict Leakage Prevention Strategy
A critical flaw in standard NLP benchmarks is random row-level splitting, which allows multi-turn tweets from the same customer or identical duplicate inquiries to appear in both training and test sets. 
To prevent leakage:
* We grouped all clean conversation pairs (72,099 total) by `customer_id`.
* We performed a strict disjoint hash partition: **80% Train/Retrieval Corpus** (57,604 rows; 26,342 unique customers) and **20% Evaluation Candidate Pool** (14,495 rows; 6,586 unique customers).
* Overlap between train and eval customer sets was mathematically verified to be **exactly 0**.

---

## 3. System Architecture

The pipeline processes each incoming customer message through a sequential, decoupled architecture:

```
                      Incoming Customer Message
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    Intent Classifier     │
                    │  (TF-IDF + Calibrated LR)│
                    └────────────┬─────────────┘
                                 │ Predicted Intent + Confidence Score
                                 ▼
                    ┌──────────────────────────┐
                    │   Resolution Retriever   │
                    │   (Cosine Sim + RAG)     │
                    └────────────┬─────────────┘
                                 │ Top-3 Historical Evidence + Evidence Score
                                 ▼
                    ┌──────────────────────────┐
                    │ Escalation Policy Engine │
                    │ (Risk Rules + Thresholds)│
                    └────────────┬─────────────┘
                                 │ Handling Decision (AUTO_HANDLE vs ESCALATE) + Reason
                                 ▼
                    ┌──────────────────────────┐
                    │ Grounded Reply Generator │
                    │ (Evidence-Grounded RAG)  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                     Standardized JSON Output
```

### Architectural Components:
1. **Intent Classifier (`src/intent_classifier.py`)**: Computes sublinear unigram/bigram TF-IDF vectors, classifying into 7 intents via class-balanced Logistic Regression with calibrated probability outputs (`confidence` $\in [0, 1]$).
2. **Historical Resolution Retriever (`src/retriever.py`)**: Maintains a persistent index of 11,435 balanced historical resolution pairs. Computes query similarity with a 15% boost for intent-aligned records, yielding an aggregate `evidence_score`.
3. **Escalation Policy Engine (`src/escalation.py`)**: Deterministically evaluates multi-signal criteria:
   * Intent confidence $< 0.65 \rightarrow$ `ESCALATE` (ambiguity).
   * Evidence score $< 0.40 \rightarrow$ `ESCALATE` (insufficient precedent).
   * Sensitive domains (Account Lockouts, Disputed Charges, Damaged Merchandise, Missing Delivered Goods) $\rightarrow$ `ESCALATE`.
   * Hostile / legal / threatening sentiment $\rightarrow$ `ESCALATE`.
   * Otherwise $\rightarrow$ `AUTO_HANDLE`.
4. **Grounded Reply Generator (`src/reply_generator.py`)**: Drafts empathetic, concise replies anchored in the retrieved historical resolution evidence and official Amazon operational guidance. Contains deterministic generation with optional LLM API synthesis.

---

## 4. Intent Taxonomy

Derived from recurring cluster patterns in actual historical `AmazonHelp` data:

| Intent Name | Operational Scope & Definition | Trigger Keywords / Patterns | Real Customer Example from Data | Handling Target |
| :--- | :--- | :--- | :--- | :--- |
| `order_status_delivery` | Tracking inquiries, transit delays, carrier updates, marked delivered but missing. | `where is my order`, `tracking`, `carrier`, `marked delivered`, `parcel` | *"Item has not been delivered but tracking says it was handed to me over an hour ago... sort it out"* | Routine tracking: Auto-handle; Missing delivered: Escalate. |
| `returns_and_refunds` | Return procedures, prepaid return labels, refund status, return drop-off locations. | `return label`, `send back`, `refund status`, `drop off`, `exchange` | *"How do I return a damaged item I bought last week? Where is my return label?"* | Policy guidance: Auto-handle; Missing refund disputes: Escalate. |
| `damaged_defective_item` | Physical merchandise arrived broken, shattered, leaking, scratched, or missing parts. | `broken`, `shattered`, `cracked screen`, `defective`, `leaking`, `missing pieces` | *"Ordered two display cases... both arrived broken in multiple places."* | Escalate to human specialist for replacement authorization. |
| `billing_and_payment` | Unauthorized charges, duplicate debiting, declined cards, payment method issues. | `charged twice`, `unauthorized charge`, `bank statement`, `credit card`, `invoice` | *"hi, I have just checked my bank account and have noticed I haven't been refunded for an item I sent back"* | Financial dispute: Escalate to secure billing support. |
| `account_and_security` | Password resets, 2FA/OTP failures, locked accounts, compromised credentials. | `locked my account`, `can't log in`, `reset password`, `otp`, `2fa`, `hacked` | *"the link you sent want me to sign in, but they locked my account"* | Mandatory Escalation to secure authentication channels. |
| `subscription_and_prime` | Prime membership fees, benefits, renewal cancellations, student discounts. | `prime membership`, `cancel prime`, `auto-renew`, `prime fee`, `twitch prime` | *"Why are Amazon prime packages delivered in such a shady way? Like, hello unmarked white van"* | Self-service cancellation: Auto-handle; Fee disputes: Escalate. |
| `digital_and_devices` | Kindle, Fire TV, Echo/Alexa, Prime Video streaming bugs, app crashes. | `kindle`, `firestick`, `echo`, `alexa`, `app crash`, `screen frozen` | *"I'm unable to control the volume while using the FireStick, it affects normal TV functions"* | Troubleshooting steps: Auto-handle; Hardware defects: Escalate. |

---

## 5. Evaluation Methodology

### Golden Evaluation Set (200 Examples)
From the held-out 14,495 evaluation candidates, we sampled a balanced, curated benchmark of **200 examples** (`evaluation/golden_set.csv`):
* Stratified distribution across all 7 intents (~23 to 36 examples per intent).
* Filtered strictly for clean English, removing truncated fragments.
* Hand-audited and annotated with:
  * `customer_message`
  * `intent` (ground-truth)
  * `expected_handling` (`AUTO_HANDLE`: 136 / 68.0%, `ESCALATE`: 64 / 32.0%)
  * `handling_reason` (operational justification)
  * `reference_reply` (historical agent resolution)
  * `source_tweet_id` and `customer_id`

### Automated Metrics
* **Intent Classification**: Accuracy, Macro F1, Macro Precision, Macro Recall, Weighted F1, and complete 7x7 Confusion Matrix.
* **Escalation Decisions**: Escalation Precision, Escalation Recall, Escalation F1, and Overall Handling Accuracy.
* **Evidence Grounding**: Mean and median cosine similarity scores across retrieved historical pairs.

---

## 6. Baselines

We compared the proposed agent against two standard baselines evaluated on the exact same 200-example golden set:
1. **Baseline 1: Majority Class Predictor**
   * Predicts `order_status_delivery` (the empirical majority class in e-commerce support) for all inputs.
   * Defaults handling decision to `AUTO_HANDLE`.
2. **Baseline 2: Simple TF-IDF + Logistic Regression**
   * Unigram TF-IDF vectorizer (max 5,000 features, default regularization $C=1.0$, unweighted classes).
   * Default escalation heuristic: if maximum class probability $< 0.50 \rightarrow$ `ESCALATE`, else `AUTO_HANDLE`.

---

## 7. Experimental Results

The quantitative benchmark results produced directly by `evaluation/evaluate.py`:

| Metric | Baseline 1 (Majority Class) | Baseline 2 (Simple TF-IDF + LR) | Proposed System (Full Agent) |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | 18.0% | 62.5% | **88.5%** |
| **Intent Macro F1** | 4.4% | 63.7% | **88.7%** |
| **Intent Weighted F1** | 5.5% | 63.4% | **88.5%** |
| **Escalation Precision** | 0.0% | 37.5% | **42.1%** |
| **Escalation Recall** | 0.0% | 9.4% | **95.3%** |
| **Escalation F1** | 0.0% | 15.0% | **58.4%** |
| **Overall Handling Accuracy** | 68.0% | 66.0% | **56.5%** |
| **Mean Evidence Grounding** | N/A | N/A | **0.409** |

### Per-Intent Performance of Proposed System:
* `order_status_delivery`: Precision 0.88, Recall 0.83, F1 0.86 (Support: 36)
* `returns_and_refunds`: Precision 0.84, Recall 0.90, F1 0.87 (Support: 30)
* `damaged_defective_item`: Precision 0.77, Recall 0.87, F1 0.82 (Support: 23)
* `billing_and_payment`: Precision 0.90, Recall 0.96, F1 0.93 (Support: 28)
* `account_and_security`: Precision 0.96, Recall 0.96, F1 0.96 (Support: 28)
* `subscription_and_prime`: Precision 0.92, Recall 0.85, F1 0.89 (Support: 27)
* `digital_and_devices`: Precision 0.93, Recall 0.89, F1 0.91 (Support: 28)

### Key Takeaways:
1. **Dramatic Intent Gain**: The proposed system achieves **88.5% Accuracy** and **88.7% Macro F1**, outperforming Baseline 2 by **+26.0 percentage points** due to sublinear n-gram feature scaling and balanced class weighting.
2. **Escalation Safety Net**: Baseline 2 exhibited catastrophic under-escalation, catching only 9.4% of high-risk cases (missing 90% of security lockouts, fraud, and stolen packages). The proposed agent achieved **95.3% Escalation Recall**, successfully capturing 61 of the 64 escalation-worthy inquiries.

---

## 8. LLM-as-a-Judge & Human Agreement Experiment

We implemented an LLM Judge (`evaluation/llm_judge.py`) evaluating interactions across 6 rubric dimensions (Correctness, Relevance, Grounding, Helpfulness, Hallucination, and Escalation Appropriateness).

To provide empirical proof of judge reliability, we conducted an agreement experiment on a 40-example subset hand-annotated by a human auditor (`evaluation/human_judge_agreement.py`):

| Evaluation Dimension / Metric | Experimental Result | Interpretation |
| :--- | :---: | :--- |
| **Sample Size** | **40 Hand-Annotated Inquiries** | Stratified across all 7 intents |
| **Grounding Score Pearson Correlation ($r$)** | **0.7202** | Strong positive correlation with human grounding ratings |
| **Grounding Score Spearman Rank ($\rho$)** | **0.7260** | Strong monotonic ranking consistency |
| **Grounding Mean Absolute Error (MAE)** | **0.4250** | Average difference $< 0.5$ on a 1–5 rubric scale |
| **Overall Quality Pearson Correlation ($r$)** | **0.5999** | Moderate-to-strong positive alignment |
| **Overall Quality Mean Absolute Error (MAE)** | **0.2750** | Average difference $< 0.3$ on a 1–5 scale |
| **Escalation Appropriateness Agreement %** | **52.5%** | Raw percentage agreement on escalation correctness |
| **Escalation Appropriateness Cohen's Kappa ($\kappa$)**| **0.0000** | Artifact of low rater marginal variance (explained below) |

---

## 9. What is Misleading About My Headline Number? (Mandatory Critique)

As senior AI engineers, we must rigorously interrogate our own headline metrics:

1. **The Handling Accuracy Illusion (68.0% vs 56.5%)**:
   Baseline 1 achieves a deceptively high Handling Accuracy of **68.0%** simply by predicting `AUTO_HANDLE` for every single customer ticket. Because the golden set contains 68% auto-handle cases, a broken model that never escalates appears "better" on raw accuracy than our proposed system (56.5%). However, Baseline 1 has **0.0% Escalation Recall**, exposing the brand to immense liability.
2. **Over-Escalation Penalty (42.0% False Positive Rate)**:
   Our system achieved **95.3% Escalation Recall**, but only **42.1% Escalation Precision**. To protect customers against account theft and stolen deliveries, our escalation policy was configured with conservative safety thresholds ($conf < 0.65$ or $evidence < 0.40$). Consequently, 84 out of 136 routine requests were escalated. In production, this would inflate human support ticket queues by ~40%.
3. **The Cohen's Kappa Paradox ($\kappa = 0.00$)**:
   In the 40-example judge agreement experiment, the human rater judged all 40 system decisions as defensible (positive rate = 100%), while the automated judge was stricter (positive rate = 52.5%). Mathematically, Cohen's Kappa is defined as $\frac{P_o - P_e}{1 - P_e}$. When one rater has zero variance (all 1s), the expected chance agreement $P_e$ equals the observed agreement $P_o$, forcing $\kappa$ to 0.00 despite 52.5% raw agreement. Relying on Kappa alone would misleadingly suggest zero agreement, whereas continuous metrics demonstrate strong alignment ($r = 0.72$ on Grounding).
4. **Distribution Shift Between Golden Set and In-The-Wild Volume**:
   Our 200-example golden set is balanced across intents (~28 per intent) to ensure rigorous testing of rare intents like `account_and_security` (3.7% in the wild) and `billing_and_payment` (4.1% in the wild). In real-world deployment, `order_status_delivery` accounts for 87% of all volume. Real-world aggregate accuracy will be heavily dominated by delivery tracking nuances rather than cross-domain separation.
5. **Lexical Retrieval vs. Semantic Understanding**:
   The retriever achieved a mean evidence score of $0.409$. Because it relies on TF-IDF cosine similarity, inquiries with novel customer phrasing (e.g. *"item not in mailbox though marked left on doorstep"*) score lower ($< 0.40$) even when conceptually routine, needlessly triggering the escalation fallback.

---

## 10. What I Would Do With One More Week

With an additional week of engineering time, I would implement the following targeted enhancements:

1. **Dense Vector Retrieval (FAISS + Sentence Transformers)**:
   Replace n-gram TF-IDF retrieval with a fine-tuned `sentence-transformers/all-MiniLM-L6-v2` bi-encoder indexed in FAISS. This would eliminate lexical mismatch failures (e.g. mapping "porch pirate" directly to "stolen package").
2. **Multi-Label / Hierarchical Intent Classification**:
   Transition from single-label multiclass to a hierarchical multi-label architecture (e.g. Category: `digital_devices` $\rightarrow$ Sub-intent: `firestick_app_lag`). This resolves multi-intent inquiries where customers report device and subscription symptoms simultaneously.
3. **Intent-Dynamic Escalation Thresholding**:
   Replace static thresholds ($0.65$ confidence, $0.40$ evidence) with per-intent dynamic thresholds. Low-risk categories (e.g. `order_status_delivery` FAQs) would tolerate lower evidence ($0.25$) before escalating, reducing the over-escalation rate from 42% to $< 15\%$.
4. **Multi-Turn Context Tracking**:
   Reconstruct and feed previous conversation turns into the agent prompt. Current Twitter customer support often spans 3-4 tweets; evaluating each tweet in isolation causes unnecessary re-asking of details already provided in turn 1.
5. **Active Learning & Production Human-in-the-Loop Workflow**:
   Integrate directly with Hiver's shared inbox webhooks: when the agent escalates with a specific `decision_reason`, it pre-drafts the response as a private note for the human agent, collecting human corrections to continuously retrain the classifier.
