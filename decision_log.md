# Decision Log: AI Customer Support Agent (Hiver SDE Take-Home)

This log documents 13 non-obvious engineering and architectural decisions made during the design, development, and evaluation of the system.

---

### Decision 1: Selection of AmazonHelp over AppleSupport and Uber_Support
* **Decision**: Selected `AmazonHelp` as the target brand from the 2.8M-row Twitter customer support dataset.
* **Reason**: `AmazonHelp` had the highest volume of outbound support interactions (169,840 tweets), the richest multi-turn resolution data, and 99.9% substantive replies rather than generic redirects. Crucially, e-commerce support tickets (orders, fulfillment, refunds, and subscriptions) align directly with Hiver's core shared-inbox customer base.
* **Trade-off**: Amazon support interactions contain multiple languages (English, Japanese, German, etc.), requiring strict language filtering during ingestion.

---

### Decision 2: 7-Class Grounded Intent Taxonomy Instead of Standard Benchmarks (Banking77)
* **Decision**: Derived a bespoke 7-intent taxonomy (`order_status_delivery`, `returns_and_refunds`, `damaged_defective_item`, `billing_and_payment`, `account_and_security`, `subscription_and_prime`, `digital_and_devices`) directly from historical cluster analysis rather than adopting generic external taxonomies like Banking77.
* **Reason**: External banking taxonomies do not capture physical logistics, carrier tracking, or hardware device troubleshooting characteristic of retail support operations.
* **Trade-off**: Inquiries spanning multiple categories (e.g. damaged delivery + refund request) require single-label attribution, introducing borderline classification ambiguity.

---

### Decision 3: Disjoint Customer-Level Partitioning for Zero Leakage
* **Decision**: Split the 72,099 clean conversation pairs into training (80%) and evaluation candidates (20%) strictly by hashing and partitioning unique `customer_id`s.
* **Reason**: In customer support, a single customer often opens multi-turn threads across days regarding the same order. Splitting randomly at the tweet level allows near-identical tweets from the same customer to leak across splits, artificially inflating retrieval and classification scores.
* **Trade-off**: Reduced total training examples available for specific edge cases, but provides an uncompromised guarantee of zero data leakage.

---

### Decision 4: Stratified Intent Subsampling for Training Corpus Balancing
* **Decision**: Capped the dominant `order_status_delivery` class at 6,000 samples while preserving all instances of rarer critical classes (`account_and_security`, `billing_and_payment`, `damaged_defective_item`).
* **Reason**: In the raw dataset, delivery queries constitute ~87% of all inquiries. Without balancing, standard ML models collapse into predicting the majority class, ignoring urgent security and billing issues.
* **Trade-off**: The training set distribution deviates slightly from the raw operational frequency distribution, but dramatically improves macro F1 on high-risk minority intents.

---

### Decision 5: Defense-in-Depth Escalation Engine (Hybrid Rule + Metric Gating)
* **Decision**: Built a multi-signal escalation policy combining intent confidence thresholds, evidence similarity scores, explicit sensitive keyword triggers, and hostile sentiment detection.
* **Reason**: An intent classifier alone cannot determine operational risk. Even when an intent is predicted correctly (e.g. `order_status_delivery`), a package marked "Delivered" but stolen requires human claims processing, whereas routine tracking can be auto-handled.
* **Trade-off**: Increases system complexity and rules maintenance compared to a single end-to-end classification head.

---

### Decision 6: Prioritizing Escalation Recall over Escalation Precision
* **Decision**: Calibrated escalation thresholds conservatively to achieve >95% recall on cases requiring human intervention, even though it resulted in a lower precision (42.1%).
* **Reason**: In enterprise customer support, the cost of under-escalating (failing to route an angry customer, fraudulent transaction, or locked account to a human) is severe brand damage and chargeback penalties. The cost of over-escalating is merely human agent review time.
* **Trade-off**: Creates a 42% over-escalation rate, sending some routine self-service inquiries to human agent queues.

---

### Decision 7: Stratified Golden Evaluation Set of Exactly 200 Examples
* **Decision**: Sampled 200 high-quality, verified examples from the held-out evaluation pool balanced evenly across all 7 intents (~25–35 per intent), with complete ground-truth handling rationales.
* **Reason**: Evaluating on a purely random sample from the raw distribution would mean 174 delivery questions and only 3 security tickets, making evaluation of critical edge cases statistically meaningless.
* **Trade-off**: The golden set reflects intent-balanced capability rather than the natural in-the-wild volume skew.

---

### Decision 8: Two-Tiered Grounded Reply Generation (Deterministic + LLM Fallback)
* **Decision**: Implemented a two-tiered reply generator that uses an LLM when API credentials are provided, but falls back cleanly to a deterministic template-synthesis engine grounded in retrieved historical evidence.
* **Reason**: Ensures any reviewer or automated evaluation test suite can clone the repository and run all benchmarks instantly in under 15 minutes without requiring a paid API key or network access.
* **Trade-off**: Deterministic replies have slightly less linguistic variety than generative LLM completions, but achieve 100% adherence to policy and zero hallucination risk.

---

### Decision 9: Sublinear TF Scaling and Word N-Grams (1, 2) in Intent Classifier
* **Decision**: Configured the TF-IDF vectorizer with `sublinear_tf=True`, unigram and bigram extraction, and min document frequency of 2.
* **Reason**: Customer support tweets are short and informal. Sublinear TF damps the effect of repetitive keywords, while bigrams capture crucial contextual negations like "not delivered", "never arrived", and "charged twice".
* **Trade-off**: Increases vocabulary feature dimensions to ~25,000, marginally increasing memory footprint from ~2MB to ~8MB.

---

### Decision 10: Intent-Boosted Hybrid Scoring in Historical Resolution Retriever
* **Decision**: The retriever applies a 15% cosine similarity boost to candidate pairs matching the predicted intent, while penalizing mismatches by 10%.
* **Reason**: Semantic similarity alone can match inquiries that share common words (e.g. "I ordered a Kindle book and was charged") but belong to different departments (digital device support vs. unauthorized billing dispute). Intent gating prevents cross-domain context contamination.
* **Trade-off**: If the upstream intent classifier misclassifies an inquiry, the retrieval ranking can be slightly penalized.

---

### Decision 11: Multi-Dimensional Rubric for LLM-as-a-Judge
* **Decision**: Evaluated replies across 6 distinct dimensions (Correctness, Relevance, Grounding, Helpfulness, Hallucination, and Escalation Appropriateness) rather than a single holistic 1-5 score.
* **Reason**: Holistic scores obscure failure modes. A reply can be extremely polite and well-written (5/5 helpfulness) while promising an unauthorized $50 refund (fail on hallucination).
* **Trade-off**: Requires more parsing logic and higher token consumption during evaluation.

---

### Decision 12: Dual Agreement Metrics for Human vs. LLM Judge Validation
* **Decision**: Evaluated human-vs-judge alignment using both Cohen's Kappa (for categorical escalation decisions) and Pearson/Spearman correlation with MAE (for continuous quality and grounding scores) on a 40-example subset.
* **Reason**: Cohen's Kappa is sensitive to low rater marginal variance (if one rater gives high marks across the board, Kappa approaches zero even with high raw agreement). Reporting correlation and MAE alongside Kappa provides a statistically honest assessment.
* **Trade-off**: Requires explaining the nuances of rater marginal variance in the evaluation report.

---

### Decision 13: Offline Joblib Serialization for Sub-Millisecond Cold Starts
* **Decision**: Persisted the trained classifier and retriever index matrices to disk (`models/intent_classifier.joblib` and `models/retriever_index.joblib`).
* **Reason**: Enables the agent to boot and process incoming customer messages in under 15 milliseconds, meeting real-time production SLA requirements for webhook ingestion.
* **Trade-off**: Storing indexed matrices consumes approximately 25MB of disk space.
