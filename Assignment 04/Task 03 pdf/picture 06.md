# Model Card — Network Intrusion Detection Model

## Model Details
- **Model type:** random_forest (see `model_artifacts/model_metadata.json`)
- **Version:** v1.0_20260801
- **Classes:** BENIGN, Botnet, BruteForce, DoS, Infiltration, PortScan, WebAttack
- **Number of input features:** 30 (selected via mutual information from 49 raw CICIDS2017-style flow features)
- **Trained by:** Lokesh Kumar, THE ARZENS Engineering Internship — AI, Automation & Security Engineering Track (Advanced)

## Intended Use
This model is a **baseline network intrusion detection classifier**, intended to flag network flows as benign or as one of six attack categories (DoS, PortScan, BruteForce, WebAttack, Infiltration, Botnet) for downstream SOC alerting and triage. It is intended to run on flow-level features extracted from network traffic (e.g. via a NetFlow/CICFlowMeter-style exporter) and to feed its predictions into an alerting/dashboard system, not to autonomously block traffic without human review.

## Training Data
Trained on a CICIDS2017-structured dataset (see `README.md` for the full data-sourcing note: a synthetic, schema-matched dataset was used in place of the original multi-gigabyte CICIDS2017 files due to sandbox network constraints). 55,000+ flow records, ~78-80% benign, with the remaining ~20-22% distributed across six attack categories of varying rarity.

## Performance (held-out test set)
- Accuracy: 99.4%
- Precision (macro): 92.8%
- Recall (macro): 95.1%
- F1 (macro): 0.937
- ROC-AUC (macro, one-vs-rest): 0.999
- PR-AUC (macro, one-vs-rest): 0.947
- False Positive Rate (binary attack-alerting view): 0.4%
- False Negative Rate (binary attack-alerting view): 1.3%
- Recommended production decision threshold: 0.6

See `evaluation_report.html` for the full per-class breakdown, model comparison, threshold analysis, and SHAP interpretability results.

## Limitations
- **Synthetic training data:** while structured to match CICIDS2017's schema and class proportions, this model was trained on synthetically generated flow data rather than captured real-world traffic, and should be retrained on real CICIDS2017/UNSW-NB15 data (or an organization's own labeled traffic) before any production use.
- **Rare-class performance:** the rarest attack categories (typically Infiltration and Botnet) have the fewest training examples and are the hardest classes to detect reliably; see Section 2 of the evaluation report for exact per-class recall.
- **Concept drift:** network traffic patterns and attack techniques evolve; this model reflects only the traffic patterns present in its training data and requires periodic retraining and drift monitoring (see the quality/drift validator from the earlier Track 09 submission) to remain effective.
- **Feature dependency:** the model requires the same 30 selected flow-level features, computed consistently with the training pipeline's preprocessing (imputation, scaling, feature selection); it cannot be applied directly to raw packet captures or differently-structured flow exports without running them through the same `preprocessing_pipeline.pkl`.

## Ethical Considerations
- **False positives have a human cost:** flagging legitimate user traffic as malicious can disrupt business operations if acted on automatically; this model's predictions should inform, not replace, analyst judgment, especially near the recommended decision threshold.
- **False negatives have a security cost:** a missed attack can lead to real harm (data breach, service disruption); the cost-sensitive evaluation in this report explicitly weights false negatives more heavily than false positives to reflect this asymmetry, but the appropriate weighting is an organizational risk decision, not a purely technical one.
- **No personally identifiable information:** the feature set used (flow durations, packet/byte statistics, protocol flags) does not include usernames, payload content, or other directly identifying information, though source/destination IPs upstream of this feature set can be quasi-identifying and should be handled per the privacy controls (pseudonymization, k-anonymity) documented in the earlier Track 09 feature-engineering submission if this model is deployed alongside that pipeline.
- **Dual-use awareness:** like any intrusion detection model, understanding what patterns it does and does not catch (see Section 6, error analysis) could theoretically help an adversary evade detection; access to this model card and the full evaluation report should be limited to the security team responsible for deploying and maintaining the model.
