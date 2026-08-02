# AI Assistance Note — Assignment 4

As required by the Task Assignment Manual ("Where you use any AI
assistance, disclose it briefly..."), this note discloses the use of an AI
assistant (Claude, by Anthropic) in completing this assignment.

## How AI was used

- **Task 1 (ML Design Brief):** Discussing structure and content for the
  design brief (problem definition, dataset analysis, model selection
  rationale, evaluation strategy), and drafting/formatting the final
  document, which I reviewed for technical accuracy.
- **Task 2 (train_model.py):** Drafting the initial pipeline structure
  (data ingestion, preprocessing, SMOTE-based imbalance handling, model
  training with RandomizedSearchCV, model persistence), then iterating
  based on test runs to fix issues — for example, capping SMOTE's
  oversampling target and adding a subsample-then-refit-on-full-data
  pattern for hyperparameter search, both needed to make training feasible
  on the available single-core compute environment.
- **Task 3 (evaluate_model.py):** Drafting the evaluation pipeline
  (performance metrics, model comparison, threshold analysis, SHAP
  interpretability, error analysis, production readiness check) and the
  HTML report / model card generation code.
- **Practical Task 1 (API):** Drafting the FastAPI deployment app and its
  documentation.
- **Dataset sourcing:** Per the assignment's own "Quick Alternative"
  guidance (downloads slow/unavailable → generate synthetic data with
  Python), a synthetic, CICIDS2017-schema-matched dataset was generated
  with AI assistance, since the sandboxed development environment used for
  this submission did not have network access to the official CICIDS2017 /
  UNSW-NB15 download sources. This is disclosed in detail in `README.md`.
- **Debugging:** Diagnosing and fixing runtime issues encountered while
  testing (e.g. a division-by-zero in synthetic data generation, a pandas
  dtype-compatibility warning, and compute-time constraints requiring
  lighter hyperparameter search settings).

## What was not AI-generated

- The overall pipeline design decisions, scenario selection (Network
  Intrusion Detection), metric choices, and interpretation of results
  (e.g. which features matter and why, what the error analysis implies)
  reflect my own understanding of the material covered in this track.
- Every script was actually executed against the dataset and its output
  inspected before being included in this submission — nothing here is
  unverified or fabricated output.

## Academic integrity statement

This submission reflects my own understanding of the ML threat detection
concepts covered in this assignment. No ML pipeline code was copied from
Kaggle or other sources without attribution, and no pre-trained models from
other sources were used — all models were trained from scratch using this
submission's own pipeline. AI tools were used as a learning and
productivity aid in the manner permitted by the assignment instructions,
not as a substitute for understanding or as a source of pre-built,
uncredited solutions.
