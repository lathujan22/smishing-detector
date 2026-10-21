# Multilingual Smishing Detector V2 — audit and implementation plan

Primary task: scam vs non-scam (promotional = non-scam). Secondary: normal / promotional / scam.
Original experiments are preserved unchanged (`results/*.csv`, `results/scam_only`, `results/three_class`,
`models/final*`, `models/scam_only`, `models/three_class`). V2 writes only to `data/v2`, `results/v2`, `models/v2`.

## 1. Audit findings (2026-10-18)

| # | Check | Finding | Consequence / fix |
|---|---|---|---|
| 1 | Train/test separation (v1) | **331 of 1,589 v1 test messages share a near-duplicate family with training** (cosine ≥ 0.8 on masked text, or same template / translation source); 259 are near-identical | v1 held-out test scores are optimistic. V2 re-splits by family (0 shared families). |
| 2 | CV groups (v1) | Groups = template / MT source only; 1,398 training rows have a near-duplicate in another group | v1 CV scores optimistic. V2 CV groups = near-duplicate families. |
| 3 | Real benchmark vs training | No real message has a family in the training pool (max cosine 0.61) | No direct contamination. |
| 4 | Real benchmark used in decisions | The 163 real SMS were inspected for error analysis, label-definition choice and earlier model comparisons | **Disclosed limitation**: it is an audited benchmark, not a pristine test. A new independent real test set must be collected for final claims. |
| 5 | Augmentation / TF-IDF / SMOTE | Fitted inside training folds only (`src/scam_pipeline.py`, `src/v2/models.py`) | OK. |
| 6 | Thresholds | Chosen on CV out-of-fold scores (training data) | OK. |
| 7 | Class definitions | Promotional = negative in scam-only; labels from datasets / researcher; never inferred | OK. 29 pool messages with identical text but conflicting labels → quarantined. |
| 8 | Synthetic data share | 44.7% of the pool is template-generated or machine-translated | Reported separately; ablation "human-written only". |
| 9 | Distribution shift | Links in 3% / 21% / 40% of training normal / promo / scam vs 31% / 65% / 97% of real SL SMS | Main cause of real-SMS false positives; needs real SL normal + promotional training data. |
| 10 | Environment | No xgboost, torch, transformers or Hugging Face access in the cloud sandbox | Gradient Boosting reported as its own algorithm (not XGBoost); XLM-R runs on Colab / a GPU PC. |

## 2. Evaluation levels

- **A – CV**: grouped 5-fold StratifiedGroupKFold on `data/v2/train.csv`, groups = family_id.
- **B – held-out test**: `data/v2/test.csv`, no shared families with training (max test→train cosine 0.80).
- **C – real-world**: (C1) the 163-message audited benchmark (limitation above); (C2) a NEW locked real test set, to be
  collected and ingested with `python -m src.v2.registry ingest FILE --role locked_test`, used once at the end.

## 3. Stages

| Stage | Content | Where | Status |
|---|---|---|---|
| 1 | Registry, provenance, PII masking, risk features, families, quarantine, ingestion (CSV + SMS Backup XML) | `src/v2/registry.py`, `src/v2/split.py` | done |
| 2 | Controlled SVM experiments (n-grams, vocabulary, C, imbalance, augmentation, context features, calibration, data ablation) + greedy CV selection; Exp A, E, F | `src/v2/svm_experiments.py` | running |
| 3 | Three-class: LinearSVC, Logistic Regression, Random Forest, Gradient Boosting; Exp C | `src/v2/three_class.py` | next |
| 4 | Exp B (more legitimate promotional / transactional data) | needs new real SL data in the dev pool | blocked on data |
| 5 | XLM-R binary + three-class fine-tuning, token-length analysis, early stopping, LR / batch sweep; Exp D, G | `src/v2/xlmr_train.py` (Colab GPU) | script to prepare |
| 6 | FastAPI service (`/health`, `/predict`, `/predict/batch`, `/model/info`) | `api/` | after evaluation |
| 7 | Research dashboard | `dashboard/` | after API |
| 8 | Final report: dataset statistics, tables with CIs, curves, per-language, errors, ablation, threats to validity | `results/v2/REPORT.md` | continuous |

## 4. Rules

No retraining on locked test data; no thresholds or hyper-parameters chosen on test or real data; no fabricated
"real" messages; promotional never auto-labelled scam; per-group metrics flagged unreliable when n < 20 or < 5 scams;
every run writes a manifest (config, seeds, data md5, library versions); real SMS text stays out of Git.
