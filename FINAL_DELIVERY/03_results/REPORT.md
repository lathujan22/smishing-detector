# Smishing detector V2 - experiment report

All numbers measured by the scripts in `src/v2/`; every run has a manifest (config, seed 42, data md5, library versions) in its results folder. Original experiments are preserved unchanged.

## Data audit and split

- Registry rows: 11199 (pool 11036, real benchmark 163)
- Families (exact + near-duplicate cosine >= 0.8 + template/translation groups): pool 5712, largest family 48 messages, families with >1 message: 1633
- **V1 split contamination:** 331 of 1589 v1 test messages share a family with v1 training messages -> v1 held-out test results are optimistic.
- Real benchmark vs pool: 0 pool messages share a family with a real benchmark message.
- Quarantined (identical text, conflicting labels): 29
- Synthetic or machine-translated rows in pool: 4932 (44.7%); human-labelled real-text rows: 6104

- Pool (non-quarantined) 11007 -> train 9433 / held-out test 1574 (14.3%)
- Families: train 4899, test 813, shared **0**
- Test -> train nearest char-n-gram cosine: max 0.802, 99th pct 0.789, median 0.334 (v1 had 354 test messages >= 0.8)
- Synthetic / MT share: train 44.7%, test 44.8%

**Disclosure:** the 163-message real benchmark (C1) has already informed research decisions (error analysis, label definition, earlier model comparisons). It is an audited benchmark, not a pristine test. Final claims require a new locked real-world test set (C2), not yet collected.

## Experiment 3 - SVM, one factor at a time (grouped 5-fold CV)

| Configuration | Factor | AUPRC | F1 | Precision | Recall | FPR | ΔAUPRC | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| E0 baseline (original recipe) | baseline | 95.31 | 90.52 | 90.30 | 90.75 | 2.61 | 0.00 | 0.00 |
| word only | features | 92.80 | 89.04 | 86.93 | 91.25 | 3.67 | -2.51 | -1.48 |
| char only | features | 94.64 | 89.98 | 88.79 | 91.20 | 3.08 | -0.67 | -0.54 |
| word 1-3 grams | n-grams | 95.23 | 90.47 | 89.76 | 91.20 | 2.78 | -0.08 | -0.05 |
| char 1-4 grams | n-grams | 95.40 | 90.72 | 89.53 | 91.96 | 2.87 | 0.09 | 0.20 |
| char 3-6 grams | n-grams | 95.02 | 90.38 | 89.18 | 91.60 | 2.97 | -0.29 | -0.14 |
| 30k word + 30k char features | vocabulary | 95.28 | 90.87 | 90.15 | 91.60 | 2.67 | -0.03 | 0.35 |
| C = 0.03 | regularisation | 94.51 | 90.13 | 88.47 | 91.86 | 3.20 | -0.80 | -0.39 |
| C = 0.3 | regularisation | 95.25 | 90.53 | 89.33 | 91.75 | 2.93 | -0.06 | 0.01 |
| C = 1.0 | regularisation | 94.76 | 90.13 | 88.47 | 91.86 | 3.20 | -0.55 | -0.39 |
| class weighting instead of SMOTE | imbalance | 95.03 | 90.37 | 89.66 | 91.10 | 2.81 | -0.28 | -0.15 |
| no resampling | imbalance | 95.80 | 90.42 | 90.20 | 90.65 | 2.63 | 0.49 | -0.10 |
| no noise augmentation | augmentation | 94.99 | 90.32 | 89.61 | 91.05 | 2.82 | -0.32 | -0.20 |
| + contextual risk features | context | 96.10 | 91.27 | 90.02 | 92.56 | 2.74 | 0.79 | 0.75 |
| + contextual risk features (x3 weight) | context | 96.12 | 91.22 | 89.08 | 93.46 | 3.06 | 0.81 | 0.70 |
| sigmoid-calibrated SVM | calibration | 95.15 | 90.42 | 89.71 | 91.15 | 2.79 | -0.16 | -0.10 |
| train without machine-translated rows | training data | 94.07 | 88.34 | 88.10 | 88.59 | 3.20 | -1.24 | -2.18 |
| train on human-written data only | training data | 84.63 | 78.17 | 66.69 | 94.42 | 12.60 | -10.68 | -12.35 |

Greedy combination (CV AUPRC, kept only if it improves CV):

| added | cv_auprc | cv_f1 | kept |
|---|---|---|---|
| + contextual risk features (x3 weight) | 96.12 | 91.22 | True |
| no resampling | 96.34 | 91.32 | True |

Selected 'improved' configuration: context features ×3.0, resampling = none, C = 0.1, word (1, 2) + char (2, 5) n-grams.

## Experiments A, E, F - three evaluation levels

| Model | Set | Precision | Recall | F1 | FPR | ROC-AUC | AUPRC | F1 95% CI | FPR 95% CI | AUPRC 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|
| A: original SVM (baseline) | cv (validation) | 90.30 | 90.75 | 90.52 | 2.61 | 98.75 | 95.31 | – | – | – |
| A: original SVM (baseline) | train (in-sample) | 96.62 | 99.25 | 97.92 | 0.93 | 99.94 | 99.72 | – | – | – |
| A: original SVM (baseline) | held-out test | 92.60 | 93.15 | 92.88 | 2.02 | 99.34 | 97.11 | [90.7, 94.8] | [1.3, 2.8] | [95.1, 98.6] |
| A: original SVM (baseline) | real benchmark | 38.89 | 96.55 | 55.45 | 32.84 | 90.99 | 65.52 | [43.0, 66.1] | [25.4, 40.4] | [49.9, 82.9] |
| A: improved SVM (CV-selected) | cv (validation) | 89.64 | 93.06 | 91.32 | 2.87 | 98.85 | 96.34 | – | – | – |
| A: improved SVM (CV-selected) | train (in-sample) | 95.85 | 98.79 | 97.30 | 1.14 | 99.93 | 99.71 | – | – | – |
| A: improved SVM (CV-selected) | held-out test | 90.18 | 90.18 | 90.18 | 2.67 | 99.01 | 97.35 | [87.9, 92.7] | [1.8, 3.6] | [96.2, 98.3] |
| A: improved SVM (CV-selected) | real benchmark | 30.77 | 96.55 | 46.67 | 47.01 | 90.68 | 73.51 | [35.0, 56.9] | [38.7, 55.1] | [58.6, 86.8] |
| E: word-only TF-IDF | cv (validation) | 86.93 | 91.25 | 89.04 | 3.67 | 98.33 | 92.80 | – | – | – |
| E: word-only TF-IDF | train (in-sample) | 94.19 | 99.40 | 96.72 | 1.64 | 99.84 | 98.56 | – | – | – |
| E: word-only TF-IDF | held-out test | 87.63 | 97.02 | 92.09 | 3.72 | 99.16 | 96.49 | [90.0, 94.1] | [2.7, 4.8] | [94.4, 98.1] |
| E: word-only TF-IDF | real benchmark | 34.67 | 89.66 | 50.00 | 36.57 | 90.68 | 73.69 | [37.0, 60.6] | [28.2, 45.1] | [58.7, 88.0] |
| E: word + char TF-IDF | cv (validation) | 90.30 | 90.75 | 90.52 | 2.61 | 98.75 | 95.31 | – | – | – |
| E: word + char TF-IDF | train (in-sample) | 96.62 | 99.25 | 97.92 | 0.93 | 99.94 | 99.72 | – | – | – |
| E: word + char TF-IDF | held-out test | 92.60 | 93.15 | 92.88 | 2.02 | 99.34 | 97.11 | [90.7, 94.8] | [1.3, 2.8] | [95.1, 98.6] |
| E: word + char TF-IDF | real benchmark | 38.89 | 96.55 | 55.45 | 32.84 | 90.99 | 65.52 | [43.0, 66.1] | [25.4, 40.4] | [49.9, 82.9] |
| F: without contextual features | cv (validation) | 90.20 | 90.65 | 90.42 | 2.63 | 98.76 | 95.80 | – | – | – |
| F: without contextual features | train (in-sample) | 97.36 | 98.19 | 97.77 | 0.71 | 99.93 | 99.74 | – | – | – |
| F: without contextual features | held-out test | 92.94 | 94.05 | 93.49 | 1.94 | 99.29 | 97.33 | [91.4, 95.4] | [1.2, 2.7] | [95.6, 98.6] |
| F: without contextual features | real benchmark | 39.19 | 100.00 | 56.31 | 33.58 | 91.69 | 70.89 | [44.6, 66.7] | [25.8, 41.3] | [55.7, 84.7] |
| F: with contextual features | cv (validation) | 89.64 | 93.06 | 91.32 | 2.87 | 98.85 | 96.34 | – | – | – |
| F: with contextual features | train (in-sample) | 95.85 | 98.79 | 97.30 | 1.14 | 99.93 | 99.71 | – | – | – |
| F: with contextual features | held-out test | 90.18 | 90.18 | 90.18 | 2.67 | 99.01 | 97.35 | [87.9, 92.7] | [1.8, 3.6] | [96.2, 98.3] |
| F: with contextual features | real benchmark | 30.77 | 96.55 | 46.67 | 47.01 | 90.68 | 73.51 | [35.0, 56.9] | [38.7, 55.1] | [58.6, 86.8] |

## Experiment C - binary vs three-class SVM (scam-vs-rest view)

| Set | Model | Scam recall | 95% CI | Non-scam→scam | 95% CI | Normal→scam | Promo→scam | Scam F1 |
|---|---|---|---|---|---|---|---|---|
| held-out test | binary SVM (original recipe) | 93.15 | [90.3, 95.7] | 2.02 | [1.3, 2.8] | 0.81 | 12.60 | 92.88 |
| held-out test | binary SVM (improved) | 90.18 | [87.0, 93.4] | 2.67 | [1.8, 3.6] | 0.99 | 17.32 | 90.18 |
| held-out test | three-class LinearSVC | 91.37 | [88.3, 94.3] | 1.78 | [1.1, 2.5] | 0.99 | 8.66 | 92.33 |
| real benchmark | binary SVM (original recipe) | 96.55 | [88.9, 100.0] | 32.84 | [25.4, 40.4] | 22.22 | 36.73 | 55.45 |
| real benchmark | binary SVM (improved) | 96.55 | [88.9, 100.0] | 47.01 | [38.7, 55.1] | 25.00 | 55.10 | 46.67 |
| real benchmark | three-class LinearSVC | 89.66 | [77.3, 100.0] | 17.91 | [11.8, 24.6] | 22.22 | 16.33 | 65.82 |

## Three-class models (Normal / Promotional / Scam)

| Model | Set | Acc | Macro F1 | Macro AUC | Macro AUPRC | Normal R | Promo R | Scam R | Scam P | Non-scam→scam |
|---|---|---|---|---|---|---|---|---|---|---|
| LinearSVC | cv (validation) | 94.68 | 87.79 | 97.83 | 91.11 | 98.80 | 74.19 | 88.79 | 91.93 | 2.08 |
| LinearSVC | train (in-sample) | 99.43 | 98.54 | 99.97 | 99.72 | 99.93 | 98.83 | 97.99 | 99.34 | 0.17 |
| LinearSVC | held-out test | 95.30 | 89.67 | 98.13 | 93.52 | 98.74 | 75.59 | 91.37 | 93.31 | 1.78 |
| LinearSVC | real benchmark | 77.30 | 73.15 | 90.18 | 81.33 | 55.56 | 81.63 | 89.66 | 52.00 | 17.91 |
| LogisticRegression | cv (validation) | 94.24 | 86.65 | 97.81 | 90.75 | 98.50 | 71.21 | 88.89 | 90.90 | 2.38 |
| LogisticRegression | train (in-sample) | 99.72 | 99.26 | 100.00 | 99.97 | 99.99 | 99.09 | 99.10 | 99.60 | 0.11 |
| LogisticRegression | held-out test | 94.66 | 89.00 | 98.36 | 93.59 | 98.11 | 77.95 | 89.58 | 91.21 | 2.34 |
| LogisticRegression | real benchmark | 75.46 | 70.62 | 89.42 | 80.13 | 50.00 | 79.59 | 93.10 | 47.37 | 22.39 |
| RandomForest | cv (validation) | 93.52 | 84.56 | 97.95 | 90.18 | 98.79 | 62.65 | 87.83 | 92.38 | 1.93 |
| RandomForest | train (in-sample) | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 |
| RandomForest | held-out test | 94.16 | 85.23 | 97.87 | 90.28 | 100.00 | 60.63 | 87.50 | 94.53 | 1.37 |
| RandomForest | real benchmark | 76.07 | 72.02 | 87.18 | 77.71 | 55.56 | 80.61 | 86.21 | 54.35 | 15.67 |
| GradientBoosting (sklearn, not XGBoost) | cv (validation) | 94.13 | 86.06 | 97.87 | 90.93 | 98.29 | 64.59 | 91.60 | 89.62 | 2.83 |
| GradientBoosting (sklearn, not XGBoost) | train (in-sample) | 99.55 | 98.87 | 99.98 | 99.79 | 99.99 | 97.80 | 98.79 | 99.39 | 0.16 |
| GradientBoosting (sklearn, not XGBoost) | held-out test | 94.54 | 87.27 | 97.84 | 91.86 | 98.11 | 68.50 | 92.56 | 89.63 | 2.91 |
| GradientBoosting (sklearn, not XGBoost) | real benchmark | 74.23 | 68.59 | 87.85 | 76.29 | 52.78 | 82.65 | 72.41 | 46.67 | 17.91 |

Gradient Boosting = scikit-learn GradientBoostingClassifier, a separate algorithm (XGBoost is not installed in this environment).

## Experiment G - per language (real benchmark)

| Model | Lang | n | Scams | Precision | Recall | F1 | FPR | Reliable (n≥20, ≥5 scams) |
|---|---|---|---|---|---|---|---|---|
| A: original SVM (baseline) | EN | 89 | 12 | 46.15 | 100.00 | 63.16 | 18.18 | True |
| A: original SVM (baseline) | SI | 67 | 17 | 37.21 | 94.12 | 53.33 | 54.00 | True |
| A: original SVM (baseline) | TA | 4 | 0 | 0.00 | 0.00 | 0.00 | 50.00 | False |
| A: original SVM (baseline) | mixed | 3 | 0 | 0.00 | 0.00 | 0.00 | 33.33 | False |
| A: improved SVM (CV-selected) | EN | 89 | 12 | 27.27 | 100.00 | 42.86 | 41.56 | True |
| A: improved SVM (CV-selected) | SI | 67 | 17 | 36.36 | 94.12 | 52.46 | 56.00 | True |
| A: improved SVM (CV-selected) | TA | 4 | 0 | 0.00 | 0.00 | 0.00 | 50.00 | False |
| A: improved SVM (CV-selected) | mixed | 3 | 0 | 0.00 | 0.00 | 0.00 | 33.33 | False |
| E: word-only TF-IDF | EN | 89 | 12 | 34.62 | 75.00 | 47.37 | 22.08 | True |
| E: word-only TF-IDF | SI | 67 | 17 | 36.96 | 100.00 | 53.97 | 58.00 | True |
| E: word-only TF-IDF | TA | 4 | 0 | 0.00 | 0.00 | 0.00 | 50.00 | False |
| E: word-only TF-IDF | mixed | 3 | 0 | 0.00 | 0.00 | 0.00 | 33.33 | False |
| E: word + char TF-IDF | EN | 89 | 12 | 46.15 | 100.00 | 63.16 | 18.18 | True |
| E: word + char TF-IDF | SI | 67 | 17 | 37.21 | 94.12 | 53.33 | 54.00 | True |
| E: word + char TF-IDF | TA | 4 | 0 | 0.00 | 0.00 | 0.00 | 50.00 | False |
| E: word + char TF-IDF | mixed | 3 | 0 | 0.00 | 0.00 | 0.00 | 33.33 | False |
| F: without contextual features | EN | 89 | 12 | 48.00 | 100.00 | 64.86 | 16.88 | True |
| F: without contextual features | SI | 67 | 17 | 36.96 | 100.00 | 53.97 | 58.00 | True |
| F: without contextual features | TA | 4 | 0 | 0.00 | 0.00 | 0.00 | 50.00 | False |
| F: without contextual features | mixed | 3 | 0 | 0.00 | 0.00 | 0.00 | 33.33 | False |
| F: with contextual features | EN | 89 | 12 | 27.27 | 100.00 | 42.86 | 41.56 | True |
| F: with contextual features | SI | 67 | 17 | 36.36 | 94.12 | 52.46 | 56.00 | True |
| F: with contextual features | TA | 4 | 0 | 0.00 | 0.00 | 0.00 | 50.00 | False |
| F: with contextual features | mixed | 3 | 0 | 0.00 | 0.00 | 0.00 | 33.33 | False |

## Experiment B (more legitimate promotional / transactional data)

**Not run - blocked on data.** No new real Sri Lankan promotional / transactional messages have been collected yet. Synthetic examples will not be generated and described as real. Ingest exports with `python -m src.v2.registry ingest FILE --role dev`, then re-run the experiments with the dev pool added to training.

## Experiment D - deep models (CNN, BiLSTM, CNN-BiLSTM-Attention, XLM-R) vs SVM

Deep models trained on the user's laptop GPU (RTX 4050, 6 GB) with the same family-grouped protocol: a family-grouped validation split of the training data is used for early stopping, checkpoint and threshold; held-out test and the real benchmark are scored once. XLM-R: xlm-roberta-base, lr 2e-5, effective batch 16 (8 x 2 accumulation), gradient checkpointing, max_len 128, class-weighted loss. CNN / BiLSTM / Hybrid: XLM-R sub-word tokens, embeddings from scratch.

### Binary scam vs non-scam

| Model | Run | Set | Precision | Recall | F1 | FPR | AUPRC | F1 95% CI | FPR 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| XLM-R (binary_lr2e-05_bs16_len128_seed42) | binary_lr2e-05_bs16_len128_seed42 | held-out test | 88.09 | 94.64 | 91.25 | 3.47 | 93.87 | [89.2, 93.5] | [2.5, 4.5] |
| XLM-R (binary_lr2e-05_bs16_len128_seed42) | binary_lr2e-05_bs16_len128_seed42 | real benchmark | 35.62 | 89.66 | 50.98 | 35.07 | 74.44 | [37.9, 62.4] | [26.7, 43.7] |
| BILSTM | bilstm_binary_seed42 | held-out test | 93.77 | 94.05 | 93.91 | 1.70 | 97.44 | [91.8, 95.7] | [1.0, 2.5] |
| BILSTM | bilstm_binary_seed42 | real benchmark | 25.00 | 65.52 | 36.19 | 42.54 | 38.83 | [24.5, 46.9] | [34.6, 51.2] |
| CNN | cnn_binary_seed42 | held-out test | 90.32 | 91.67 | 90.99 | 2.67 | 97.23 | [88.6, 93.2] | [1.8, 3.5] |
| CNN | cnn_binary_seed42 | real benchmark | 24.73 | 79.31 | 37.70 | 52.24 | 56.52 | [26.5, 48.6] | [43.4, 61.1] |
| HYBRID | hybrid_binary_seed42 | held-out test | 85.52 | 94.94 | 89.99 | 4.36 | 97.09 | [87.4, 92.2] | [3.2, 5.5] |
| HYBRID | hybrid_binary_seed42 | real benchmark | 21.67 | 89.66 | 34.90 | 70.15 | 46.40 | [25.0, 44.0] | [62.1, 77.8] |
| A: original SVM (baseline) | svm (src/v2/svm_experiments) | held-out test | 92.60 | 93.15 | 92.88 | 2.02 | 97.11 | [90.7, 94.8] | [1.3, 2.8] |
| A: original SVM (baseline) | svm (src/v2/svm_experiments) | real benchmark | 38.89 | 96.55 | 55.45 | 32.84 | 65.52 | [43.0, 66.1] | [25.4, 40.4] |

### Three-class

| Model | Run | Set | Macro F1 | Scam recall | Non-scam->scam | Normal->scam | Promo->scam |
|---|---|---|---|---|---|---|---|
| XLM-R (three_lr2e-05_bs16_len128_seed42) | three_lr2e-05_bs16_len128_seed42 | held-out test | 90.83 | 91.96 | 0.89 | 0.00 | 8.66 |
| XLM-R (three_lr2e-05_bs16_len128_seed42) | three_lr2e-05_bs16_len128_seed42 | real benchmark | 69.96 | 89.66 | 20.15 | 27.78 | 17.35 |
| HYBRID | hybrid_three_seed42 | held-out test | 84.10 | 90.48 | 0.89 | 0.00 | 8.66 |
| HYBRID | hybrid_three_seed42 | real benchmark | 57.18 | 58.62 | 32.09 | 16.67 | 37.76 |
| LinearSVC | three-class LinearSVC | held-out test | 89.67 | 91.37 | 1.78 | 0.99 | 8.66 |
| LinearSVC | three-class LinearSVC | real benchmark | 73.15 | 89.66 | 17.91 | 22.22 | 16.33 |

## Threats to validity

- Real benchmark is small (163; 29 scams); Tamil n = 4 and mixed n = 3 -> no per-language conclusion for them.
- The real benchmark informed earlier decisions (see disclosure) -> mild optimistic bias possible.
- 44.7% of training data is template-generated or machine-translated; the 'human-written only' ablation shows the Sinhala / Tamil coverage depends on it.
- Distribution shift between training data and real Sri Lankan SMS (links / money far more common in real legitimate SMS) dominates the real-world error; CV and held-out test do not measure it.
- Hyper-parameter search is limited (one factor at a time + greedy); improvements smaller than the CI width are not claimed.
- Overfitting to the training distribution is not ruled out: training (in-sample) ≫ real-benchmark performance.

## Figures

`results/v2/figures/`: fig_v2_svm_ablation, fig_v2_levels, fig_v2_roc_pr, fig_v2_exp_c, fig_v2_three_class_cm_real, fig_v2_by_language, fig_v2_exp_d, fig_v2_dl_curves.