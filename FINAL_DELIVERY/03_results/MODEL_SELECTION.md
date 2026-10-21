# Final model selection

**Selection rule** (fixed before looking at the ranking):
  1. Task formulation = three-class (Normal / Promotional / Scam), because Experiment C showed it halves false alarms
     on real SMS with non-overlapping confidence intervals (binary models are kept for comparison).
  2. Within ML and within DL, rank by HELD-OUT TEST macro-F1 (evaluation level B - never the real benchmark).
  3. The real benchmark (C1) is reported for the selected models only as an audited check; it has informed earlier
     decisions, so it is not used to pick the winner.

## Three-class ranking (held-out test, level B)

| Rank | Family | Model | Test macro-F1 | Test scam recall | Test non-scam->scam |
|---|---|---|---|---|---|
| 1 | DL | XLM-R | 90.83 | 91.96 | 0.89 |
| 2 | ML | LinearSVC | 89.67 | 91.37 | 1.78 |
| 3 | ML | LogisticRegression | 89.00 | 89.58 | 2.34 |
| 4 | ML | GradientBoosting | 87.27 | 92.56 | 2.91 |
| 5 | ML | RandomForest | 85.23 | 87.50 | 1.37 |
| 6 | DL | CNN-BiLSTM-Attention | 84.10 | 90.48 | 0.89 |

## Selected models

- **Best ML: LinearSVC (three-class)** - test macro-F1 89.67; real benchmark macro-F1 73.15, scam recall 89.66%, non-scam flagged as scam 17.91%. Artifact: `models/v2/three_linearsvc.joblib`
- **Best DL: XLM-R (three-class)** - test macro-F1 90.83; real benchmark macro-F1 69.96, scam recall 89.66%, non-scam flagged as scam 20.15%. Artifact: `models/v2/xlmr/three_lr2e-05_bs16_len128_seed42`

On the real benchmark the two selected models are statistically indistinguishable (scam recall identical, non-scam->scam 17.9% [11.8, 24.6] vs 20.2% [13.3, 27.0], bootstrap 95% CI; paired McNemar test on scam-vs-rest decisions p = 0.71). The web application therefore uses the **LinearSVC as the default** (fast, CPU-only, explainable n-gram evidence) and offers **XLM-R as a second opinion** when its checkpoint and PyTorch are available.

## All models (held-out test and real benchmark)

| Family | Task | Model | Set | Macro-F1 | Scam recall | Non-scam->scam |
|---|---|---|---|---|---|---|
| ML | three-class | LinearSVC | held-out test | 89.67 | 91.37 | 1.78 |
| ML | three-class | LinearSVC | real benchmark | 73.15 | 89.66 | 17.91 |
| ML | three-class | LogisticRegression | held-out test | 89.00 | 89.58 | 2.34 |
| ML | three-class | LogisticRegression | real benchmark | 70.62 | 93.10 | 22.39 |
| ML | three-class | RandomForest | held-out test | 85.23 | 87.50 | 1.37 |
| ML | three-class | RandomForest | real benchmark | 72.02 | 86.21 | 15.67 |
| ML | three-class | GradientBoosting | held-out test | 87.27 | 92.56 | 2.91 |
| ML | three-class | GradientBoosting | real benchmark | 68.59 | 72.41 | 17.91 |
| DL | three-class | XLM-R | held-out test | 90.83 | 91.96 | 0.89 |
| DL | three-class | XLM-R | real benchmark | 69.96 | 89.66 | 20.15 |
| DL | three-class | CNN-BiLSTM-Attention | held-out test | 84.10 | 90.48 | 0.89 |
| DL | three-class | CNN-BiLSTM-Attention | real benchmark | 57.18 | 58.62 | 32.09 |
| ML | binary | SVM (binary) | held-out test | 95.47 | 93.15 | 2.02 |
| ML | binary | SVM (binary) | real benchmark | 67.72 | 96.55 | 32.84 |
| DL | binary | XLM-R (binary) | held-out test | 94.38 | 94.64 | 3.47 |
| DL | binary | XLM-R (binary) | real benchmark | 64.33 | 89.66 | 35.07 |
| DL | binary | BiLSTM (binary) | held-out test | 96.13 | 94.05 | 1.70 |
| DL | binary | BiLSTM (binary) | real benchmark | 52.94 | 65.52 | 42.54 |
| DL | binary | CNN (binary) | held-out test | 94.26 | 91.67 | 2.67 |
| DL | binary | CNN (binary) | real benchmark | 50.23 | 79.31 | 52.24 |
| DL | binary | CNN-BiLSTM-Attention (binary) | held-out test | 93.54 | 94.94 | 4.36 |
| DL | binary | CNN-BiLSTM-Attention (binary) | real benchmark | 40.05 | 89.66 | 70.15 |

Limitations: real benchmark n=163 (29 scams) and it informed earlier decisions; a new locked real test set is required for final claims. Gradient Boosting is scikit-learn's GradientBoostingClassifier, not XGBoost.