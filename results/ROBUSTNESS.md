# Robustness evaluation (F1 %)

| Scenario | n | Naive Bayes | SVM | Random Forest |
|---|---|---|---|---|
| 1 Standard test | 1589 | 87.32 | 94.57 | 95.40 |
| 2 Public datasets only | 1221 | 94.58 | 95.98 | 97.10 |
| 3 Noisy test (evasion) | 1589 | 85.65 | 94.06 | 92.58 |
| 4 Real SMS | 108 | 82.84 | 95.29 | 96.30 |
| 5 Unseen Sinhala | 180 | 73.64 | 89.14 | 58.57 |
| 6 Unseen Tamil | 188 | 68.07 | 81.52 | 58.99 |
| 7 Unseen templates | 368 | 55.14 | 66.35 | 63.74 |

Scenarios 5-7 retrain the model without that language / source (same noise + SMOTE settings).

Full metrics (acc, precision, recall, F1, FPR): `results/robustness.csv`.