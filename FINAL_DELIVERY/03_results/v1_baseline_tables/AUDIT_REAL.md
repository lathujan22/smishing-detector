# Real-SMS audit (%, mean +- std over 3 repeats x 5 folds; near-duplicates grouped at cosine >= 0.3)

| Definition | Model | Acc | Precision | Recall | F1 | FPR | AUC | Normal flagged | Promo flagged | Scam caught |
|---|---|---|---|---|---|---|---|---|---|---|
| A spam+scam | RandomForest | 86.5 ± 0.6 | 86.4 ± 0.4 | 98.2 ± 0.5 | 91.9 ± 0.4 | 54.6 ± 1.6 | 88.5 ± 1.7 | 54.6 ± 1.6 | 97.6 ± 0.6 | 100.0 ± 0.0 |
| A spam+scam | SVM | 88.3 ± 0.0 | 88.0 ± 0.0 | 98.4 ± 0.0 | 92.9 ± 0.0 | 47.2 ± 0.0 | 88.5 ± 0.7 | 47.2 ± 0.0 | 98.0 ± 0.0 | 100.0 ± 0.0 |
| B scam-only | RandomForest | 91.0 ± 1.9 | 68.6 ± 3.3 | 90.8 ± 8.7 | 78.1 ± 5.1 | 9.0 ± 0.8 | 96.1 ± 0.8 | 10.2 ± 1.6 | 8.5 ± 1.2 | 90.8 ± 8.7 |
| B scam-only | SVM | 86.1 ± 0.7 | 56.3 ± 1.2 | 97.7 ± 2.0 | 71.4 ± 1.3 | 16.4 ± 0.8 | 96.4 ± 0.3 | 12.0 ± 1.6 | 18.0 ± 1.6 | 97.7 ± 2.0 |

## Held-out test set (n=1,589), models trained on the training set only

| Definition | Model | Acc | Precision | Recall | F1 | FPR | AUC | Normal flagged | Promo flagged | Scam caught |
|---|---|---|---|---|---|---|---|---|---|---|
| A spam+scam | SVM | 97.2 | 91.8 | 97.5 | 94.6 | 2.8 | 99.7 | 2.8 | 92.2 | 100.0 |
| A spam+scam | RandomForest | 97.7 | 95.9 | 94.9 | 95.4 | 1.3 | 99.6 | 1.3 | 86.7 | 98.9 |
| B scam-only | SVM | 97.2 | 87.5 | 97.4 | 92.1 | 2.8 | 99.2 | 1.4 | 15.6 | 97.4 |
| B scam-only | RandomForest | 97.6 | 94.5 | 90.9 | 92.7 | 1.1 | 99.5 | 0.0 | 10.9 | 90.9 |