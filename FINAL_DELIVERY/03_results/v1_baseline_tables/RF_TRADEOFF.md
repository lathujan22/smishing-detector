# Random Forest - capacity and threshold trade-off (%)

## A. Shallow forests (threshold 0.5)

| Setting | Train acc | Test acc | Test F1 | Test FPR | Real acc | Real F1 | Real FPR |
|---|---|---|---|---|---|---|---|
| depth=full, leaf=1 | 100.0 | 97.17 | 94.31 | 2.09 | 93.52 | 96.3 | 37.5 |
| depth=20, leaf=5 | 98.48 | 95.47 | 90.89 | 3.18 | 93.52 | 96.3 | 37.5 |
| depth=8, leaf=10 | 96.59 | 94.15 | 88.27 | 4.18 | 92.59 | 95.74 | 37.5 |
| depth=5, leaf=20 | 95.24 | 92.01 | 84.72 | 7.19 | 92.59 | 95.74 | 37.5 |
| depth=3, leaf=50 | 91.98 | 88.8 | 79.82 | 11.45 | 92.59 | 95.74 | 37.5 |
| depth=2, leaf=100 | 92.66 | 88.92 | 79.91 | 11.12 | 92.59 | 95.74 | 37.5 |

## B + C. Cross-validated training accuracy and threshold tuning (final RF)

| Threshold | t | CV acc | CV F1 | CV FPR | Test acc | Test F1 | Test FPR | Real acc | Real F1 | Real FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| default 0.50 | 0.5 | 97.28 | 95.42 | 1.12 | 97.17 | 94.31 | 2.09 | 93.52 | 96.3 | 37.5 |
| CV-FPR <= 1% | 0.52 | 97.34 | 95.51 | 0.93 | 97.67 | 95.26 | 1.34 | 93.52 | 96.3 | 37.5 |
| CV-FPR <= 0.5% | 0.62 | 97.03 | 94.89 | 0.47 | 97.17 | 94.18 | 1.34 | 94.44 | 96.81 | 31.25 |