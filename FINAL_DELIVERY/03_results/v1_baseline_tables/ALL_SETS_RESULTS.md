# Baseline results on every set (values in %)

Train = fitted data (how well it learned) · CV = unseen folds of train · Test = held-out patterns · Real SMS = collected messages.

| Model         | Set               |    Acc |   Prec |    Rec |     F1 |    AUC |   FPR |   TN |   FP |   FN |   TP |
|:--------------|:------------------|-------:|-------:|-------:|-------:|-------:|------:|-----:|-----:|-----:|-----:|
| Naive Bayes   | Train (n=9,447)   |  96.74 |  93.11 |  96.36 |  94.71 |  99.29 |  3.1  | 6384 |  204 |  104 | 2755 |
| Naive Bayes   | 10-fold CV (mean) |  92.13 |  82.35 |  94.37 |  87.92 |  98.12 |  8.85 |      |      |      |      |
| Naive Bayes   | Test (n=1,589)    |  89.87 |  73.29 |  92.88 |  81.93 |  97.54 | 11.12 | 1063 |  133 |   28 |  365 |
| Naive Bayes   | Real SMS (n=108)  |  75    |  90.12 |  79.35 |  84.39 |  73.51 | 50    |    8 |    8 |   19 |   73 |
| SVM           | Train (n=9,447)   |  99.99 |  99.97 | 100    |  99.98 |  99.99 |  0.02 | 6587 |    1 |    0 | 2859 |
| SVM           | 10-fold CV (mean) |  97.91 |  96.97 |  96.15 |  96.55 |  99.55 |  1.32 |      |      |      |      |
| SVM           | Test (n=1,589)    |  96.04 |  87.5  |  97.96 |  92.44 |  99.7  |  4.6  | 1141 |   55 |    8 |  385 |
| SVM           | Real SMS (n=108)  |  90.74 |  91.84 |  97.83 |  94.74 |  89.67 | 50    |    8 |    8 |    2 |   90 |
| Random Forest | Train (n=9,447)   | 100    | 100    | 100    | 100    | 100    |  0    | 6588 |    0 |    0 | 2859 |
| Random Forest | 10-fold CV (mean) |  97.02 |  97.43 |  92.62 |  94.93 |  99.34 |  1.08 |      |      |      |      |
| Random Forest | Test (n=1,589)    |  96.6  |  93.57 |  92.62 |  93.09 |  99.49 |  2.09 | 1171 |   25 |   29 |  364 |
| Random Forest | Real SMS (n=108)  |  92.59 |  94.68 |  96.74 |  95.7  |  89.57 | 31.25 |   11 |    5 |    3 |   89 |