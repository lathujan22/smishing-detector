# Retraining with real Sri Lankan SMS (%)

Real SMS n=163 (ham 36, smishing 29, spam 98); 'after' = pooled 5-fold out-of-fold predictions. Test n=1589.

| Model | Version | Real acc | Real F1 | Real FPR | Scams caught | Test acc | Test F1 | Test FPR |
|---|---|---|---|---|---|---|---|---|
| NaiveBayes | before (models/final) | 69.94 | 79.5 | 47.22 | 82.76 | 93.2 | 87.32 | 7.27 |
| NaiveBayes | after (+ real SMS, 5-fold CV) | 85.89 | 91.32 | 47.22 | 89.66 | 93.58 | 88.06 | 7.11 |
| SVM | before (models/final) | 85.89 | 91.58 | 58.33 | 100.0 | 97.23 | 94.57 | 2.84 |
| SVM | after (+ real SMS, 5-fold CV) | 88.34 | 92.94 | 47.22 | 100.0 | 96.54 | 93.32 | 3.85 |
| RandomForest | before (models/final) | 85.28 | 91.24 | 61.11 | 100.0 | 97.73 | 95.4 | 1.34 |
| RandomForest | after (+ real SMS, 5-fold CV) | 87.73 | 92.59 | 50.0 | 100.0 | 97.61 | 95.17 | 1.59 |