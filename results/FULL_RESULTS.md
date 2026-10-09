# Full results — baseline models (Step 6)

Train 9,447 msgs (incl. 2,781 NLLB translations) · Test 1,589 (patterns held out) · Real SMS 108. Positive class = scam OR promo spam. Values in %.

## 1. 10-fold grouped cross-validation (training set)

| Model        |   cv_accuracy |   cv_precision |   cv_recall |   cv_f1 |   cv_f1_std |   cv_roc_auc |
|:-------------|--------------:|---------------:|------------:|--------:|------------:|-------------:|
| NaiveBayes   |          92.1 |           82.4 |        94.4 |    87.9 |         2.2 |         98.1 |
| SVM          |          97.9 |           97   |        96.2 |    96.6 |         2.3 |         99.6 |
| RandomForest |          97   |           97.4 |        92.6 |    94.9 |         1.7 |         99.3 |

## 2. Overall — test set and real SMS

| Model        | Set            |    n |   acc |   precision |   recall |   f1 |   roc_auc |   fpr |   tp |   fp |   tn |   fn |
|:-------------|:---------------|-----:|------:|------------:|---------:|-----:|----------:|------:|-----:|-----:|-----:|-----:|
| NaiveBayes   | Test (1,589)   | 1589 |  89.9 |        73.3 |     92.9 | 81.9 |      97.5 |  11.1 |  365 |  133 | 1063 |   28 |
| NaiveBayes   | Real SMS (108) |  108 |  75   |        90.1 |     79.4 | 84.4 |      73.5 |  50   |   73 |    8 |    8 |   19 |
| SVM          | Test (1,589)   | 1589 |  96   |        87.5 |     98   | 92.4 |      99.7 |   4.6 |  385 |   55 | 1141 |    8 |
| SVM          | Real SMS (108) |  108 |  90.7 |        91.8 |     97.8 | 94.7 |      89.7 |  50   |   90 |    8 |    8 |    2 |
| RandomForest | Test (1,589)   | 1589 |  96.6 |        93.6 |     92.6 | 93.1 |      99.5 |   2.1 |  364 |   25 | 1171 |   29 |
| RandomForest | Real SMS (108) |  108 |  92.6 |        94.7 |     96.7 | 95.7 |      89.6 |  31.2 |   89 |    5 |   11 |    3 |

## 3. By language

| Model        | Set            | lang   |    n |   acc |   precision |   recall |    f1 |   fpr |   tp |   fp |   tn |   fn |
|:-------------|:---------------|:-------|-----:|------:|------------:|---------:|------:|------:|-----:|-----:|-----:|-----:|
| NaiveBayes   | Test (1,589)   | EN     | 1221 |  97.5 |        98.9 |     86.9 |  92.5 |   0.2 |  185 |    2 | 1006 |   28 |
| NaiveBayes   | Test (1,589)   | SI     |  180 |  70   |        62.5 |    100   |  76.9 |  60   |   90 |   54 |   36 |    0 |
| NaiveBayes   | Test (1,589)   | TA     |  188 |  59   |        53.9 |    100   |  70   |  78.6 |   90 |   77 |   21 |    0 |
| NaiveBayes   | Real SMS (108) | EN     |   53 |  60.4 |        87.9 |     63   |  73.4 |  57.1 |   29 |    4 |    3 |   17 |
| NaiveBayes   | Real SMS (108) | SI     |   51 |  88.2 |        90.9 |     95.2 |  93   |  44.4 |   40 |    4 |    5 |    2 |
| NaiveBayes   | Real SMS (108) | TA     |    2 | 100   |       100   |    100   | 100   | nan   |    2 |    0 |    0 |    0 |
| NaiveBayes   | Real SMS (108) | mixed  |    2 | 100   |       100   |    100   | 100   | nan   |    2 |    0 |    0 |    0 |
| SVM          | Test (1,589)   | EN     | 1221 |  99   |        98.1 |     96.2 |  97.2 |   0.4 |  205 |    4 | 1004 |    8 |
| SVM          | Test (1,589)   | SI     |  180 |  83.3 |        75   |    100   |  85.7 |  33.3 |   90 |   30 |   60 |    0 |
| SVM          | Test (1,589)   | TA     |  188 |  88.8 |        81.1 |    100   |  89.6 |  21.4 |   90 |   21 |   77 |    0 |
| SVM          | Real SMS (108) | EN     |   53 |  86.8 |        88.2 |     97.8 |  92.8 |  85.7 |   45 |    6 |    1 |    1 |
| SVM          | Real SMS (108) | SI     |   51 |  94.1 |        95.4 |     97.6 |  96.5 |  22.2 |   41 |    2 |    7 |    1 |
| SVM          | Real SMS (108) | TA     |    2 | 100   |       100   |    100   | 100   | nan   |    2 |    0 |    0 |    0 |
| SVM          | Real SMS (108) | mixed  |    2 | 100   |       100   |    100   | 100   | nan   |    2 |    0 |    0 |    0 |
| RandomForest | Test (1,589)   | EN     | 1221 |  98.9 |        99.5 |     93.9 |  96.6 |   0.1 |  200 |    1 | 1007 |   13 |
| RandomForest | Test (1,589)   | SI     |  180 |  90.6 |        91   |     90   |  90.5 |   8.9 |   81 |    8 |   82 |    9 |
| RandomForest | Test (1,589)   | TA     |  188 |  87.8 |        83.8 |     92.2 |  87.8 |  16.3 |   83 |   16 |   82 |    7 |
| RandomForest | Real SMS (108) | EN     |   53 |  86.8 |        89.8 |     95.6 |  92.6 |  71.4 |   44 |    5 |    2 |    2 |
| RandomForest | Real SMS (108) | SI     |   51 |  98   |       100   |     97.6 |  98.8 |   0   |   41 |    0 |    9 |    1 |
| RandomForest | Real SMS (108) | TA     |    2 | 100   |       100   |    100   | 100   | nan   |    2 |    0 |    0 |    0 |
| RandomForest | Real SMS (108) | mixed  |    2 | 100   |       100   |    100   | 100   | nan   |    2 |    0 |    0 |    0 |

## 4. By message type (% classified correctly)

| Model        | Class    |   Real SMS (108) |   Test (1,589) |
|:-------------|:---------|-----------------:|---------------:|
| NaiveBayes   | ham      |             50   |           88.9 |
| NaiveBayes   | smishing |             89.5 |           98.5 |
| NaiveBayes   | spam     |             76.7 |           81.2 |
| RandomForest | ham      |             68.8 |           97.9 |
| RandomForest | smishing |             89.5 |           98.5 |
| RandomForest | spam     |             98.6 |           80.5 |
| SVM          | ham      |             50   |           95.4 |
| SVM          | smishing |            100   |          100   |
| SVM          | spam     |             97.3 |           93.8 |

## 5. Effect of adding translations (test set, before → after)

| model        | lang   |   f1_before |   f1_after |   fpr_before |   fpr_after |
|:-------------|:-------|------------:|-----------:|-------------:|------------:|
| NaiveBayes   | EN     |        94.7 |       92.5 |          0.6 |         0.2 |
| NaiveBayes   | SI     |        76.9 |       76.9 |         60   |        60   |
| NaiveBayes   | TA     |        68.7 |       70   |         83.7 |        78.6 |
| SVM          | EN     |        97.1 |       97.2 |          0.3 |         0.4 |
| SVM          | SI     |        88.7 |       85.7 |         25.6 |        33.3 |
| SVM          | TA     |        85.7 |       89.6 |         30.6 |        21.4 |
| RandomForest | EN     |        96.1 |       96.6 |          0.1 |         0.1 |
| RandomForest | SI     |        83.5 |       90.5 |         25.6 |         8.9 |
| RandomForest | TA     |        84.4 |       87.8 |         21.4 |        16.3 |

fpr = false-alarm rate (normal messages wrongly flagged). Lower is better. Target: F1 ≥ 95% (EN), ≥ 85% (SI/TA), FPR < 3%.
