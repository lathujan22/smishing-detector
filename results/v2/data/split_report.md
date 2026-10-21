# V2 split report

- Pool (non-quarantined) 11007 -> train 9433 / held-out test 1574 (14.3%)
- Families: train 4899, test 813, shared **0**
- Test -> train nearest char-n-gram cosine: max 0.802, 99th pct 0.789, median 0.334 (v1 had 354 test messages >= 0.8)
- Synthetic / MT share: train 44.7%, test 44.8%

## Train

| lang_family   |   normal |   promotional |   scam |   All |
|:--------------|---------:|--------------:|-------:|------:|
| EN            |     4321 |           440 |    459 |  5220 |
| SI            |     1173 |           166 |    762 |  2101 |
| TA            |     1179 |           165 |    768 |  2112 |
| All           |     6673 |           771 |   1989 |  9433 |

## Held-out test

| lang_family   |   normal |   promotional |   scam |   All |
|:--------------|---------:|--------------:|-------:|------:|
| EN            |      720 |            73 |     76 |   869 |
| SI            |      195 |            27 |    129 |   351 |
| TA            |      196 |            27 |    131 |   354 |
| All           |     1111 |           127 |    336 |  1574 |