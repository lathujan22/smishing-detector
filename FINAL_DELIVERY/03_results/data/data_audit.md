# V2 data audit

- Registry rows: 11199 (pool 11036, real benchmark 163)
- Families (exact + near-duplicate cosine >= 0.8 + template/translation groups): pool 5712, largest family 48 messages, families with >1 message: 1633
- **V1 split contamination:** 331 of 1589 v1 test messages share a family with v1 training messages -> v1 held-out test results are optimistic.
- Real benchmark vs pool: 0 pool messages share a family with a real benchmark message.
- Quarantined (identical text, conflicting labels): 29
- Synthetic or machine-translated rows in pool: 4932 (44.7%); human-labelled real-text rows: 6104

## Class by source

|                                 |   normal |   promotional |   scam |   All |
|:--------------------------------|---------:|--------------:|-------:|------:|
| ('pool', 'mendeley')            |     4834 |           454 |    543 |  5831 |
| ('pool', 'mt_nllb')             |     1740 |           219 |    822 |  2781 |
| ('pool', 'multilingual')        |     1003 |           172 |    976 |  2151 |
| ('pool', 'uci')                 |      207 |            66 |      0 |   273 |
| ('real_benchmark', 'real_form') |       36 |            98 |     29 |   163 |
| ('All', '')                     |     7820 |          1009 |   2370 | 11199 |

## Class by language family

|                             |   normal |   promotional |   scam |   All |
|:----------------------------|---------:|--------------:|-------:|------:|
| ('pool', 'EN')              |     5041 |           520 |    543 |  6104 |
| ('pool', 'SI')              |     1368 |           196 |    895 |  2459 |
| ('pool', 'TA')              |     1375 |           195 |    903 |  2473 |
| ('real_benchmark', 'EN')    |       23 |            54 |     12 |    89 |
| ('real_benchmark', 'SI')    |       12 |            38 |     17 |    67 |
| ('real_benchmark', 'TA')    |        0 |             4 |      0 |     4 |
| ('real_benchmark', 'mixed') |        1 |             2 |      0 |     3 |
| ('All', '')                 |     7820 |          1009 |   2370 | 11199 |

## Provenance

- **uci**: public dataset label (UCI SMS Spam Collection, human); public dataset; licence: CC BY 4.0 (UCI)
- **mendeley**: public dataset label (Mendeley SMS phishing dataset, human); public dataset; licence: CC BY 4.0 (Mendeley Data)
- **multilingual**: researcher-written template (synthetic text, label by design); template generation (src/gen_multilingual.py) + handwritten templates; licence: own work
- **mt_nllb**: label inherited from English source; text machine-translated (NLLB); machine translation of training messages; licence: derived from source dataset
- **real_form**: researcher label of real SMS (respondent label kept separately); Google Form / own phones; licence: research consent