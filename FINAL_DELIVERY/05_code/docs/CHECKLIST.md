# Development Checklist — Smishing Detector

Tick `[x]` as we go. One step at a time, in this chat.

## PART A — 50% milestone: train + test on real SMS

### Step 1 · Project setup ✅
- [x] Repo structure (`src/ data/ lexicons/ notebooks/ models/ results/ api/ demo/`)
- [x] `requirements.txt`, `.gitignore`, README
- [x] `notebooks/00_colab_setup.ipynb`
- [x] Architecture + checklist (`docs/`)

### Step 2 · Datasets
- [x] **YOU:** download UCI SMS Spam Collection → `data/raw/SMSSpamCollection`
      (https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- [x] **YOU:** download Mendeley SMS Phishing Dataset → `data/raw/Dataset_5971.csv`
      (https://data.mendeley.com/datasets/f45bkkt8pr/1)
- [x] `src/data_prep.py` — merge, clean duplicates, label map, stratified 80/20 split
- [x] Run `python -m src.data_prep` → `data/processed/train.csv`, `test.csv`, `data_report.txt`
      (with multilingual: 7,894 msgs → train 6,343 / test 1,551; 1,519 smishing, 692 spam, 5,683 ham)

### Step 3 · Real SMS test set (your proof)
- [x] Google Form designed: `docs/GOOGLE_FORM.md` (trilingual, anonymous) + importer `src/import_form.py` (masks phone/NIC/ACC, dedupes)
- [x] **YOU:** create the form + share - day 1: 57 responses → 50 unique real SMS (16 smishing / 30 spam / 4 ham)
- [x] Researcher re-labelling in `data/real_sms/relabel.csv` (respondent vs researcher agreement 75%, kappa 0.55)
- [ ] **YOU:** keep sharing - NEED: normal/OTP/bank-alert SMS (only 4 ham!), Sinhala & Tamil script, Tanglish
- [ ] OCR screenshot-only responses (e.g. "Included in ss")
- [ ] **YOU:** download responses → `data/real_sms/form_responses.csv` → `python -m src.import_form`
- [ ] **YOU:** collect real SMS into `data/real_sms/real_test.csv` (template provided)
  - [ ] ≥ 80 real phishing/scam SMS (own phone, friends/family, Sri Lanka CERT / police / bank warning posts)
  - [ ] ≥ 80 real legitimate SMS (bank OTPs you received, delivery, telco, personal)
  - [ ] Mix: English, Sinhala, Tamil, code-switched (aim ≥ 20 per language)
  - [ ] Remove names, phone numbers, account numbers (replace with `<NAME>`, `<PHONE>`, `<ACC>`)
- [ ] Never used for training — test only

### Step 4 · Multilingual training data
No open Sinhala SMS dataset exists; Tamil one (IEEE DataPort "Spam SMS in Dravidian Languages") is subscription-only.
So we build it in two layers:
- [x] Layer A - Sri Lankan messages, `python -m src.gen_multilingual` → `data/raw/multilingual_train.csv`
      1,790 msgs from 416 patterns (206 templates x 8 fills + 210 one-off handwritten in `multilingual_handwritten.csv`)
      si 470 · ta 470 · Singlish 425 · Tanglish 425 (smishing 976 / spam 172 / ham 642)
      split by template in data_prep (84 patterns held out for test → no near-copy leakage)
- [ ] **YOU / native speaker:** review wording - fix the TEMPLATES in `src/gen_multilingual.py` (one fix corrects all 8 variants), then re-run it + data_prep
- [ ] Layer B - run `notebooks/01_translate_nllb.ipynb` on Colab GPU → `data/raw/multilingual_mt.csv` (~2,000 NLLB-200 translations of English TRAIN rows; added to TRAIN only, no test leakage)
- [ ] **YOU:** spot-check `data/raw/mt_review_sample.csv` (25 per language), note error rate for the report
- [ ] Optional: ask supervisor/library for IEEE DataPort access (Tamil + Tanglish real spam)

### Step 5 · Preprocessing + features ✅
- [x] `src/preprocess.py` — normalise (keeps Sinhala ZWJ, strips "DIALOG WARNING - SPAM" leak), URL/phone/money/email/emoji extract + mask, language tag incl. Singlish/Tanglish
- [x] `lexicons/` urgency · impersonation · credential · request · safety phrases · shorteners · risky TLDs · official domains · brand keywords (EN + SI + TA + romanised)
- [x] `src/features.py` — 8 model flags + urgency / impersonation / link-risk sub-scores; `python -m src.features --build` → `data/processed/*_feat.csv`
- [x] Unit tests: 10 hand-made messages, all pass (`python -m unittest -v tests.test_features`)
- [x] Sanity check `results/feature_summary.txt`: smishing > spam > ham on every feature (real SMS link-risk 0.67 / 0.16 / 0.05); language detection vs researcher 93%

### Step 6 · Baselines ✅
- [x] `src/baselines.py` — TF-IDF (word 1-2 + char 2-5 grams) + Naive Bayes, linear SVM, Random Forest, class-balanced
- [x] 10-fold **grouped** stratified CV (template variants never split across folds), final score on test + real SMS
- [x] `src/evaluate.py` — Acc/P/R/F1/ROC-AUC/FPR, per-language split, McNemar helper
- [x] `results/baselines.csv`, `results/baselines_by_language.csv`
- Best = SVM: CV F1 0.976 · test F1 0.922 · EN F1 0.971 (FPR 0.3%) · SI/TA recall 100% but FPR 26-31% on UNSEEN normal patterns · real SMS F1 0.971
- [ ] Fix SI/TA false alarms BEFORE deep models:
  - [x] **YOU:** Colab → `notebooks/01_translate_nllb.ipynb` → 2,781 NLLB translations (si 1,391 / ta 1,390; 1,740 ham) added to TRAIN only (train 6,666 → 9,447)
    - Sinhala joiner fix: NLLB wrote 'ප් රමාණය' for 'ප්‍රමාණය' in 31% of rows → auto re-joined (5% left)
    - CV groups: each translation grouped with its English source
    - Result: RF SI false alarms 26% → 9%, TA 21% → 16%; SVM TA 31% → 21% (SI 26% → 33%). Remaining false alarms are mostly Singlish/Tanglish (45/55) - MT can't produce romanised text
  - [ ] **YOU:** spot-check `data/raw/mt_review_sample.csv` (fill ok_yes_no for 60 rows)
  - [ ] **YOU:** fill `data/real_sms/my_messages.txt` (or SMS Backup & Restore .xml) with real normal SI/TA/EN SMS → `src/import_bulk.py`
  - [ ] Re-run data_prep → features → baselines, compare SI/TA false-alarm rate
- Note: added 44 extra SI/TA/Singlish/Tanglish normal templates (bank alerts, bills, tickets, school, family) → multilingual set 2,151 msgs / 474 templates

### Step 7 · Deep models (Colab GPU)
- [ ] XLM-R SentencePiece tokenisation
- [ ] Standalone CNN, standalone BiLSTM
- [ ] **Hybrid CNN-BiLSTM-Attention** (+ feature flags), class weights, dropout, early stopping
- [ ] Save `models/hybrid.keras` + tokeniser config
- [ ] `results/deep_models.csv`, training curves

### Step 8 · Evaluation
- [ ] Comparison table: Acc, Precision, Recall, F1, ROC-AUC, FPR (all 6 models)
- [ ] Confusion matrices + ROC plot
- [ ] McNemar test: hybrid vs best baseline
- [ ] **Real SMS test** — results split by EN / SI / TA / mixed
- [ ] Error analysis: 10 wrong predictions explained

### Step 9 · Severity engine v1
- [ ] `src/severity.py` — WSM formula + tiers
- [ ] Reproduce Appendix E example (86.5 → Critical)
- [ ] Tier distribution on real SMS set

### Step 10 · 50% demo pack
- [ ] `src/predict.py` — text → JSON packet
- [ ] `notebooks/04_real_sms_demo.ipynb` — live demo on real messages
- [ ] Slides: architecture, dataset stats, comparison table, real-SMS results, demo screenshots

**Targets:** English F1 ≥ 0.95 · multilingual F1 ≥ 0.85 · FPR < 3%

---

## PART B — 100% (after 50% review)
- [ ] Attention explainability: top-k tokens + highlighted text
- [ ] Severity ground truth + tier accuracy ≥ 88% + Cohen's Kappa
- [ ] Weight sensitivity analysis (±0.05 on each weight)
- [ ] FastAPI `/predict`, `/batch` + latency test (≤150 ms/msg, 5,000 msg/min)
- [ ] Demo dashboard + ranked threat table
- [ ] Docker image
- [ ] JSON schema agreed + integration with Final Severity Aggregator
- [ ] Final report + research paper
