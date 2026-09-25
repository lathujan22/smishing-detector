# Development Checklist — Smishing Detector

Tick `[x]` as we go. One step at a time, in this chat.

## PART A — 50% milestone: train + test on real SMS

### Step 1 · Project setup ✅
- [x] Repo structure (`src/ data/ lexicons/ notebooks/ models/ results/ api/ demo/`)
- [x] `requirements.txt`, `.gitignore`, README
- [x] `notebooks/00_colab_setup.ipynb`
- [x] Architecture + checklist (`docs/`)

### Step 2 · Datasets
- [ ] **YOU:** download UCI SMS Spam Collection → `data/raw/SMSSpamCollection`
      (https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- [ ] **YOU:** download Mendeley SMS Phishing Dataset → `data/raw/Dataset_5971.csv`
      (https://data.mendeley.com/datasets/f45bkkt8pr/1)
- [x] `src/data_prep.py` — merge, clean duplicates, label map, stratified 80/20 split
- [ ] Run `python -m src.data_prep` → `data/processed/train.csv`, `test.csv`, `data_report.txt`

### Step 3 · Real SMS test set (your proof)
- [ ] **YOU:** collect real SMS into `data/real_sms/real_test.csv` (template provided)
  - [ ] ≥ 80 real phishing/scam SMS (own phone, friends/family, Sri Lanka CERT / police / bank warning posts)
  - [ ] ≥ 80 real legitimate SMS (bank OTPs you received, delivery, telco, personal)
  - [ ] Mix: English, Sinhala, Tamil, code-switched (aim ≥ 20 per language)
  - [ ] Remove names, phone numbers, account numbers (replace with `<NAME>`, `<PHONE>`, `<ACC>`)
- [ ] Never used for training — test only

### Step 4 · Multilingual training data
- [ ] 300–600 Sinhala / Tamil / code-switched messages (translate English smishing + ham templates, native-speaker check)
- [ ] `data/raw/multilingual_train.csv` (same columns as template)

### Step 5 · Preprocessing + features
- [ ] `src/preprocess.py` — clean, script tag, emoji→token, URL/phone extract + mask
- [ ] `lexicons/` urgency / impersonation / risky TLD + shorteners (EN + SI + TA)
- [ ] `src/features.py` — flags + urgency / impersonation / link-risk sub-scores
- [ ] Unit tests on 10 hand-made messages

### Step 6 · Baselines
- [ ] TF-IDF + Naive Bayes, SVM, Random Forest
- [ ] 10-fold stratified CV on train, final score on test
- [ ] `results/baselines.csv`

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
