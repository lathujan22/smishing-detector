# FINAL DELIVERY — Multilingual SMS Smishing Detector (English · Sinhala · Tamil · Singlish · Tanglish)

Everything needed to show, reproduce and demo the research, in one place.
(Generated copies. The working project stays in the folders above this one.)

| Folder | Contents |
|---|---|
| `01_selected_models/best_ML_LinearSVC/` | **Best ML model** – three-class LinearSVC (`three_linearsvc.joblib`) |
| `01_selected_models/best_DL_XLM-R/` | **Best DL model** – fine-tuned XLM-R three-class checkpoint (1.1 GB) |
| `02_all_models/` | Every trained V2 model: binary SVMs, three-class LinearSVC / LogReg / RandomForest / GradientBoosting, CNN / BiLSTM / CNN-BiLSTM-Attention (`.pt`) |
| `03_results/` | All result tables (`REPORT.md`, `MODEL_SELECTION.md`, metrics CSVs, per-language, experiment C/D, configs with seeds + data checksums) |
| `04_figures/` | All figures: `v1_baseline/`, `scam_only/`, `three_class/`, `v2/` (paper-ready, 300 dpi) |
| `05_code/` | Source code: `src/`, `app/` (web app), `scripts/`, `notebooks/`, `docs/` |
| `06_web_app/` | How to run the web application + demo SMS samples |

## Selected models (rule fixed in advance: three-class formulation, ranked by held-out test macro-F1)

| | Model | Held-out test macro-F1 | Real SMS (n=163) scam recall | Real SMS non-scam flagged as scam |
|---|---|---|---|---|
| **Best ML** | LinearSVC (three-class) | 89.67 | 89.7 % | 17.9 % [11.8–24.6] |
| **Best DL** | XLM-R (three-class) | 90.83 | 89.7 % | 20.2 % [13.3–27.0] |

On real SMS the two are statistically tied (McNemar p = 0.71). The web app uses LinearSVC as the default and XLM-R as a second opinion.

## Run the web app
Double-click `start_app.bat` in the project folder (or run `python -m uvicorn app.main:app --port 8000`), then open http://127.0.0.1:8000

## Privacy
Real SMS text is NOT included in this folder (no data files, no per-message prediction files).
