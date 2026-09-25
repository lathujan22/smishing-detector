# Smishing Detector — Component 2, Project J26-IT-424

Hybrid CNN-BiLSTM multilingual (English / Sinhala / Tamil) SMS phishing detection
with severity-aware risk scoring (Low / Medium / High / Critical).

Author: Lathujan R (IT23214866) · Supervisor: Ms. Fathima Fanoon · SLIIT

## Folder layout
| Folder | Contents |
|---|---|
| data/raw | Mendeley + UCI datasets as downloaded (not committed) |
| data/processed | train.csv / test.csv — fixed stratified 80/20 split |
| data/real_sms | real_test.csv — anonymised real SMS incl. Sinhala/Tamil |
| lexicons | urgency, impersonation and URL-risk word lists |
| src | data_prep, preprocess, features, model, severity, predict |
| notebooks | 00 setup, 01 EDA, 02 baselines, 03 CNN-BiLSTM, 04 real-SMS test |
| models | saved model + tokeniser files (not committed) |
| results | metric tables and plots |
| api | FastAPI service (`/predict`, `/batch`) |
| demo | paste-an-SMS demo page |

## Quick start (laptop)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Training (Google Colab)
Upload this folder to Google Drive as `MyDrive/smishing-detector`, then open
`notebooks/00_colab_setup.ipynb` in Colab (Runtime → GPU).
