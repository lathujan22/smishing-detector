@echo off
REM Starts the smishing detector web app at http://127.0.0.1:8000
cd /d %~dp0
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
start "" http://127.0.0.1:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
