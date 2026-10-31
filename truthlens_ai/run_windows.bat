@echo off
cd /d %~dp0
if not exist .venv (
  py -3.11 -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo TruthLens is starting. Open http://127.0.0.1:8000/ in your browser.
echo Keep this window open while using TruthLens.
start "" http://127.0.0.1:8000/
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
