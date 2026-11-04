@echo off
title Related-Work Summarizer
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Could not find .venv in this folder. Put run_app.bat in the project folder.
    pause & exit /b 1
)
call ".venv\Scripts\activate.bat"

python -c "import streamlit" 2>nul || (
    echo Installing Streamlit...
    pip install streamlit || (pause & exit /b 1)
)

if not exist "demo_data.json" (
    echo Preparing demo data - first run only, this takes several minutes...
    python precompute.py 50 || (echo Precompute failed. & pause & exit /b 1)
)

echo Starting the app - it will open in your browser...
streamlit run app.py
pause
