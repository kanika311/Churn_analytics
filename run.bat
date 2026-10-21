@echo off
echo Starting Customer Churn Analytics Streamlit Dashboard...
set PYTHON_EXE=%LOCALAPPDATA%\Python\bin\python.exe
if not exist "%PYTHON_EXE%" (
    python -m streamlit run app\streamlit_app.py
) else (
    "%PYTHON_EXE%" -m streamlit run app\streamlit_app.py
)
pause
