Write-Host "Starting Customer Churn Analytics Streamlit Dashboard..." -ForegroundColor Cyan
$pythonExe = "$env:LOCALAPPDATA\Python\bin\python.exe"
if (Test-Path $pythonExe) {
    & $pythonExe -m streamlit run app/streamlit_app.py
} else {
    streamlit run app/streamlit_app.py
}
