# 🚀 Quick Start Script
# This script activates the virtual environment and starts the app

Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   NeuroStock: Stock Price Prediction App" -ForegroundColor Yellow
Write-Host "   🔶 Now with Advanced Neuro-Fuzzy Analysis!" -ForegroundColor Magenta
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Activating virtual environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "Starting Streamlit app..." -ForegroundColor Green
Write-Host ""
Write-Host "🚀 Features:" -ForegroundColor Yellow
Write-Host "   • Neural Network Predictions (LSTM)" -ForegroundColor White
Write-Host "   • Advanced Neuro-Fuzzy Analysis" -ForegroundColor White
Write-Host "   • Model Performance Comparison" -ForegroundColor White
Write-Host "   • Technical Indicators Analysis" -ForegroundColor White
Write-Host "   • Interactive Charts and Visualizations" -ForegroundColor White
Write-Host ""
Write-Host "The app will open at: http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 Tip: Enable 'Advanced Analysis' in the sidebar!" -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop the app, press Ctrl+C" -ForegroundColor Red
Write-Host ""

streamlit run app.py