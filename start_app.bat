@echo off
echo.
echo ===============================================
echo   NeuroStock: Stock Price Prediction App
echo   🔶 Now with Advanced Neuro-Fuzzy Analysis!
echo ===============================================
echo.
echo Activating virtual environment...
call .venv\Scripts\activate

echo.
echo Starting Streamlit app...
echo.
echo 🚀 Features:
echo    • Neural Network Predictions (LSTM)
echo    • Advanced Neuro-Fuzzy Analysis
echo    • Model Performance Comparison  
echo    • Technical Indicators Analysis
echo    • Interactive Charts and Visualizations
echo.
echo The app will open at: http://localhost:8501
echo.
echo 💡 Tip: Enable "Advanced Analysis" in the sidebar!
echo.
echo To stop the app, press Ctrl+C
echo.

streamlit run app.py

pause