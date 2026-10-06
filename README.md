# Ridely AI — Future Fare Prediction & Smart Booking Advisor

Workflow:
Input -> Preprocessing -> Feature Engineering -> ML (Linear Regression, Random Forest, XGBoost)
-> DL (RNN, LSTM) -> Future Fare Prediction -> Platform Comparison -> Booking Recommendation

## 1. Create environment

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Generate data
```powershell
python src/generate_data.py
```

Creates:
`data/raw/ridely_historical.csv`

## 3. Train ML models
```powershell
python src/train_ml.py
```

Creates:
- `models/ml_models.joblib`
- `outputs/ml_metrics.csv`
- `outputs/ml_comparison.png`

## 4. Train RNN and LSTM
```powershell
python src/train_dl.py
```

Creates:
- `models/rnn_model.keras`
- `models/lstm_model.keras`
- `models/dl_scalers.joblib`
- `outputs/dl_metrics.csv`
- `outputs/dl_comparison.png`

## 5. Run application
```powershell
streamlit run app/app.py
```

## Important
The supplied project abstract describes historical/contextual fare prediction for Uber, Ola and Rapido. This implementation uses synthetic historical data for development/demo purposes. It does NOT scrape or claim live fares from those services.

For a real deployment, replace the data-generation step with legally permitted real data/API sources.
