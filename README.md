# Pearls AQI Predictor

*Islamabad Air Quality Forecasting System*

**Author:** Amna Nazir | **Program:** 10P Shine Internship Program|**Domain:** Data Science

An end-to-end machine learning system that forecasts the Air Quality Index (AQI) for Islamabad over the next three days. The project fetches live air-quality and weather data, engineers predictive features, trains multi-output regression models, and serves the forecast through an interactive Streamlit dashboard.

## Live Demo

The dashboard is deployed on Streamlit Community Cloud and is publicly accessible:

**https://aqi-dashboard-iqynzk7oxdupfvvpjjbrhj.streamlit.app/**

## Overview

Air pollution in Islamabad varies sharply with weather and season. This system predicts PM2.5 concentrations 24, 48, and 72 hours ahead, converts them to the US EPA Air Quality Index, and presents the result with health alerts. All data is sourced from the free Open-Meteo API, so no API key is required.

## Features

- Two years of hourly air-quality and weather history from Open-Meteo
- US EPA AQI computation from PM2.5 concentrations
- Feature engineering: lag features, rolling statistics, wind vectors, cyclical time encodings, a stagnation index, and weather-tendency features
- Multi-output models (XGBoost, LightGBM, Random Forest) compared and selected automatically
- Model evaluation with MAE, RMSE, and R-squared for each horizon
- SHAP-based feature importance analysis
- Interactive dashboard with current conditions, a three-day forecast, health alerts, and a trend chart

## Technology Stack

| Component        | Tool                            |
|------------------|---------------------------------|
| Data source      | Open-Meteo API                  |
| Data handling    | pandas, NumPy                   |
| Models           | scikit-learn, XGBoost, LightGBM |
| Explainability   | SHAP                            |
| Dashboard        | Streamlit                       |
| Model storage    | joblib                          |

## Project Structure

```
aqi-dashboard/
|- app.py                          # Streamlit dashboard
|- aqi_models.pkl                  # Trained multi-output model
|- requirements.txt                # Python dependencies
|- Pearls AQI Predictor.ipynb   # Full training pipeline 
|- README.md
```

## How It Works

1. **Data acquisition** - hourly pollutant and weather data are fetched from Open-Meteo and merged on the timestamp.
2. **Feature engineering** - the raw data is transformed into lag, rolling, wind-vector, cyclical, and interaction features.
3. **Model training** - three multi-output regressors are trained with a chronological 80/20 split; the best is selected by average R-squared.
4. **Forecasting** - the most recent feature row is used to predict PM2.5 at three horizons, which is converted to AQI.
5. **Dashboard** - the saved model is loaded by the Streamlit app, which fetches live data and displays the forecast.

## Running Locally

1. Install Python 3.10 or later.
2. Place `app.py`, `requirements.txt`, and `aqi_models.pkl` in one folder.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Launch the dashboard:
   ```
   streamlit run app.py
   ```
5. Open http://localhost:8501 in a browser.

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub.
2. Sign in at https://share.streamlit.io with GitHub.
3. Create a new app, select this repository, set the branch to `main` and the main file to `app.py`.
4. Deploy. A public URL is generated in a few minutes.

## Model Performance

Validation results on a held-out chronological test set (approximate):

| Horizon | MAE   | R-squared |
|---------|-------|-----------|
| 24h     | ~8.7  | ~0.706     |
| 48h     | ~9.83  | ~0.62     |
| 72h     | ~10.20| ~0.602     |

Accuracy is highest for the 24-hour horizon and decreases with the forecast distance, which is expected for air-quality forecasting. Islamabad's pollution is more volatile than coastal cities, so its forecasts are inherently harder than for smoother locations.

## Data Source

All data is provided by Open-Meteo (https://open-meteo.com), which offers free air-quality and historical weather APIs without authentication.

## Conclusion
This project delivers a complete, automated AQI forecasting system for Islamabad. Two years of air-quality and weather data are used to train multi-output models that predict PM2.5 for the next 24, 48, and 72 hours, which are converted to the US EPA Air Quality Index. The 24-hour forecast achieves the strongest accuracy, with performance tapering at longer horizons, as is typical for air-quality forecasting. The trained model is served through an interactive Streamlit dashboard, while scheduled GitHub Actions keep the feature store and model up to date without manual intervention. Together these components demonstrate an end-to-end machine learning pipeline from data ingestion to live deployment.
## Future Work

Migrate the feature store to a managed cloud database (e.g. MongoDB Atlas or Hopsworks)
Add sequence models (LSTM) and compare against the tree-based ensemble
Support multiple cities with per-city models
Send email or SMS alerts when hazardous AQI is forecast
