"""
Training pipeline (runs daily via GitHub Actions).

Fetches two years of history from Open-Meteo, engineers features and targets,
trains multi-output models (XGBoost, LightGBM, Random Forest), selects the best
by average R-squared, and saves a compact LightGBM model to aqi_models.pkl.
"""

import numpy as np
import pandas as pd
import requests
import joblib
from datetime import date, timedelta

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CITY_NAME = "Islamabad"
LAT, LON = 33.6844, 73.0479
DAYS_BACK = 730
MODEL_PATH = "aqi_models.pkl"


def calculate_aqi(pm25):
    if pd.isna(pm25) or pm25 < 0:
        return 0
    bp = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
          (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
          (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
    for c_low, c_high, i_low, i_high in bp:
        if pm25 <= c_high:
            return round((i_high - i_low) / (c_high - c_low) * (pm25 - c_low) + i_low)
    return 500


def fetch_history():
    end_date = date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=DAYS_BACK)
    aq = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LAT, "longitude": LON,
            "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "start_date": str(start_date), "end_date": str(end_date), "timezone": "auto",
        }, timeout=120,
    ).json()
    df_aq = pd.DataFrame(aq["hourly"])
    wx = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": LAT, "longitude": LON,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                      "wind_direction_10m,precipitation,surface_pressure",
            "start_date": str(start_date), "end_date": str(end_date), "timezone": "auto",
        }, timeout=120,
    ).json()
    df_wx = pd.DataFrame(wx["hourly"])
    df = pd.merge(df_aq, df_wx, on="time", how="inner")
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).sort_values("datetime").reset_index(drop=True)
    df = df.drop(columns=[c for c in df.columns if df[c].isna().all()])
    return df.dropna(subset=["pm2_5"]).reset_index(drop=True)


def build_features(df):
    df = df.copy()
    df["pm25_log"] = np.log1p(df["pm2_5"])
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f"pm25_lag_{lag}h"] = df["pm25_log"].shift(lag)
    for w in [3, 6, 12, 24]:
        df[f"pm25_roll_mean_{w}h"] = df["pm25_log"].rolling(w).mean()
        df[f"pm25_roll_std_{w}h"] = df["pm25_log"].rolling(w).std()
    for col in ["pm10", "nitrogen_dioxide", "carbon_monoxide"]:
        df[f"{col}_lag_1h"] = df[col].shift(1)
        df[f"{col}_lag_6h"] = df[col].shift(6)
    df["wind_x"] = df["wind_speed_10m"] * np.cos(np.deg2rad(df["wind_direction_10m"]))
    df["wind_y"] = df["wind_speed_10m"] * np.sin(np.deg2rad(df["wind_direction_10m"]))
    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["datetime"].dt.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["datetime"].dt.month - 1) / 12)
    df["stagnation_index"] = df["pm25_log"] / (df["wind_speed_10m"] + 1)
    df["is_weekend"] = df["datetime"].dt.dayofweek.isin([5, 6]).astype(int)
    df["temp_diff_24h"] = df["temperature_2m"].diff(24)
    df["hum_diff_24h"] = df["relative_humidity_2m"].diff(24)
    df["target_pm25_24h"] = df["pm2_5"].shift(-24)
    df["target_pm25_48h"] = df["pm2_5"].shift(-48)
    df["target_pm25_72h"] = df["pm2_5"].shift(-72)
    return df.dropna().reset_index(drop=True)


def build_models():
    models = {"RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=3,
        max_features="sqrt", random_state=42, n_jobs=-1)}
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=800, max_depth=8, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=-1, tree_method="hist")
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(
            n_estimators=800, max_depth=10, num_leaves=31, learning_rate=0.05,
            subsample=0.8, random_state=42, n_jobs=-1, verbosity=-1)
    except ImportError:
        pass
    return models


def main():
    print("Fetching two years of history...")
    df = build_features(fetch_history())

    TARGETS = ["target_pm25_24h", "target_pm25_48h", "target_pm25_72h"]
    EXCLUDE = ["datetime"] + TARGETS
    FEATURES = [c for c in df.columns if c not in EXCLUDE]

    X, y = df[FEATURES], df[TARGETS]
    split = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    fitted, scores = {}, {}
    for name, base in build_models().items():
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        preds = np.maximum(model.predict(X_val), 0)
        avg_r2 = np.mean([r2_score(y_val.iloc[:, i], preds[:, i]) for i in range(3)])
        fitted[name] = model
        scores[name] = avg_r2
        print(f"{name}: average R2 = {avg_r2:.3f}")

    best_name = max(scores, key=scores.get)
    print(f"Best by average R2: {best_name} ({scores[best_name]:.3f})")

    # Save the compact LightGBM model for deployment if available
    deploy_name = "LightGBM" if "LightGBM" in fitted else best_name
    joblib.dump(
        {"model": fitted[deploy_name], "features": FEATURES, "model_type": deploy_name},
        MODEL_PATH, compress=3,
    )
    print(f"Saved model '{deploy_name}' to {MODEL_PATH}")


if __name__ == "__main__":
    main()
