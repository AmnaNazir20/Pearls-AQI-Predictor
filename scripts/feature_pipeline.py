"""
Feature pipeline (runs hourly via GitHub Actions).

Fetches recent air-quality and weather data from Open-Meteo, engineers the
model features, and updates the feature store (feature_store.csv) by merging
new records with the existing ones.
"""

import numpy as np
import pandas as pd
import requests
from datetime import date, timedelta

CITY_NAME = "Islamabad"
LAT, LON = 33.6844, 73.0479
FEATURE_STORE = "feature_store.csv"
RECENT_DAYS = 35   # enough history for 48h lags and 24h diffs


def fetch_recent():
    end_date = date.today()
    start_date = end_date - timedelta(days=RECENT_DAYS)

    aq = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": LAT, "longitude": LON,
            "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
            "start_date": str(start_date), "end_date": str(end_date), "timezone": "auto",
        }, timeout=60,
    ).json()
    df_aq = pd.DataFrame(aq["hourly"])

    wx = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LAT, "longitude": LON,
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                      "wind_direction_10m,precipitation,surface_pressure",
            "past_days": RECENT_DAYS, "forecast_days": 1, "timezone": "auto",
        }, timeout=60,
    ).json()
    df_wx = pd.DataFrame(wx["hourly"])

    df = pd.merge(df_aq, df_wx, on="time", how="inner")
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).sort_values("datetime").reset_index(drop=True)
    return df.dropna(subset=["pm2_5"]).reset_index(drop=True)


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


def build_features(df):
    df = df.copy()
    df["aqi"] = df["pm2_5"].apply(calculate_aqi)
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
    return df.dropna().reset_index(drop=True)


def main():
    print("Fetching recent data...")
    new = build_features(fetch_recent())

    try:
        existing = pd.read_csv(FEATURE_STORE, parse_dates=["datetime"])
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["datetime"], keep="last")
        combined = combined.sort_values("datetime").reset_index(drop=True)
    except FileNotFoundError:
        combined = new

    combined.to_csv(FEATURE_STORE, index=False)
    print(f"Feature store updated: {len(combined)} rows in {FEATURE_STORE}")


if __name__ == "__main__":
    main()
