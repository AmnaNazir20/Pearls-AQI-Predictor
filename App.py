"""
AQI Predictor — Streamlit Dashboard
====================================
Loads the trained model (aqi_models.pkl), fetches LIVE data from Open-Meteo
(no API key needed), rebuilds the same features used in training, predicts the
next 3 days of PM2.5 -> AQI, and shows a clean dashboard with alerts.

Run locally:   streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import streamlit as st

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CITY_NAME = "Islamabad"
LAT, LON = 33.6844, 73.0479
MODEL_PATH = "aqi_models.pkl"

st.set_page_config(page_title=f"{CITY_NAME} AQI Forecast",
                   page_icon="🌫️", layout="wide")

# ----------------------------------------------------------------------------
# STYLING
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;800&family=Inter:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Sora', sans-serif; letter-spacing: -0.02em; }
.block-container { padding-top: 2rem; max-width: 1100px; }
.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 18px; padding: 28px 32px; color: #fff; margin-bottom: 22px;
}
.hero h1 { margin: 0; font-size: 2.1rem; }
.hero p  { margin: 6px 0 0; opacity: .8; }
.card {
    border-radius: 16px; padding: 20px; color: #fff; text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,.12); height: 100%;
}
.card .day   { font-size: .9rem; opacity: .9; font-weight: 500; }
.card .aqi   { font-family: 'Sora', sans-serif; font-size: 3rem; font-weight: 800; line-height: 1.1; }
.card .cat   { font-size: .95rem; font-weight: 600; margin-top: 4px; }
.card .pm    { font-size: .8rem; opacity: .85; margin-top: 8px; }
.now-badge {
    display:inline-block; padding: 4px 12px; border-radius: 999px;
    font-size:.8rem; font-weight:600; color:#fff;
}
.footnote { color:#888; font-size:.8rem; margin-top: 24px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# AQI HELPERS
# ----------------------------------------------------------------------------
def calculate_aqi(pm25):
    if pm25 is None or pm25 < 0:
        return 0
    bp = [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
          (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
          (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)]
    for c_low, c_high, i_low, i_high in bp:
        if pm25 <= c_high:
            return round((i_high - i_low) / (c_high - c_low) * (pm25 - c_low) + i_low)
    return 500


def aqi_category(a):
    if a <= 50:  return "Good"
    if a <= 100: return "Moderate"
    if a <= 150: return "Unhealthy for Sensitive"
    if a <= 200: return "Unhealthy"
    if a <= 300: return "Very Unhealthy"
    return "Hazardous"


def aqi_color(a):
    if a <= 50:  return "#1a9850"
    if a <= 100: return "#d6a000"
    if a <= 150: return "#e8590c"
    if a <= 200: return "#d12f2f"
    if a <= 300: return "#7048a8"
    return "#841a1a"


# ----------------------------------------------------------------------------
# DATA + FEATURES
# ----------------------------------------------------------------------------
@st.cache_data(ttl=1800)   # 30 min cache
def fetch_live_data():
    """Recent hourly air-quality + weather (past 15 days) from Open-Meteo."""
    aq = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": LAT, "longitude": LON,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "past_days": 15, "forecast_days": 1, "timezone": "auto",
    }, timeout=30).json()
    df_aq = pd.DataFrame(aq["hourly"])

    wx = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": LAT, "longitude": LON,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                  "wind_direction_10m,precipitation,surface_pressure",
        "past_days": 15, "forecast_days": 1, "timezone": "auto",
    }, timeout=30).json()
    df_wx = pd.DataFrame(wx["hourly"])

    df = pd.merge(df_aq, df_wx, on="time", how="inner")
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).sort_values("datetime").reset_index(drop=True)
    return df


def build_features(df):
    """Reproduce EXACTLY the feature engineering used in training."""
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
    return df


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"], bundle.get("model_type", "model")


# ----------------------------------------------------------------------------
# APP
# ----------------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>🌫️ {CITY_NAME} Air Quality Forecast</h1>
  <p>Machine-learning predicted AQI for the next 3 days · powered by Open-Meteo</p>
</div>
""", unsafe_allow_html=True)

# Load model
try:
    model, FEATURES, model_type = load_model()
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' nahi mili. Notebook se `aqi_models.pkl` is folder "
             "mein rakho, phir app refresh karo.")
    st.stop()

# Fetch + features
with st.spinner("Live data fetch ho raha hai..."):
    try:
        raw = fetch_live_data()
        feat = build_features(raw).dropna(subset=FEATURES)
    except Exception as e:
        st.error(f"Data fetch mein masla: {e}")
        st.stop()

if feat.empty:
    st.error("Kaafi recent data nahi mila features banane ke liye.")
    st.stop()

latest = feat.iloc[[-1]]
current_pm = float(latest["pm2_5"].iloc[0])
current_aqi = calculate_aqi(current_pm)
last_time = latest["datetime"].iloc[0]

# Predict next 3 days
pm_preds = np.maximum(model.predict(latest[FEATURES])[0], 0)
days = ["Tomorrow", "Day 2", "Day 3"]
hours = ["+24h", "+48h", "+72h"]

# ---- Current ----
st.markdown("### Current")
c1, c2 = st.columns([1, 2])
with c1:
    st.markdown(f"""
    <div class="card" style="background:{aqi_color(current_aqi)};">
        <div class="day">RIGHT NOW</div>
        <div class="aqi">{current_aqi}</div>
        <div class="cat">{aqi_category(current_aqi)}</div>
        <div class="pm">PM2.5: {current_pm:.1f} µg/m³</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.write("")
    st.metric("Last updated", last_time.strftime("%d %b %Y, %H:%M"))
    st.caption(f"Model: {model_type} · {len(FEATURES)} features")

# ---- Forecast cards ----
st.markdown("### 3-Day Forecast")
cols = st.columns(3)
for i, col in enumerate(cols):
    pm = pm_preds[i]
    aqi = calculate_aqi(pm)
    with col:
        st.markdown(f"""
        <div class="card" style="background:{aqi_color(aqi)};">
            <div class="day">{days[i]} ({hours[i]})</div>
            <div class="aqi">{aqi}</div>
            <div class="cat">{aqi_category(aqi)}</div>
            <div class="pm">PM2.5: {pm:.1f} µg/m³</div>
        </div>""", unsafe_allow_html=True)

# ---- Alerts ----
st.markdown("### Alerts")
worst = max(calculate_aqi(p) for p in pm_preds)
if worst > 200:
    st.error(f"🚨 Khatarnak AQI ({worst}) aane wala hai — bahar na niklein, mask zaroori.")
elif worst > 150:
    st.warning(f"⚠️ AQI kharab ({worst}) ho sakta hai — bahar kam rahein.")
elif worst > 100:
    st.info(f"🟡 AQI {worst} tak ja sakta hai — sensitive log (bachay, buzurg) ehtiyaat karein.")
else:
    st.success(f"✅ Agle 3 din AQI theek rehne ki ummeed ({worst} tak).")

# ---- Trend chart ----
st.markdown("### Recent Trend + Forecast")
recent = feat.tail(120).copy()
recent["AQI"] = recent["pm2_5"].apply(calculate_aqi)
hist = recent[["datetime", "AQI"]].rename(columns={"datetime": "time"}).set_index("time")

future_times = pd.date_range(last_time, periods=4, freq="24h")[1:]
future = pd.DataFrame({"Forecast AQI": [calculate_aqi(p) for p in pm_preds]},
                      index=future_times)
chart_df = pd.concat([hist.rename(columns={"AQI": "Recent AQI"}), future], axis=1)
st.line_chart(chart_df, height=320)

st.markdown(f"""<div class="footnote">
Data: Open-Meteo (free). Predictions are model estimates and may differ from actual readings,
especially during sudden pollution spikes. AQI computed from PM2.5 (US EPA scale).
</div>""", unsafe_allow_html=True)
