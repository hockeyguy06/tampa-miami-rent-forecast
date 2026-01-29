import streamlit as st
import pandas as pd
import joblib
from prophet.plot import plot_plotly
import os
from datetime import datetime
import json
import subprocess
import os

model_files = [
    f"rent_model_{file_suffix}.pkl",
    f"forecast_{file_suffix}.csv"
]

if not all(os.path.exists(f) for f in model_files):
    st.warning("Models not found on Cloud. Training now (may take 1-2 min)...")
    result = subprocess.run(["python3", "model_train.py"], capture_output=True, text=True)
    if result.returncode == 0:
        st.success("Training complete! Refresh the page.")
    else:
        st.error("Training failed:\n" + result.stderr)
    st.stop()

st.set_page_config(page_title="Tampa & Miami Rent Forecast", page_icon="🏠", layout="wide")

st.title("🏠 Tampa & Miami Rent Index Forecast")
st.markdown("Separate, high-accuracy predictions using Zillow ZORI + local unemployment data")

# === Last Updated Banner ===
model_path = '/Users/tylerbrenner/Documents/Projects/Housing/rent_model_Tampa_FL.pkl'  # any model works for timestamp
if os.path.exists(model_path):
    last_trained = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%B %d, %Y at %I:%M %p')
    st.markdown(
        f"<div style='background:#f5f7fa; padding:16px; border-radius:10px; text-align:center; font-size:1.1em; color:#1a1a2e; border:1px solid #d1d9e6;'>"
        f"🔄 <strong>Last model training:</strong> {last_trained}<br>"
        f"Data current through latest Zillow release • Next update automatically when new data drops"
        f"</div><br>",
        unsafe_allow_html=True
    )

metro = st.selectbox("Select Metro Area", ["Miami, FL", "Tampa, FL"], index=1)

metro_file_map = {"Miami, FL": "Miami_FL", "Tampa, FL": "Tampa_FL"}
suffix = metro_file_map[metro]

model_path = f'/Users/tylerbrenner/Documents/Projects/Housing/rent_model_{suffix}.pkl'
forecast_path = f'/Users/tylerbrenner/Documents/Projects/Housing/forecast_{suffix}.csv'

try:
    model = joblib.load(model_path)
    forecast = pd.read_csv(forecast_path)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
except Exception as e:
    st.error(f"Files not found — run `python3 model_train.py` first!")
    st.stop()

# Load CV metrics (auto-updated by model_train.py)
try:
    with open('/Users/tylerbrenner/Documents/Projects/Housing/cv_metrics.json') as f:
        cv_all = json.load(f)
    metrics = cv_all.get(metro, {"mape_1yr": None})
except:
    metrics = {"mape_1yr": None}

mape = metrics.get("mape_1yr", None)
if mape:
    mape_pct = mape * 100
    level = "Excellent" if mape_pct < 5 else "Good" if mape_pct < 8 else "Fair"
    color = "green" if mape_pct < 5 else "orange" if mape_pct < 8 else "red"
    st.markdown(f"### Model Confidence: <span style='color:{color};font-weight:bold;'>{level} ({mape_pct:.2f}% MAPE on 1-year forecasts)</span>", unsafe_allow_html=True)

# Disclaimer / Methodology
with st.expander("📊 Methodology & Disclaimer"):
    st.markdown("""
    - **Data**: Zillow Observed Rent Index (ZORI) – repeat-rent weighted, metro-level  
    - **Model**: Facebook Prophet with lag features, ZORF growth signal, and local unemployment rate  
    - **Accuracy**: Backtested 1-year MAPE shown above (lower = better)  
    - **Limitations**: Does not capture sudden supply shocks, policy changes, or short-term rental (Airbnb) effects  
    - **Purpose**: Decision support tool only — not financial advice  
    """)

# Rest of app (chart + table + download) stays the same as before
months = st.slider("Forecast months ahead", 3, 36, 12)

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader(f"{metro} Forecast")
    fig = plot_plotly(model, forecast)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(f"Next {months} Months")
    disp = forecast.tail(months)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].round(0)
    disp['ds'] = disp['ds'].dt.strftime('%Y-%m')
    disp.columns = ['Month', 'Predicted Rent ($)', 'Lower', 'Upper']
    st.dataframe(disp.style.format({'Predicted Rent ($)': '${:,.0f}', 'Lower': '${:,.0f}', 'Upper': '${:,.0f}'}))

st.download_button("📥 Download Full Forecast", forecast.to_csv(index=False).encode(), f"{metro.replace(', ', '_')}_forecast.csv")