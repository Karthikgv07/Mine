# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# --- Streamlit Config ---
st.set_page_config(page_title="MCP Forecast with SARIMA", layout="wide")
st.title("⚡ Market Clearing Price Forecast using SARIMA")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your CSV file (e.g., DAM_Market Snapshot.csv)", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'mcprsmwh'])
    df.set_index('date', inplace=True)
    df = df.asfreq('D')
    df['mcprsmwh'] = pd.to_numeric(df['mcprsmwh'], errors='coerce')
    df = df.fillna(method='ffill')
    return df[['mcprsmwh']]

@st.cache_resource
def train_sarima_model(series):
    model_auto = auto_arima(series, seasonal=True, m=7, trace=False, error_action='ignore', suppress_warnings=True)
    order = model_auto.order
    seasonal_order = model_auto.seasonal_order
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    return results, order, seasonal_order

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.subheader("📈 Historical MCP Data")
    st.line_chart(df)

    st.subheader("🔧 Training SARIMA Model")
    with st.spinner("Model training in progress..."):
        model_fit, order, seasonal_order = train_sarima_model(df)

    st.success(f"Model trained! Order: {order}, Seasonal Order: {seasonal_order}")

    # --- Forecast ---
    st.subheader("📊 Forecast for Next 30 Days")
    forecast_steps = 30
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    forecast_index = forecast_mean.index

    # --- Plot Forecast ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df[-100:], label="Historical MCP", color='blue')
    ax.plot(forecast_mean, label="Forecasted MCP", color='red')
    ax.fill_between(forecast_index,
                    conf_int.iloc[:, 0],
                    conf_int.iloc[:, 1],
                    color='pink', alpha=0.3, label="Confidence Interval")
    ax.set_xlabel("Date")
    ax.set_ylabel("MCP (Rs/MWh)")
    ax.set_title("SARIMA Forecast - Market Clearing Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- Download Forecast ---
    st.subheader("📥 Download Forecasted Data")
    forecast_df = pd.DataFrame({
        'Date': forecast_index,
        'Forecasted_MCP': forecast_mean.values,
        'Lower Bound': conf_int.iloc[:, 0].values,
        'Upper Bound': conf_int.iloc[:, 1].values
    })
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "SARIMA_MCP_Forecast.csv", "text/csv")
else:
    st.info("📤 Please upload a CSV file to start.")
