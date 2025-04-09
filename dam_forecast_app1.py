# --- app.py ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# --- Streamlit Config ---
st.set_page_config(page_title="MCP Forecast with SARIMA", layout="wide")
st.title("Market Clearing Price Forecast using SARIMA")

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload DAM_Market Snapshot CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Clean & Prepare ---
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'mcprsmwh'])

    # --- Clean & Prepare ---
df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date', 'mcprsmwh'])

# Handle duplicates: average MCP per day
df = df.groupby('date', as_index=False)['mcprsmwh'].mean()

# Reindex to daily frequency
df.set_index('date', inplace=True)
df = df.asfreq('D')

df['mcprsmwh'] = pd.to_numeric(df['mcprsmwh'], errors='coerce')
df = df.ffill()

    df['mcprsmwh'] = pd.to_numeric(df['mcprsmwh'], errors='coerce')
    df = df.ffill()

    data = df[['mcprsmwh']]

    st.subheader("📊 Raw MCP Time Series")
    st.line_chart(data)

    # --- SARIMA Modeling ---
    st.subheader("🔁 Fitting SARIMA Model")

    with st.spinner("Training SARIMA model..."):
        # Manually set model order (adjustable based on experience)
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 7)  # Weekly seasonality

        model = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)

    st.success(f"Model trained with order={order}, seasonal_order={seasonal_order}")

    # --- Forecast ---
    forecast_steps = 30
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    forecast_index = forecast_mean.index

    # --- Plot ---
    st.subheader("📈 MCP Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data[-90:], label="Historical MCP", color='blue')
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
        'Lower_Bound': conf_int.iloc[:, 0].values,
        'Upper_Bound': conf_int.iloc[:, 1].values
    })

    st.dataframe(forecast_df)

    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast as CSV", csv, "SARIMA_MCP_Forecast.csv", "text/csv")

else:
    st.info("📁 Please upload a CSV file to begin.")
