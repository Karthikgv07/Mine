# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from datetime import timedelta

# --- Streamlit Config ---
st.set_page_config(page_title="MCP Forecast with SARIMA", layout="wide")
st.title("📈 Market Clearing Price Forecast using SARIMA")

# --- Upload CSV File ---
uploaded_file = st.file_uploader("Upload the DAM_Market Snapshot CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Using sample data.")
    df = pd.read_csv("DAM_Market Snapshot.csv")

# --- Data Cleaning ---
df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date', 'mcprsmwh'])
df.set_index('date', inplace=True)
df = df.asfreq('D')  # daily frequency
df['mcprsmwh'] = pd.to_numeric(df['mcprsmwh'], errors='coerce')
df = df.ffill()  # Fill missing values forward

data = df[['mcprsmwh']]

# --- Show Raw Plot ---
st.subheader("📊 Raw MCP Time Series")
st.line_chart(data)

# --- Fit SARIMA ---
st.subheader("⚙️ Fitting SARIMA Model")
with st.spinner("Auto-tuning model. Please wait..."):
    auto_model = auto_arima(data, seasonal=True, m=7, trace=False, error_action='ignore', suppress_warnings=True)
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order

    model = SARIMAX(data, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

st.success(f"Model Fitted ✔️ | Order: {order}, Seasonal Order: {seasonal_order}")

# --- Forecast Next 30 Days ---
st.subheader("📅 Forecast for Next 30 Days")
forecast_steps = 30
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()
forecast_index = forecast_mean.index

# --- Forecast Plot ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data[-100:], label="Historical MCP", color='blue')
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

# --- Forecast Table + Download ---
forecast_df = pd.DataFrame({
    'Date': forecast_index,
    'Forecasted_MCP': forecast_mean.values,
    'Lower Bound': conf_int.iloc[:, 0].values,
    'Upper Bound': conf_int.iloc[:, 1].values
})
st.subheader("📥 Download Forecast Data")
st.dataframe(forecast_df)
csv = forecast_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Forecast CSV", csv, "MCP_Forecast.csv", "text/csv")
