# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import skew, kurtosis, ks_2samp
from pmdarima import auto_arima
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools


st.set_page_config(page_title="MCP Forecasting App", layout="wide")

st.title("📈 Market Clearing Price Forecasting App")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your DAM Market Snapshot CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Data Cleaning ---
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    numeric_cols = ["purchasebidmwh", "sellbidmwh", "mcvmwh", 
                    "finalscheduledvolumemwh", "weightedmcprsmwh", "mcprsmwh"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['mcprsmwh'])

    df = df.groupby('date').mean()
    df = df.asfreq('D')
    df = df.ffill()

    # --- Summary Statistics ---
    st.subheader("📊 Summary Statistics")
    st.write(df.describe())
    st.write(f"**Skewness:** {skew(df['mcprsmwh'].dropna()):.2f}")
    st.write(f"**Kurtosis:** {kurtosis(df['mcprsmwh'].dropna()):.2f}")

    # --- Decomposition ---
    st.subheader("📉 Time Series Decomposition")
    fig, ax = plt.subplots(figsize=(10, 6))
    seasonal_decompose(df['mcprsmwh'], model='additive', period=365).plot()
    st.pyplot(plt)

    # --- ACF & PACF ---
    st.subheader("🔁 ACF and PACF")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    plot_acf(df['mcprsmwh'].dropna(), lags=50, ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    plot_pacf(df['mcprsmwh'].dropna(), lags=50, ax=ax2)
    st.pyplot(fig2)

    # --- Stationarity Tests ---
    def adf_test(series):
        result = adfuller(series.dropna())
        return result[1] < 0.05, result

    def kpss_test(series):
        result = kpss(series.dropna(), nlags="auto")
        return result[1] > 0.05, result

    def ks_test(series):
        statistic, p_value = ks_2samp(series.dropna(), 
                                      np.random.normal(series.mean(), series.std(), len(series)))
        return p_value > 0.05, (statistic, p_value)

    st.subheader("🧪 Statistical Tests")
    adf_result = adf_test(df['mcprsmwh'])
    kpss_result = kpss_test(df['mcprsmwh'])
    ks_result = ks_test(df['mcprsmwh'])

    st.write(f"**ADF Test (Stationary?):** {'Yes' if adf_result[0] else 'No'} (p-value: {adf_result[1][1]:.5f})")
    st.write(f"**KPSS Test (Stationary?):** {'Yes' if kpss_result[0] else 'No'} (p-value: {kpss_result[1][1]:.5f})")
    st.write(f"**KS Test (Normality?):** {'Yes' if ks_result[0] else 'No'} (p-value: {ks_result[1][1]:.5f})")

    # --- Auto ARIMA ---
    st.subheader("🤖 Auto ARIMA Model Selection")
    auto_model = auto_arima(df['mcprsmwh'], seasonal=True, m=7, trace=False)
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order
    st.write(f"**Selected Order:** {order}, **Seasonal Order:** {seasonal_order}")

    # --- Train/Test Split ---
    train_size = int(0.8 * len(df))
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # --- SARIMAX ---
    model = SARIMAX(train['mcprsmwh'], order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    forecast = model_fit.get_forecast(steps=len(test))
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # --- Forecast Accuracy ---
    rmse = np.sqrt(mean_squared_error(test['mcprsmwh'], forecast_mean))
    mae = mean_absolute_error(test['mcprsmwh'], forecast_mean)
    mape = np.mean(np.abs((test['mcprsmwh'] - forecast_mean) / test['mcprsmwh'])) * 100

    st.subheader("📏 Forecast Accuracy")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")

    # --- Final Forecast ---
    final_model = SARIMAX(df['mcprsmwh'], order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False)
    final_model_fit = final_model.fit(disp=False)
    future_steps = 30
    final_forecast = final_model_fit.get_forecast(steps=future_steps)
    forecast_mean = final_forecast.predicted_mean
    conf_int = final_forecast.conf_int()
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_MCP': forecast_mean.values,
        'Lower Bound': conf_int.iloc[:, 0].values,
        'Upper Bound': conf_int.iloc[:, 1].values
    })

    st.subheader("🔮 Forecast for Next 30 Days")
    st.dataframe(forecast_df)

    # --- Plot ---
    st.subheader("📈 Forecast Plot")
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(df['mcprsmwh'], label='Historical MCP')
    plt.plot(forecast_df['Date'], forecast_df['Forecasted_MCP'], color='red', label='Forecasted MCP')
    plt.fill_between(forecast_df['Date'], forecast_df['Lower Bound'], forecast_df['Upper Bound'], 
                     color='pink', alpha=0.3, label='Confidence Interval')
    plt.xlabel("Date")
    plt.ylabel("MCP (Rs/MWh)")
    plt.title("MCP Forecast - Next 30 Days")
    plt.legend()
    plt.grid()
    st.pyplot(fig)
