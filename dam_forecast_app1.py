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
import io

# --- Page Config ---
st.set_page_config(page_title="MCP Forecast with SARIMA", layout="wide")
st.title(" Market Clearing Price (MCP) Forecast with SARIMA")


# --- Upload File ---
uploaded_file = st.file_uploader(" Upload DAM_Market Snapshot CSV", type=["csv"])

if uploaded_file:
    # --- Read and Clean Data ---
    df = pd.read_csv(uploaded_file)
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

    st.subheader(" Summary Statistics")
    st.dataframe(df.describe())
    st.write(f"**Skewness:** {skew(df['mcprsmwh'].dropna()):.2f}")
    st.write(f"**Kurtosis:** {kurtosis(df['mcprsmwh'].dropna()):.2f}")

    # --- Decomposition ---
    st.subheader(" Seasonal Decomposition")
    fig1 = seasonal_decompose(df['mcprsmwh'], model='additive', period=365).plot()
    st.pyplot(fig1)

    # --- ACF & PACF ---
    st.subheader(" ACF & PACF Plots")
    fig2, ax = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(df['mcprsmwh'].dropna(), lags=50, ax=ax[0])
    plot_pacf(df['mcprsmwh'].dropna(), lags=50, ax=ax[1])
    plt.tight_layout()
    st.pyplot(fig2)

    # --- Stationarity Tests ---
    st.subheader(" Stationarity & Normality Tests")

    def adf_test(series):
        result = adfuller(series.dropna())
        return result[0], result[1]

    def kpss_test(series):
        result = kpss(series.dropna(), nlags="auto")
        return result[0], result[1]

    def ks_test(series):
        stat, pval = ks_2samp(series.dropna(), 
                              np.random.normal(series.mean(), series.std(), len(series)))
        return stat, pval

    adf_stat, adf_p = adf_test(df['mcprsmwh'])
    kpss_stat, kpss_p = kpss_test(df['mcprsmwh'])
    ks_stat, ks_p = ks_test(df['mcprsmwh'])

    st.write(f"**ADF Test**: Statistic = {adf_stat:.4f}, p-value = {adf_p:.6f} → {'Stationary' if adf_p < 0.05 else 'Non-Stationary'}")
    st.write(f"**KPSS Test**: Statistic = {kpss_stat:.4f}, p-value = {kpss_p:.6f} → {'Non-Stationary' if kpss_p < 0.05 else 'Stationary'}")
    st.write(f"**KS Test**: Statistic = {ks_stat:.4f}, p-value = {ks_p:.6f} → {'Non-normal' if ks_p < 0.05 else 'Normal'}")

    # --- Train SARIMA Model ---
    st.subheader(" SARIMA Model Training")
    with st.spinner("Running Auto ARIMA..."):
        auto_model = auto_arima(df['mcprsmwh'], seasonal=True, m=7, 
                                trace=False, error_action='ignore', suppress_warnings=True)
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order
    st.success(f"Best Model: SARIMA{order}x{seasonal_order}")

    model = SARIMAX(df['mcprsmwh'], order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    st.success(" Model Training Complete")

    # --- Forecast ---
    st.subheader(" MCP Forecast - Next 30 Days")
    forecast_steps = 30
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps + 1)]

    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted_MCP': forecast_mean.values,
        'Lower_Bound': conf_int.iloc[:, 0].values,
        'Upper_Bound': conf_int.iloc[:, 1].values
    })

    st.dataframe(forecast_df)

    # --- Plot Forecast ---
    st.subheader(" Forecast Visualization")
    fig3, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['mcprsmwh'], label="Historical MCP", color='blue')
    ax.plot(forecast_df['Date'], forecast_df['Forecasted_MCP'], label="Forecasted MCP", color='red')
    ax.fill_between(forecast_df['Date'], forecast_df['Lower_Bound'], forecast_df['Upper_Bound'], 
                    color='pink', alpha=0.3, label="Confidence Interval")
    ax.set_xlabel("Date")
    ax.set_ylabel("MCP (Rs/MWh)")
    ax.set_title("Market Clearing Price Forecast (Next 30 Days)")
    ax.legend()
    ax.grid()
    st.pyplot(fig3)

    # --- Download Button ---
    st.subheader(" Download Forecast CSV")
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(" Download CSV", csv, file_name="MCP_Forecast.csv", mime="text/csv")

else:
    st.info(" Please upload a CSV file to begin.")
