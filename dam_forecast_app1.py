# --- Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import skew, kurtosis, ks_2samp
import seaborn as sns

# --- Load Dataset ---
file_path = "DAM_Market Snapshot.csv"
df = pd.read_csv(file_path)

# --- Clean and Prepare Data ---
df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

numeric_cols = ["purchasebidmwh", "sellbidmwh", "mcvmwh", 
                "finalscheduledvolumemwh", "weightedmcprsmwh", "mcprsmwh"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df = df.dropna(subset=['mcprsmwh'])

# --- Set Date as Index and handle duplicates ---
df['date'] = pd.to_datetime(df['date'])  # ensure it's datetime
df = df.groupby('date').mean()  # aggregate to avoid duplicates
df = df.asfreq('D')  # now safe to reindex by day
df = df.ffill()  # optional: fill missing days using forward fill


# --- Summary Stats ---
print("Summary Stats:\n", df.describe())
print(f"Skewness: {skew(df['mcprsmwh'].dropna()):.2f}")
print(f"Kurtosis: {kurtosis(df['mcprsmwh'].dropna()):.2f}")

# --- Decomposition ---
seasonal_decompose(df['mcprsmwh'], model='additive', period=365).plot()
plt.tight_layout()
plt.show()

# --- ACF & PACF ---
plot_acf(df['mcprsmwh'].dropna(), lags=50)
plot_pacf(df['mcprsmwh'].dropna(), lags=50)
plt.show()

# --- Stationarity Tests ---
def adf_test(series):
    result = adfuller(series.dropna())
    print(f"ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.6f}")
    print("Stationary" if result[1] < 0.05 else "Non-stationary")

def kpss_test(series):
    result = kpss(series.dropna(), nlags="auto")
    print(f"KPSS Statistic: {result[0]}, p-value: {result[1]}")
    print("Non-stationary" if result[1] < 0.05 else "Stationary")

def ks_test(series):
    statistic, p_value = ks_2samp(series.dropna(), 
                                  np.random.normal(series.mean(), series.std(), len(series)))
    print(f"KS Statistic: {statistic}, p-value: {p_value}")
    print("Non-normal distribution" if p_value < 0.05 else "Normal distribution")

print("\nADF Test:"); adf_test(df['mcprsmwh'])
print("\nKPSS Test:"); kpss_test(df['mcprsmwh'])
print("\nKS Test:"); ks_test(df['mcprsmwh'])

# --- SARIMA Model Training ---
model = SARIMAX(df['mcprsmwh'], order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)

# --- Forecast Next 30 Days ---
from datetime import timedelta

forecast_steps = 30
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_steps + 1)]

forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean

# --- Create Forecast DataFrame ---
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted_MCP': forecast_mean.values
})

# --- Display ---
print("\nForecast for next 30 days:")
print(forecast_df)

# --- Optional: Plot Forecast ---
plt.figure(figsize=(12,6))
plt.plot(df['mcprsmwh'], label="Historical MCP")
plt.plot(forecast_df['Date'], forecast_df['Forecasted_MCP'], label="Forecasted MCP", color='red')
plt.xlabel("Date")
plt.ylabel("MCP (Rs/MWh)")
plt.title("Market Clearing Price Forecast (Next 30 Days)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



# --- app.py ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

# --- Streamlit Config ---
st.set_page_config(page_title="MCP Forecast with SARIMA", layout="wide")
st.title(" Market Clearing Price Forecast using SARIMA")


# File uploader
uploaded_file = st.file_uploader("DAM_Market Snapshot.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Clean and Prepare ---
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'mcprsmwh'])

    df.set_index('date', inplace=True)
    df = df.asfreq('D')  # daily frequency
    data = df[['mcprsmwh']].copy()

    st.subheader(" Raw MCP Time Series")
    st.line_chart(data)

    # --- SARIMA Modeling ---
    st.subheader(" Fitting SARIMA Model... (Auto ARIMA)")

    with st.spinner("Training SARIMA model..."):
        auto_model = auto_arima(data, seasonal=True, m=7, trace=False, error_action='ignore', suppress_warnings=True)
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order

        model = SARIMAX(data, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)

    st.success(f" Model trained! Order: {order}, Seasonal Order: {seasonal_order}")

    # --- Forecast Next 30 Days ---
    st.subheader(" Forecasting MCP for Next 30 Days")
    forecast_steps = 30
    forecast = results.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()
    forecast_index = forecast_mean.index

    # --- Plot ---
    st.subheader(" MCP Forecast Plot")

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
    st.download_button("Download Forecast as CSV", csv, "SARIMA_MCP_Forecast.csv", "text/csv")

else:
    st.info(" Please upload a CSV file to begin.")
