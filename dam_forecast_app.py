#XGBOOS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import skew, kurtosis, ks_2samp
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from math import sqrt

# Load dataset
file_path = "https://github.com/Karthikgv07/Mine/blob/main/DAM_Market%20Snapshot.csv"
df = pd.read_csv(file_path)

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-zA-Z0-9_]", "", regex=True)

# Convert date and drop invalids
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

# Set datetime index early
df.set_index('date', inplace=True)

# Convert numeric columns
numeric_cols = ["purchasebidmwh", "sellbidmwh", "mcvmwh", "finalscheduledvolumemwh", "weightedmcprsmwh", "mcprsmwh"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Target variable
target_variable = "weightedmcprsmwh"

# Time Series Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df[target_variable], label='MCP (Rs/MWh)', color='blue')
plt.title("Electricity Price Trends")
plt.xlabel("Date")
plt.ylabel("MCP (Rs/MWh)")
plt.legend()
plt.grid()
plt.show()

# Summary statistics
print("Summary Statistics:\n", df.describe())

# Skewness and Kurtosis
print(f"Skewness: {skew(df[target_variable].dropna()):.4f}")
print(f"Kurtosis: {kurtosis(df[target_variable].dropna()):.4f}")

# Seasonal Decomposition (ensure sufficient data for period=365)
if len(df) >= 365:
    decomposition = seasonal_decompose(df[target_variable].dropna(), model='additive', period=365)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    decomposition.trend.plot(ax=axes[0], title='Trend', color='blue')
    decomposition.seasonal.plot(ax=axes[1], title='Seasonality', color='green')
    decomposition.resid.plot(ax=axes[2], title='Residuals', color='red')
    plt.tight_layout()
    plt.show()

# ACF & PACF
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
plot_acf(df[target_variable].dropna(), ax=ax[0], lags=50)
plot_pacf(df[target_variable].dropna(), ax=ax[1], lags=50)
ax[0].set_title("ACF")
ax[1].set_title("PACF")
plt.tight_layout()
plt.show()

# ADF Test
def adf_test(series):
    result = adfuller(series.dropna())
    print(f"\nADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.6f}")
    print("=> Stationary" if result[1] < 0.05 else "=> Non-stationary")

adf_test(df[target_variable])

# KPSS Test
def kpss_test(series):
    result = kpss(series.dropna(), nlags="auto")
    print(f"\nKPSS Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.6f}")
    print("=> Non-stationary" if result[1] < 0.05 else "=> Stationary")

kpss_test(df[target_variable])

# KS Test
def ks_test(series):
    stat, p = ks_2samp(series.dropna(), np.random.normal(series.mean(), series.std(), len(series)))
    print(f"\nKS Statistic: {stat:.4f}")
    print(f"p-value: {p:.6f}")
    print("=> Non-normal distribution" if p < 0.05 else "=> Normal distribution")

ks_test(df[target_variable])

# Feature Engineering
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['day_of_week'] = df.index.dayofweek
df['linear_trend'] = np.arange(len(df))
df['quadratic_trend'] = df['linear_trend'] ** 2
df['log_mcp'] = np.log1p(df[target_variable])  # log(x + 1)
df['mcp_lag1'] = df[target_variable].shift(1)
df['rolling_mean'] = df[target_variable].rolling(window=7).mean()
df['ewma'] = df[target_variable].ewm(span=10).mean()

# Lag Feature Generation
def add_lag_features(df, target_col, lags=10):
    for lag in range(1, lags + 1):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df

df_lagged = add_lag_features(df.copy(), target_variable, 10)
df_lagged.dropna(inplace=True)

# Train-test split
train = df_lagged['2022-04-01':'2024-03-24']
test = df_lagged['2024-03-25':'2025-04-04']

X_train = train.drop(columns=[target_variable])
y_train = train[target_variable]
X_test = test.drop(columns=[target_variable])
y_test = test[target_variable]

# One-hot encoding if needed (safe step)
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Model training
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions and evaluation
test['XGBoost_Pred'] = xgb_model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, test['XGBoost_Pred']))
mape = mean_absolute_percentage_error(y_test, test['XGBoost_Pred'])

print(f"\nXGBoost RMSE: {rmse:.2f}")
print(f"XGBoost MAPE: {mape:.2%}")

# Forecast Plot
plt.figure(figsize=(12, 6))
plt.plot(train.index, y_train, label="Train", color='blue')
plt.plot(test.index, y_test, label="Actual", color='green')
plt.plot(test.index, test['XGBoost_Pred'], label="Predicted", linestyle='--', color='red')
plt.axvline(test.index.min(), linestyle='--', color='black')
plt.title("XGBoost Forecast of MCP (Rs/MWh)")
plt.xlabel("Date")
plt.ylabel("MCP (Rs/MWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta

st.set_page_config(layout="wide", page_title="DAM Market Forecast App")

st.title(" DAM Market Forecasting App (1-Month, 90% CI)")

# File uploader
uploaded_file = st.file_uploader("https://github.com/Karthikgv07/Mine/blob/main/DAM_Market%20Snapshot.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Let user choose the date and target columns
    st.sidebar.header("Configuration")
    date_col = st.sidebar.selectbox("Select Date Column", df.columns)
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    # Parse date column
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    df.set_index(date_col, inplace=True)

    # Resample to daily if needed (optional)
    df = df.resample("D").mean().fillna(method="ffill")

    st.subheader(" Raw Time Series Data")
    st.line_chart(df[target_col])

    # Fit SARIMAX model (can be replaced with ARIMA or Prophet later)
    model = SARIMAX(df[target_col], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    results = model.fit(disp=False)

    # Forecast 30 days
    future_steps = 30
    forecast = results.get_forecast(steps=future_steps)
    forecast_df = forecast.summary_frame(alpha=0.10)  # 90% CI

    # Build forecast index
    last_date = df.index[-1]
    forecast_index = pd.date_range(start=last_date + timedelta(days=1), periods=future_steps, freq="D")
    forecast_df.index = forecast_index

    # Plot
    st.subheader("🔮 Forecast for Next 1 Month (90% Confidence Interval)")
    fig, ax = plt.subplots(figsize=(12, 5))
    df[target_col].plot(ax=ax, label="Observed", color="blue")
    forecast_df["mean"].plot(ax=ax, label="Forecast", color="orange")
    ax.fill_between(forecast_df.index, forecast_df["mean_ci_lower"], forecast_df["mean_ci_upper"],
                    color='orange', alpha=0.3, label="90% CI")
    ax.legend()
    ax.set_title("Forecast with 90% Confidence Interval")
    st.pyplot(fig)

    # Optional: Show forecast table
    st.subheader(" Forecast Data Table")
    st.dataframe(forecast_df[["mean", "mean_ci_lower", "mean_ci_upper"]].round(2))
else:
    st.info(" Please upload a CSV file to get started.")



