import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis import load_and_clean_data, get_daily_series
from models import train_models
from datetime import timedelta
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Energy Predictor", layout="wide")

st.title("⚡ Smart Energy Forecasting Dashboard")
st.markdown("AI-based Electricity Consumption Prediction System")

# -----------------------------
# Load Data
# -----------------------------
df = load_and_clean_data()   # ✅ Updated: no path needed
daily_power = get_daily_series(df)

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Dashboard Controls")
forecast_days = st.sidebar.slider("Select Future Forecast Days", 7, 60, 30)

# -----------------------------
# KPI Section
# -----------------------------
col1, col2, col3 = st.columns(3)

peak_day = daily_power.idxmax()
peak_val = daily_power.max()
avg_power = daily_power.mean()

col1.metric("Average Daily Power (kW)", round(avg_power, 3))
col2.metric("Peak Demand (kW)", round(peak_val, 3))
col3.metric("Peak Demand Day", str(peak_day.date()))

st.divider()

# -----------------------------
# Trend Chart
# -----------------------------
st.subheader("📈 Daily Energy Consumption Trend")
st.line_chart(daily_power)

# -----------------------------
# Train Models
# -----------------------------
baseline_mae, lr_mae, test_df, lr_pred = train_models(daily_power)

st.subheader("📊 Model Performance")

colA, colB = st.columns(2)
colA.metric("Baseline MAE", round(baseline_mae, 4))
colB.metric("Linear Regression MAE", round(lr_mae, 4))

if lr_mae < baseline_mae:
    st.success("✅ Linear Regression improves prediction accuracy!")
else:
    st.warning("⚠ Model needs improvement.")

st.divider()

# -----------------------------
# Forecast vs Actual
# -----------------------------
st.subheader("🔮 Forecast vs Actual Comparison")

forecast_df = pd.DataFrame({
    "Date": test_df["Datetime"],
    "Actual": test_df["Global_active_power"],
    "Predicted": lr_pred
}).set_index("Date")

st.line_chart(forecast_df)

st.divider()

# -----------------------------
# Future Forecast
# -----------------------------
st.subheader(f"🚀 Future {forecast_days}-Day Forecast")

last_day_number = len(daily_power)
future_days = np.arange(last_day_number, last_day_number + forecast_days)

# Re-train on full data
df_daily = daily_power.reset_index()
df_daily['Day'] = range(len(df_daily))

model = LinearRegression()
model.fit(df_daily[['Day']], df_daily['Global_active_power'])

future_pred = model.predict(future_days.reshape(-1,1))
future_dates = [daily_power.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]

future_forecast_df = pd.DataFrame({
    "Future_Predicted_Power": future_pred
}, index=future_dates)

st.line_chart(future_forecast_df)

st.divider()

# -----------------------------
# Download Section
# -----------------------------
st.subheader("📥 Download Forecast Data")

download_df = pd.concat([forecast_df, future_forecast_df])
csv = download_df.to_csv().encode('utf-8')

st.download_button(
    label="Download Forecast Results as CSV",
    data=csv,
    file_name="energy_forecast_results.csv",
    mime='text/csv'
)

st.info("Project by Mani – AI Energy Forecasting System 🚀")