import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import datetime
from ucimlrepo import fetch_ucirepo

# -------------------------------
# 1. Load & Clean Data
# -------------------------------
def load_and_clean_data():
    # Fetch dataset from UCI (Household Power Consumption)
    dataset = fetch_ucirepo(id=235)

    X = dataset.data.features
    y = dataset.data.targets

    # Combine features and target
    df = pd.concat([X, y], axis=1)

    # Create Datetime column
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
    df.set_index('Datetime', inplace=True)

    # Drop old Date and Time columns
    df.drop(columns=['Date', 'Time'], inplace=True)

    # Replace missing and convert to numeric
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df.dropna(inplace=True)

    return df

# -------------------------------
# 2. Train & Evaluate Models
# -------------------------------
def train_models(daily):
    train_size = int(len(daily) * 0.8)

    train = daily[:train_size]
    test = daily[train_size:]

    # ---- Baseline (Moving Average) ----
    rolling_avg = train.rolling(7).mean().dropna()
    baseline_pred = np.full(len(test), rolling_avg.iloc[-1])
    baseline_mae = mean_absolute_error(test, baseline_pred)

    # ---- Linear Regression ----
    df_daily = daily.reset_index()
    df_daily['Day'] = range(len(df_daily))

    train_df = df_daily[:train_size]
    test_df = df_daily[train_size:]

    model = LinearRegression()
    model.fit(train_df[['Day']], train_df['Global_active_power'])

    lr_pred = model.predict(test_df[['Day']])
    lr_mae = mean_absolute_error(test_df['Global_active_power'], lr_pred)

    return baseline_mae, lr_mae, test_df, lr_pred

# -------------------------------
# 3. Plot Results
# -------------------------------
def plot_forecast(test_df, predictions):
    plt.figure(figsize=(12,6))
    plt.plot(test_df['Datetime'], test_df['Global_active_power'], label="Actual")
    plt.plot(test_df['Datetime'], predictions, label="Linear Regression")
    plt.legend()
    plt.title("Energy Forecast")
    plt.tight_layout()

    # ✅ Save with today’s date (e.g., forecast_2026-02-27.png)
    filename = f"forecast_{datetime.date.today()}.png"
    plt.savefig(filename)
    plt.show()

# -------------------------------
# 4. Full Pipeline
# -------------------------------
if __name__ == "__main__":
    df = load_and_clean_data()

    # Aggregate daily consumption
    daily = df['Global_active_power'].resample('D').mean()

    baseline_mae, lr_mae, test_df, lr_pred = train_models(daily)

    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Linear Regression MAE: {lr_mae:.4f}")

    plot_forecast(test_df, lr_pred)