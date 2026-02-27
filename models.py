import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import datetime   # ✅ Added for dynamic file naming

# -------------------------------
# 4. Train & Evaluate Models
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
# 5. Plot Results
# -------------------------------
def plot_forecast(test_df, predictions):
    import matplotlib.pyplot as plt
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