import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load and Clean Dataset
# -------------------------------
def load_and_clean_data(path):
    df = pd.read_csv(
        path,
        sep=';',
        parse_dates={'Datetime': ['Date', 'Time']},
        dayfirst=True,
        low_memory=False
    )
    df.set_index('Datetime', inplace=True)
    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric)
    df.dropna(inplace=True)
    return df

# -------------------------------
# 2. Prepare Daily Series
# -------------------------------
def get_daily_series(df):
    daily = df['Global_active_power'].resample('D').mean()
    return daily.dropna()

# -------------------------------
# 3. Exploratory Analysis
# -------------------------------
def perform_eda(daily):
    print("\nPeak Demand Analysis")
    print("Peak Day:", daily.idxmax())
    print("Peak Value:", daily.max())

    plt.figure(figsize=(12, 5))
    daily.plot()
    plt.title("Daily Average Power")
    plt.tight_layout()
    plt.show()