from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_clean_data():

    dataset = fetch_ucirepo(id=235)

    X = dataset.data.features
    y = dataset.data.targets

    df = pd.concat([X, y], axis=1)

    # Create Datetime column
    df['Datetime'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'],
        dayfirst=True,
        errors='coerce'
    )

    df.set_index('Datetime', inplace=True)

    df.drop(columns=['Date', 'Time'], inplace=True)

    df.replace('?', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
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