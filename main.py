from analysis import load_and_clean_data, get_daily_series, perform_eda
from models import train_models, plot_forecast

def main():
    print("Loading and Cleaning Data...")
    df = load_and_clean_data("data/household_power_consumption.txt")

    daily_power = get_daily_series(df)

    perform_eda(daily_power)

    baseline_mae, lr_mae, test_df, lr_pred = train_models(daily_power)

    print("\nModel Results")
    print("Baseline MAE:", baseline_mae)
    print("Linear Regression MAE:", lr_mae)

    plot_forecast(test_df, lr_pred)

if __name__ == "__main__":
    main()