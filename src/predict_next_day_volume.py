
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "daily_forecasting_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports"


def predict_next_day_volume():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE, parse_dates=["Open_Date"])
    df = df.sort_values("Open_Date").reset_index(drop=True)

    if len(df) < 7:
        print("At least seven days of historical data are required.")
        return

    # Use the latest available date and its ticket count
    latest_date = df["Open_Date"].max()
    latest_count = int(
        df.loc[df["Open_Date"] == latest_date, "Ticket_Count"].iloc[0]
    )

    # Predict the same weekday one week later.
    # This is a seven-day-ahead forecast, not a next-calendar-day forecast.
    forecast_date = latest_date + pd.Timedelta(days=7)
    forecast_count = latest_count

    result = pd.DataFrame([{
        "Forecast_Date": forecast_date,
        "Forecast_Tickets": forecast_count,
        "Method": "Previous-week baseline",
        "Based_On_Date": latest_date,
        "Based_On_Tickets": latest_count,
    }])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "next_week_volume_forecast.csv"
    result.to_csv(output_file, index=False)

    print("=" * 60)
    print("ITSM TICKET VOLUME FORECAST")
    print("=" * 60)
    print(f"Latest available date: {latest_date.date()}")
    print(f"Tickets on that date: {latest_count}")
    print(f"Forecast date: {forecast_date.date()}")
    print(f"Predicted tickets: {forecast_count}")
    print("Method: Previous-week baseline")
    print("\nImportant: This is a simple baseline forecast.")
    print("It assumes the same weekday's volume repeats after seven days.")
    print(f"\nForecast saved to: {output_file}")


if __name__ == "__main__":
    predict_next_day_volume()
