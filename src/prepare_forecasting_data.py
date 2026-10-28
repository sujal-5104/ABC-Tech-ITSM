
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "daily_forecasting_data.csv"


def prepare_forecasting_data():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    df["Open_Time_Parsed"] = pd.to_datetime(
        df["Open_Time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    df = df.dropna(subset=["Open_Time_Parsed"]).copy()

    # Keep the high-volume period for the first forecasting experiment
    start_date = pd.Timestamp("2013-10-01")
    end_date = pd.Timestamp("2014-03-31")

    df = df[
        (df["Open_Time_Parsed"] >= start_date)
        & (df["Open_Time_Parsed"] < end_date + pd.Timedelta(days=1))
    ].copy()

    df["Open_Date"] = df["Open_Time_Parsed"].dt.normalize()

    daily = (
        df.groupby("Open_Date")
        .size()
        .rename("Ticket_Count")
        .to_frame()
    )

    # Include calendar dates with zero tickets
    all_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    daily = daily.reindex(all_dates, fill_value=0)
    daily.index.name = "Open_Date"
    daily = daily.reset_index()

    # Calendar features for analysis and later modeling
    daily["Day_Of_Week"] = daily["Open_Date"].dt.day_of_week
    daily["Is_Weekend"] = (
        daily["Day_Of_Week"] >= 5
    ).astype(int)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("FORECASTING DATA PREPARATION COMPLETED")
    print("=" * 60)
    print(f"Date range: {daily['Open_Date'].min().date()} to "
          f"{daily['Open_Date'].max().date()}")
    print(f"Calendar days: {len(daily)}")
    print(f"Total tickets: {daily['Ticket_Count'].sum():,}")
    print(f"Zero-ticket days: {(daily['Ticket_Count'] == 0).sum()}")
    print(f"Average daily tickets: {daily['Ticket_Count'].mean():.2f}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("\nOriginal dataset was not modified.")


if __name__ == "__main__":
    prepare_forecasting_data()
