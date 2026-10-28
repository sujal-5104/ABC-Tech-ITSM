
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"


def inspect_monthly_volume():
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

    # Count tickets for every calendar day
    df["Open_Date"] = df["Open_Time_Parsed"].dt.normalize()

    daily_counts = df.groupby("Open_Date").size()

    # Include every day between the first and last dates
    all_days = pd.date_range(
        daily_counts.index.min(),
        daily_counts.index.max(),
        freq="D"
    )

    daily_counts = daily_counts.reindex(
        all_days,
        fill_value=0
    )

    daily_counts.index.name = "Open_Date"

    # Summarize daily activity by calendar month
    monthly = daily_counts.resample("MS").agg(
        total_tickets="sum",
        active_days=lambda values: int((values > 0).sum()),
        zero_ticket_days=lambda values: int((values == 0).sum()),
        calendar_days="count",
        average_daily_tickets="mean"
    )

    monthly["average_daily_tickets"] = (
        monthly["average_daily_tickets"].round(2)
    )

    print("=" * 75)
    print("MONTHLY TICKET VOLUME AND ZERO-DAY INVESTIGATION")
    print("=" * 75)

    print(monthly.to_string())

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)

    print("Total tickets:", int(monthly["total_tickets"].sum()))
    print("Total active days:", int(monthly["active_days"].sum()))
    print("Total zero-ticket days:", int(monthly["zero_ticket_days"].sum()))

    print("\nMonthly summary saved to the console only.")
    print("Original CSV was not modified.")


if __name__ == "__main__":
    inspect_monthly_volume()

