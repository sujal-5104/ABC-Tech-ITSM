
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def analyze_daily_volume():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", low_memory=False)

    df["Open_Time_Parsed"] = pd.to_datetime(
        df["Open_Time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    df = df.dropna(subset=["Open_Time_Parsed"]).copy()

    # Count tickets by calendar day
    df["Open_Date"] = df["Open_Time_Parsed"].dt.normalize()

    daily_volume = (
        df.groupby("Open_Date")
        .size()
        .rename("Ticket_Count")
        .sort_index()
    )

    # Include dates with zero tickets
    full_date_range = pd.date_range(
        daily_volume.index.min(),
        daily_volume.index.max(),
        freq="D"
    )

    daily_volume = daily_volume.reindex(
        full_date_range,
        fill_value=0
    )
    daily_volume.index.name = "Open_Date"

    print("=" * 65)
    print("DAILY ITSM TICKET VOLUME ANALYSIS")
    print("=" * 65)

    print(f"First date: {daily_volume.index.min().date()}")
    print(f"Last date: {daily_volume.index.max().date()}")
    print(f"Number of calendar days: {len(daily_volume):,}")

    print("\nDaily volume summary:")
    print(daily_volume.describe().round(2).to_string())

    print("\nHighest-volume days:")
    print(
        daily_volume.sort_values(ascending=False)
        .head(10)
        .to_string()
    )

    print("\nDays with zero tickets:", int((daily_volume == 0).sum()))

    # Seven-day rolling average helps reveal the longer-term pattern
    rolling_average = daily_volume.rolling(
        window=7,
        min_periods=1
    ).mean()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 6))
    plt.plot(
        daily_volume.index,
        daily_volume.values,
        label="Daily tickets",
        alpha=0.55
    )
    plt.plot(
        rolling_average.index,
        rolling_average.values,
        label="7-day rolling average",
        linewidth=2
    )

    plt.axvline(
        pd.Timestamp("2013-09-01"),
        linestyle="--",
        label="September 2013"
    )

    plt.title("Daily ITSM Ticket Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Number of tickets")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "daily_ticket_volume.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nChart saved to: {output_path}")
    print("\nAnalysis completed. Source CSV was not modified.")


if __name__ == "__main__":
    analyze_daily_volume()
