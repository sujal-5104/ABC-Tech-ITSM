
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"


def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)

    df["Priority"] = pd.to_numeric(df["Priority"], errors="coerce")
    df["Open_Time"] = pd.to_datetime(
        df["Open_Time"],
        dayfirst=True,
        errors="coerce",
    )

    df = df.dropna(subset=["Open_Time", "Priority"]).copy()
    df["High_Priority"] = df["Priority"].isin([1, 2]).astype(int)
    df["Month"] = df["Open_Time"].dt.to_period("M").astype(str)

    print("=" * 60)
    print("PRIORITY TIMELINE ANALYSIS")
    print("=" * 60)
    print("First ticket date:", df["Open_Time"].min())
    print("Last ticket date:", df["Open_Time"].max())
    print("Total dated, labelled tickets:", len(df))

    monthly = df.groupby("Month").agg(
        Tickets=("High_Priority", "size"),
        High_Priority_Tickets=("High_Priority", "sum"),
        High_Priority_Rate=("High_Priority", "mean"),
    )

    monthly["High_Priority_Rate"] *= 100

    print("\nMonthly ticket counts and high-priority rates:")
    print(monthly.round(2).to_string())


if __name__ == "__main__":
    main()
