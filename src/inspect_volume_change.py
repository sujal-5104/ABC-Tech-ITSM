
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"


def inspect_volume_change():
    # Check whether the dataset exists
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    # Load the existing dataset
    df = pd.read_csv(
        DATA_FILE,
        encoding="utf-8-sig",
        low_memory=False
    )

    # Parse incident opening timestamps
    df["Open_Time_Parsed"] = pd.to_datetime(
        df["Open_Time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    # Remove records with invalid opening timestamps
    df = df.dropna(subset=["Open_Time_Parsed"]).copy()

    print("=" * 65)
    print("INSPECTING THE ITSM TICKET VOLUME CHANGE")
    print("=" * 65)

    # Examine representative periods before and after the increase
    periods = {
        "Before increase": ("2013-08-01", "2013-08-31"),
        "Early increase": ("2013-09-23", "2013-09-30"),
        "High-volume period": ("2013-10-01", "2013-10-31"),
        "Later period": ("2014-01-01", "2014-01-31"),
    }

    for label, (start, end) in periods.items():

        period_df = df[
            df["Open_Time_Parsed"].between(
                pd.Timestamp(start),
                pd.Timestamp(end)
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1)
            )
        ].copy()

        print(f"\n{'=' * 65}")
        print(f"{label}: {start} to {end}")
        print(f"Record count: {len(period_df):,}")

        print("\nCategories:")
        print(
            period_df["Category"]
            .value_counts(dropna=False)
            .to_string()
        )

        # Display the first five records chronologically
        print("\nIncident ID sample:")
        print(
            period_df[
                [
                    "Open_Time_Parsed",
                    "Open_Time",
                    "Incident_ID",
                    "Category",
                    "Status",
                ]
            ]
            .sort_values("Open_Time_Parsed")
            .head(5)
            .to_string(index=False)
        )

        # Display the last five records chronologically
        print("\nLatest IDs in period:")
        print(
            period_df[
                [
                    "Open_Time_Parsed",
                    "Open_Time",
                    "Incident_ID",
                    "Category",
                    "Status",
                ]
            ]
            .sort_values("Open_Time_Parsed")
            .tail(5)
            .to_string(index=False)
        )

    # Examine weekday versus weekend patterns during October 2013
    october = df[
        (df["Open_Time_Parsed"] >= "2013-10-01")
        & (df["Open_Time_Parsed"] < "2013-11-01")
    ].copy()

    october["Day_Type"] = october["Open_Time_Parsed"].dt.dayofweek.map(
        lambda day: "Weekend" if day >= 5 else "Weekday"
    )

    print(f"\n{'=' * 65}")
    print("OCTOBER 2013: WEEKDAY VS WEEKEND")
    print("=" * 65)

    if not october.empty:
        summary = (
            october.groupby("Day_Type")
            .agg(
                tickets=("Incident_ID", "size"),
                unique_ids=("Incident_ID", "nunique"),
                average_hour=(
                    "Open_Time_Parsed",
                    lambda values: round(values.dt.hour.mean(), 2)
                )
            )
        )

        print(summary.to_string())
    else:
        print("No records found for October 2013.")

    print("\n" + "=" * 65)
    print("Inspection completed. Source CSV was not modified.")


if __name__ == "__main__":
    inspect_volume_change()
