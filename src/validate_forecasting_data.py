
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "daily_forecasting_data.csv"


def validate_forecasting_data():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE, parse_dates=["Open_Date"])

    print("=" * 60)
    print("FORECASTING DATASET VALIDATION")
    print("=" * 60)

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    print("\nLast 5 rows:")
    print(df.tail().to_string(index=False))

    print("\nDataset shape:", df.shape)
    print("\nColumn data types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate dates:", df["Open_Date"].duplicated().sum())
    print("Duplicate rows:", df.duplicated().sum())

    expected_dates = pd.date_range(
        start="2013-10-01",
        end="2014-03-31",
        freq="D"
    )

    print("\nExpected calendar days:", len(expected_dates))
    print("Actual calendar days:", len(df))
    print(
        "Date range complete:",
        df["Open_Date"].reset_index(drop=True).equals(
            pd.Series(expected_dates, name="Open_Date")
        )
    )

    print("\nTicket count summary:")
    print(df["Ticket_Count"].describe().round(2).to_string())

    print("\nNegative ticket counts:", (df["Ticket_Count"] < 0).sum())

    print("\nValidation finished.")


if __name__ == "__main__":
    validate_forecasting_data()
