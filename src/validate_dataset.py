
from pathlib import Path

import pandas as pd


# Locate the exported dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"


def validate_dataset():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    print("Loading exported ITSM dataset...")
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", low_memory=False)

    print("\n" + "=" * 50)
    print("1. DATASET DIMENSIONS")
    print("=" * 50)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n" + "=" * 50)
    print("2. COLUMN NAMES")
    print("=" * 50)
    print(df.columns.tolist())

    print("\n" + "=" * 50)
    print("3. DATA TYPES")
    print("=" * 50)
    print(df.dtypes.to_string())

    print("\n" + "=" * 50)
    print("4. MISSING VALUES")
    print("=" * 50)
    missing = df.isnull().sum()
    missing_report = pd.DataFrame({
        "Missing_Count": missing,
        "Missing_Percentage": (missing / len(df) * 100).round(2)
    })
    print(missing_report[missing_report["Missing_Count"] > 0].to_string())

    if missing.sum() == 0:
        print("Pandas found no standard missing values.")

    print("\n" + "=" * 50)
    print("5. DUPLICATE ROWS")
    print("=" * 50)
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    print("\n" + "=" * 50)
    print("6. PRIORITY DISTRIBUTION")
    print("=" * 50)
    if "Priority" in df.columns:
        print(df["Priority"].value_counts(dropna=False).to_string())
    else:
        print("Priority column not found.")

    print("\n" + "=" * 50)
    print("7. SAMPLE RECORDS")
    print("=" * 50)
    print(df.head(5).to_string(index=False))

    print("\nValidation completed. Original CSV was not modified.")


if __name__ == "__main__":
    validate_dataset()
