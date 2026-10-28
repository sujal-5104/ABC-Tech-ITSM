
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "priority_training_data.csv"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Convert target labels to numeric and exclude missing priorities.
    df["Priority"] = pd.to_numeric(df["Priority"], errors="coerce")
    df = df[df["Priority"].isin([1, 2, 3, 4, 5])].copy()

    # Binary target: Priority 1 or 2 = 1; Priority 3, 4, or 5 = 0.
    df["High_Priority"] = df["Priority"].isin([1, 2]).astype(int)

    # Keep only candidate features that could be available at ticket creation.
    feature_columns = [
        "Impact",
        "Urgency",
        "CI_Cat",
        "CI_Subcat",
        "Category",
        "Open_Time",
    ]

    # Remove identifiers and post-resolution fields from model inputs.
    prepared = df[feature_columns + ["High_Priority"]].copy()

    # Normalize mixed categorical/numeric fields as text.
    for column in ["Impact", "Urgency", "CI_Cat", "CI_Subcat", "Category"]:
        prepared[column] = (
            prepared[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )
        prepared[column] = prepared[column].replace("", "Unknown")

    # Extract calendar features from ticket opening time.
    opened = pd.to_datetime(
        prepared.pop("Open_Time"),
        dayfirst=True,
        errors="coerce",
    )

    prepared["Open_Weekday"] = opened.dt.dayofweek.astype("Int64").astype(str)
    prepared["Open_Hour"] = opened.dt.hour.astype("Int64").astype(str)

    prepared["Open_Weekday"] = prepared["Open_Weekday"].replace("<NA>", "Unknown")
    prepared["Open_Hour"] = prepared["Open_Hour"].replace("<NA>", "Unknown")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(OUTPUT_FILE, index=False)

    print("=" * 55)
    print("PRIORITY TRAINING DATA PREPARATION")
    print("=" * 55)
    print(f"Input rows: {len(pd.read_csv(DATA_FILE, low_memory=False))}")
    print(f"Rows with known priorities: {len(prepared)}")
    print(f"High-priority tickets: {int(prepared['High_Priority'].sum())}")
    print(
        "Other-priority tickets:",
        int((prepared["High_Priority"] == 0).sum()),
    )
    print("\nTarget distribution:")
    print(prepared["High_Priority"].value_counts().sort_index().to_string())
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
