
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "priority_training_data.csv"


def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Inspect the relationship between inputs and the target.
    print("=" * 60)
    print("CHECKING PRIORITY RULES")
    print("=" * 60)

    print("\nHigh-priority rate by Impact and Urgency:")
    print(
        df.pivot_table(
            index="Impact",
            columns="Urgency",
            values="High_Priority",
            aggfunc="mean",
            observed=False,
        ).round(3).to_string()
    )

    print("\nHigh-priority rate by Impact alone:")
    print(
        df.groupby("Impact", dropna=False)["High_Priority"]
        .agg(["count", "sum", "mean"])
        .round(4)
        .to_string()
    )

    print("\nHigh-priority rate by Urgency alone:")
    print(
        df.groupby("Urgency", dropna=False)["High_Priority"]
        .agg(["count", "sum", "mean"])
        .round(4)
        .to_string()
    )

    # Count unique target values for each Impact/Urgency combination.
    combinations = (
        df.groupby(["Impact", "Urgency"], dropna=False)["High_Priority"]
        .agg(["count", "nunique", "mean"])
        .reset_index()
    )

    print("\nImpact/Urgency combinations with mixed target labels:")
    mixed = combinations[combinations["nunique"] > 1]
    if mixed.empty:
        print("None: each observed combination has one target label.")
    else:
        print(mixed.round(4).to_string(index=False))

    print("\nTotal rows checked:", len(df))


if __name__ == "__main__":
    main()
