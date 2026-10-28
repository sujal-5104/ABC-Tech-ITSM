
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def run_eda():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", low_memory=False)

    print("=" * 60)
    print("ABC TECH ITSM - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Dataset overview
    print("\n1. DATASET OVERVIEW")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Unique incident IDs: {df['Incident_ID'].nunique():,}")

    # 2. Ticket category distribution
    print("\n2. TICKET CATEGORY DISTRIBUTION")
    print(df["Category"].value_counts(dropna=False).to_string())

    category_counts = df["Category"].fillna("Missing").value_counts()
    plt.figure(figsize=(10, 6))
    category_counts.plot(kind="bar")
    plt.title("Ticket Count by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Tickets")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ticket_categories.png", dpi=150)
    plt.close()

    # 3. Priority distribution
    print("\n3. PRIORITY DISTRIBUTION")
    print(df["Priority"].value_counts(dropna=False).sort_index().to_string())

    priority_counts = (
        df["Priority"].fillna("Missing").astype(str).value_counts()
    )
    plt.figure(figsize=(8, 5))
    priority_counts.plot(kind="bar")
    plt.title("Ticket Count by Priority")
    plt.xlabel("Priority")
    plt.ylabel("Number of Tickets")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "priority_distribution.png", dpi=150)
    plt.close()

    # 4. Configuration item categories
    print("\n4. TOP CONFIGURATION ITEM CATEGORIES")
    print(df["CI_Cat"].value_counts(dropna=False).head(15).to_string())

    # 5. Open dates and monthly ticket counts
    print("\n5. INCIDENT TRENDS OVER TIME")

    df["Open_Time_Parsed"] = pd.to_datetime(
        df["Open_Time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    invalid_dates = df["Open_Time_Parsed"].isna().sum()
    print(f"Dates that could not be parsed: {invalid_dates}")

    valid_dates = df.dropna(subset=["Open_Time_Parsed"]).copy()

    if not valid_dates.empty:
        monthly_counts = (
            valid_dates
            .set_index("Open_Time_Parsed")
            .resample("MS")
            .size()
        )

        print("\nMonthly ticket counts:")
        print(monthly_counts.to_string())

        plt.figure(figsize=(12, 5))
        monthly_counts.plot()
        plt.title("Monthly ITSM Ticket Volume")
        plt.xlabel("Month")
        plt.ylabel("Number of Tickets")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "monthly_ticket_volume.png", dpi=150)
        plt.close()

    # 6. Missing values
    print("\n6. TOP 10 COLUMNS BY MISSING VALUES")
    missing_counts = df.isna().sum().sort_values(ascending=False)
    print(missing_counts.head(10).to_string())

    print("\nEDA completed.")
    print(f"Charts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
