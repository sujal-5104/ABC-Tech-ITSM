
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"


def investigate_trends():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig", low_memory=False)

    # Parse opening timestamps
    df["Open_Time_Parsed"] = pd.to_datetime(
        df["Open_Time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    df = df.dropna(subset=["Open_Time_Parsed"]).copy()
    df["Month"] = df["Open_Time_Parsed"].dt.to_period("M").astype(str)

    print("=" * 60)
    print("ITSM INCIDENT TREND INVESTIGATION")
    print("=" * 60)

    # 1. Monthly volume by ticket category
    print("\n1. MONTHLY TICKET COUNTS BY CATEGORY")

    monthly_category = pd.crosstab(df["Month"], df["Category"])
    print(monthly_category.to_string())

    monthly_category.plot(kind="line", figsize=(12, 6), marker="o")
    plt.title("Monthly Ticket Volume by Category")
    plt.xlabel("Month")
    plt.ylabel("Number of Tickets")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_volume_by_category.png", dpi=150)
    plt.close()

    # 2. Check ticket ID ranges by year
    print("\n2. INCIDENT ID RANGES BY YEAR")

    df["Year"] = df["Open_Time_Parsed"].dt.year
    id_summary = df.groupby("Year")["Incident_ID"].agg(
        ["count", "nunique", "min", "max"]
    )
    print(id_summary.to_string())

    # 3. Weekly volume to inspect the sudden increase
    print("\n3. WEEKLY TICKET VOLUME")

    weekly = (
        df.set_index("Open_Time_Parsed")
        .resample("W")
        .size()
    )

    print(weekly.loc["2013-07-01":"2013-12-31"].to_string())

    plt.figure(figsize=(12, 5))
    weekly.plot()
    plt.title("Weekly ITSM Ticket Volume (July–December 2013)")
    plt.xlabel("Week")
    plt.ylabel("Number of Tickets")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weekly_ticket_volume_2013.png", dpi=150)
    plt.close()

    # 4. Daily record counts in the period with the sharp rise
    print("\n4. DAILY COUNTS: SEPTEMBER–OCTOBER 2013")

    daily = (
        df.set_index("Open_Time_Parsed")
        .resample("D")
        .size()
    )

    print(daily.loc["2013-09-01":"2013-10-31"].to_string())

    print("\nInvestigation completed.")
    print(f"Charts saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    investigate_trends()
