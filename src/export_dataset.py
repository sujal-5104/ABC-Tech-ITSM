
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from database import engine


# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "itsm_dataset.csv"


def export_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    query = text("""
        SELECT *
        FROM dataset_list
    """)

    try:
        print("Connecting to the ITSM database...")

        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection)

        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        print("\nDataset exported successfully!")
        print(f"Total rows: {len(df):,}")
        print(f"Total columns: {len(df.columns)}")
        print(f"File saved at: {OUTPUT_FILE}")

        print("\nFirst 5 rows:")
        print(df.head())

    except SQLAlchemyError as error:
        print("Database export failed.")
        print(f"Error type: {type(error).__name__}")
        print("Please check your database connection and permissions.")


if __name__ == "__main__":
    export_dataset()
