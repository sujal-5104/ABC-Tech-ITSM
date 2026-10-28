
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "priority_classifier.joblib"

THRESHOLD = 0.90

FEATURE_COLUMNS = [
    "CI_Cat",
    "CI_Subcat",
    "Category",
    "Open_Weekday",
    "Open_Hour",
]


def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)

    df["Priority"] = pd.to_numeric(
        df["Priority"], errors="coerce"
    )
    df["Open_Time"] = pd.to_datetime(
        df["Open_Time"],
        dayfirst=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["Priority", "Open_Time"]
    ).copy()

    df = df[df["Priority"].isin([1, 2, 3, 4, 5])].copy()

    df["High_Priority"] = (
        df["Priority"].isin([1, 2]).astype(int)
    )

    df["Open_Weekday"] = (
        df["Open_Time"].dt.day_of_week.astype(str)
    )
    df["Open_Hour"] = df["Open_Time"].dt.hour.astype(str)

    for column in ["CI_Cat", "CI_Subcat", "Category"]:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    # Keep the chronological final test period excluded.
    df = df.sort_values("Open_Time").reset_index(drop=True)
    train_end = int(len(df) * 0.80)
    training_data = df.iloc[:train_end].copy()

    X_train = training_data[FEATURE_COLUMNS]
    y_train = training_data["High_Priority"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "pipeline": pipeline,
        "threshold": THRESHOLD,
        "feature_columns": FEATURE_COLUMNS,
        "target_definition": (
            "High priority = original Priority 1 or 2"
        ),
        "training_records": len(training_data),
    }

    joblib.dump(model_bundle, MODEL_FILE)

    # Verify that the saved artifact can be loaded.
    loaded_bundle = joblib.load(MODEL_FILE)

    print("=" * 65)
    print("PRIORITY CLASSIFICATION PIPELINE SAVED")
    print("=" * 65)
    print("Training records:", len(training_data))
    print("Training high-priority records:", int(y_train.sum()))
    print("Decision threshold:", loaded_bundle["threshold"])
    print("Features:", loaded_bundle["feature_columns"])
    print("Saved model:", MODEL_FILE)
    print("Reload verification: successful")


if __name__ == "__main__":
    main()
