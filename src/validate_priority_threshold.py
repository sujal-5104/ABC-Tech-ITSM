
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURE_COLUMNS = [
    "CI_Cat",
    "CI_Subcat",
    "Category",
    "Open_Weekday",
    "Open_Hour",
]

THRESHOLDS = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90,
]


def main():
    # 1. Load and prepare the original dataset
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

    # 2. Sort chronologically before splitting
    df = df.sort_values("Open_Time").reset_index(drop=True)

    total_records = len(df)
    train_end = int(total_records * 0.60)
    validation_end = int(total_records * 0.80)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    print("=" * 70)
    print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    for name, part in [
        ("Training", train),
        ("Validation", validation),
        ("Final test", test),
    ]:
        print(f"\n{name} records: {len(part):,}")
        print(
            f"{name} period: "
            f"{part['Open_Time'].min()} to "
            f"{part['Open_Time'].max()}"
        )
        print(
            f"{name} high-priority records: "
            f"{int(part['High_Priority'].sum()):,}"
        )

    if (
        train["High_Priority"].nunique() < 2
        or validation["High_Priority"].nunique() < 2
        or test["High_Priority"].nunique() < 2
    ):
        raise ValueError(
            "One of the chronological splits does not contain "
            "both classes. Review the split boundaries."
        )

    # 3. Build a pipeline using the same feature strategy
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

    model = Pipeline(
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

    X_train = train[FEATURE_COLUMNS]
    y_train = train["High_Priority"]

    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["High_Priority"]

    # 4. Train on training data only
    model.fit(X_train, y_train)

    # 5. Compare thresholds on validation data only
    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    rows = []

    for threshold in THRESHOLDS:
        predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1],
        ).ravel()

        rows.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "F1_Score": f1_score(
                    y_validation,
                    predictions,
                    zero_division=0,
                ),
                "True_Positives": int(tp),
                "False_Positives": int(fp),
                "False_Negatives": int(fn),
                "True_Negatives": int(tn),
                "Predicted_High": int(predictions.sum()),
            }
        )

    results = pd.DataFrame(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = (
        REPORT_DIR / "priority_validation_thresholds.csv"
    )
    results.to_csv(output_file, index=False)

    print("\n" + "=" * 70)
    print("VALIDATION THRESHOLD COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\nAverage precision on validation data:",
          f"{average_precision_score(y_validation, validation_probabilities):.4f}")

    best_row = results.loc[results["F1_Score"].idxmax()]

    print("\nBest threshold by validation F1-score:",
          f"{best_row['Threshold']:.2f}")
    print("Validation F1-score:",
          f"{best_row['F1_Score']:.4f}")
    print("Validation precision:",
          f"{best_row['Precision']:.4f}")
    print("Validation recall:",
          f"{best_row['Recall']:.4f}")

    print("\nValidation results saved to:", output_file)
    print(
        "\nNote: The final test set has not been evaluated "
        "or used to select a threshold."
    )


if __name__ == "__main__":
    main()
