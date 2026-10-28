
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
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

THRESHOLD = 0.90

FEATURE_COLUMNS = [
    "CI_Cat",
    "CI_Subcat",
    "Category",
    "Open_Weekday",
    "Open_Hour",
]


def main():
    # 1. Load and prepare data
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

    # 2. Reproduce the same chronological split
    df = df.sort_values("Open_Time").reset_index(drop=True)

    total_records = len(df)
    train_end = int(total_records * 0.60)
    validation_end = int(total_records * 0.80)

    # Train on the first 80%; test on the latest 20%
    train_validation = df.iloc[:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    X_train = train_validation[FEATURE_COLUMNS]
    y_train = train_validation["High_Priority"]

    X_test = test[FEATURE_COLUMNS]
    y_test = test["High_Priority"]

    print("=" * 70)
    print("FINAL CHRONOLOGICAL PRIORITY MODEL EVALUATION")
    print("=" * 70)
    print("Training records:", len(train_validation))
    print("Final test records:", len(test))
    print("Final test high-priority tickets:", int(y_test.sum()))
    print("Selected validation threshold:", THRESHOLD)
    print("Training ends:", train_validation["Open_Time"].max())
    print("Testing starts:", test["Open_Time"].min())

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Training and test sets must both contain both classes."
        )

    # 3. Build preprocessing and classification pipeline
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

    # 4. Fit on the earliest 80% only
    model.fit(X_train, y_train)

    # 5. Evaluate once on the latest 20%
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Other priority", "High priority"],
            zero_division=0,
        )
    )

    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print([[int(tn), int(fp)], [int(fn), int(tp)]])

    print("\nFinal test metrics at threshold 0.90:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Average precision: {average_precision_score(y_test, probabilities):.4f}")

    # 6. Save final test predictions
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output = test[
        ["Incident_ID", "Open_Time", "Priority", "High_Priority"]
    ].copy()

    output = output.rename(
        columns={"High_Priority": "Actual_High_Priority"}
    )

    output["Predicted_High_Priority_Probability"] = probabilities
    output["Predicted_High_Priority"] = predictions
    output["Decision_Threshold"] = THRESHOLD

    output_file = (
        REPORT_DIR / "priority_final_test_predictions.csv"
    )
    output.to_csv(output_file, index=False)

    print("\nFinal test predictions saved to:", output_file)
    print("Threshold was selected using validation data.")


if __name__ == "__main__":
    main()
