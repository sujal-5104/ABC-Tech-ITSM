
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"
REPORT_DIR = PROJECT_ROOT / "reports"


def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)

    df["Priority"] = pd.to_numeric(df["Priority"], errors="coerce")
    df["Open_Time"] = pd.to_datetime(
        df["Open_Time"], dayfirst=True, errors="coerce"
    )

    df = df.dropna(subset=["Priority", "Open_Time"]).copy()
    df = df[df["Priority"].isin([1, 2, 3, 4, 5])].copy()
    df["High_Priority"] = df["Priority"].isin([1, 2]).astype(int)

    df["Open_Weekday"] = df["Open_Time"].dt.day_of_week.astype(str)
    df["Open_Hour"] = df["Open_Time"].dt.hour.astype(str)

    # Exclude Impact and Urgency to test alternative signals.
    feature_columns = [
        "CI_Cat",
        "CI_Subcat",
        "Category",
        "Open_Weekday",
        "Open_Hour",
    ]

    for column in ["CI_Cat", "CI_Subcat", "Category"]:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()
        df[column] = df[column].replace("", "Unknown")

    df = df.sort_values("Open_Time").reset_index(drop=True)

    split_index = int(len(df) * 0.80)
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()

    X_train = train[feature_columns]
    y_train = train["High_Priority"]
    X_test = test[feature_columns]
    y_test = test["High_Priority"]

    print("=" * 65)
    print("CHRONOLOGICAL HIGH-PRIORITY MODEL EVALUATION")
    print("=" * 65)
    print("Training period:", train["Open_Time"].min(), "to", train["Open_Time"].max())
    print("Testing period:", test["Open_Time"].min(), "to", test["Open_Time"].max())
    print("Training records:", len(train))
    print("Testing records:", len(test))
    print("Training high-priority tickets:", int(y_train.sum()))
    print("Testing high-priority tickets:", int(y_test.sum()))

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one target class; "
            "a classifier cannot be trained reliably."
        )

    if y_test.sum() == 0 or y_test.nunique() < 2:
        raise ValueError(
            "The chronological test set does not contain both target classes. "
            "A different date boundary is needed."
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                feature_columns,
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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

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

    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nHigh-priority precision:", round(
        precision_score(y_test, predictions, zero_division=0), 4
    ))
    print("High-priority recall:", round(
        recall_score(y_test, predictions, zero_division=0), 4
    ))
    print("High-priority F1:", round(
        f1_score(y_test, predictions, zero_division=0), 4
    ))
    print("Average precision:", round(
        average_precision_score(y_test, probabilities), 4
    ))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(
        {
            "Open_Time": test["Open_Time"].values,
            "Actual_High_Priority": y_test.values,
            "Predicted_High_Priority": predictions,
            "Predicted_High_Priority_Probability": probabilities,
        }
    )
    output_file = REPORT_DIR / "priority_chronological_predictions.csv"
    results.to_csv(output_file, index=False)
    print("\nPredictions saved to:", output_file)


if __name__ == "__main__":
    main()
