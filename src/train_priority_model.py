
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "priority_training_data.csv"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, low_memory=False)

    target = "High_Priority"
    X = df.drop(columns=[target, "Impact", "Urgency"])
    y = df[target].astype(int)

    # Diagnose whether features strongly relate to the target.
    print("\n" + "=" * 65)
    print("FEATURE-TO-TARGET DIAGNOSTICS")
    print("=" * 65)

    for column in X.columns:
        table = pd.crosstab(X[column], y, normalize="index")
        high_rate = (
            table[1]
            if 1 in table.columns
            else pd.Series(dtype=float)
        )

        print(f"\nFeature: {column}")
        print("Unique values:", X[column].nunique())
        print("Highest high-priority rate by category:")
        print(
            high_rate.sort_values(ascending=False)
            .head(10)
            .round(4)
            .to_string()
        )

    categorical_features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
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
                categorical_features,
            )
        ]
    )

    models = {
        "Dummy baseline": DummyClassifier(strategy="most_frequent"),
        "Class-weighted Logistic Regression": Pipeline(
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
        ),
    }

    results = []
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("ABC TECH ITSM - HIGH-PRIORITY TICKET CLASSIFICATION")
    print("=" * 65)
    print(f"Total records: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"High-priority training records: {int(y_train.sum())}")
    print(f"High-priority testing records: {int(y_test.sum())}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)[:, 1]
            avg_precision = average_precision_score(y_test, probabilities)
        else:
            avg_precision = float("nan")

        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)

        results.append(
            {
                "Model": name,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "Average_Precision": avg_precision,
            }
        )

        print(f"\n--- {name} ---")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Other priority", "High priority"],
                zero_division=0,
            )
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=[0, 1],
            display_labels=["Other", "High"],
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(f"Confusion Matrix - {name}")
        fig.tight_layout()

        filename = (
            "priority_confusion_dummy.png"
            if name == "Dummy baseline"
            else "priority_confusion_logistic_regression.png"
        )
        fig.savefig(FIGURE_DIR / filename, dpi=150)
        plt.close(fig)

    results_df = pd.DataFrame(results)
    results_file = REPORT_DIR / "priority_model_comparison.csv"
    results_df.to_csv(results_file, index=False)

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nComparison saved to: {results_file}")
    print(f"Confusion matrices saved in: {FIGURE_DIR}")

    print(
        "\nNote: This uses a stratified random split, not a time-based test. "
        "Investigate unusually perfect results before relying on the model."
    )


if __name__ == "__main__":
    main()
