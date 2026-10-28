
from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_FILE = (
    PROJECT_ROOT / "reports" / "priority_chronological_predictions.csv"
)


def main():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {PREDICTIONS_FILE}"
        )

    df = pd.read_csv(PREDICTIONS_FILE)

    y_true = df["Actual_High_Priority"].astype(int)
    probabilities = df["Predicted_High_Priority_Probability"]

    results = []

    print("=" * 70)
    print("HIGH-PRIORITY DECISION THRESHOLD COMPARISON")
    print("=" * 70)

    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(
                    y_true, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_true, predictions, zero_division=0
                ),
                "F1_Score": f1_score(
                    y_true, predictions, zero_division=0
                ),
                "True_Positives": tp,
                "False_Positives": fp,
                "False_Negatives": fn,
                "Predicted_High": int(predictions.sum()),
            }
        )

    results_df = pd.DataFrame(results)
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    output_file = PROJECT_ROOT / "reports" / "priority_threshold_comparison.csv"
    results_df.to_csv(output_file, index=False)
    print("\nSaved threshold comparison to:", output_file)


if __name__ == "__main__":
    main()
