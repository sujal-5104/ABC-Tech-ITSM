
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "daily_forecasting_data.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


def train_volume_forecast():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE, parse_dates=["Open_Date"])
    df = df.sort_values("Open_Date").reset_index(drop=True)

    # Features based on the date and past ticket volumes only
    df["Day_Of_Week"] = df["Open_Date"].dt.day_of_week
    df["Is_Weekend"] = (df["Day_Of_Week"] >= 5).astype(int)
    df["Lag_1"] = df["Ticket_Count"].shift(1)
    df["Lag_7"] = df["Ticket_Count"].shift(7)
    df["Rolling_Mean_7"] = (
        df["Ticket_Count"].shift(1).rolling(window=7).mean()
    )

    # Lag features require the first seven days for history
    model_df = df.dropna().copy()

    feature_columns = [
        "Day_Of_Week",
        "Is_Weekend",
        "Lag_1",
        "Lag_7",
        "Rolling_Mean_7"
    ]

    X = model_df[feature_columns]
    y = model_df["Ticket_Count"]

    # Chronological split: latest 20% is held out for testing
    split_index = int(len(model_df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    test_dates = model_df["Open_Date"].iloc[split_index:]

    # Baseline: same ticket count as seven days earlier
    baseline_predictions = model_df["Lag_7"].iloc[split_index:]

    # Train the Random Forest model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate metrics
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    model_mae = mean_absolute_error(y_test, predictions)

    baseline_rmse = mean_squared_error(
        y_test, baseline_predictions
    ) ** 0.5

    model_rmse = mean_squared_error(
        y_test, predictions
    ) ** 0.5

    model_r2 = r2_score(y_test, predictions)

    print("=" * 65)
    print("ITSM DAILY TICKET FORECASTING RESULTS")
    print("=" * 65)

    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")
    print(f"Test period: {test_dates.min().date()} to "
          f"{test_dates.max().date()}")

    print("\nBaseline (previous week's volume):")
    print(f"MAE:  {baseline_mae:.2f}")
    print(f"RMSE: {baseline_rmse:.2f}")

    print("\nRandom Forest:")
    print(f"MAE:  {model_mae:.2f}")
    print(f"RMSE: {model_rmse:.2f}")
    print(f"R²:   {model_r2:.4f}")

    print("\nFeature importance:")
    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False)

    print(importance.round(4).to_string())

    # Save actual and predicted counts for evaluation
    results = pd.DataFrame({
        "Open_Date": test_dates.to_numpy(),
        "Actual_Tickets": y_test.to_numpy(),
        "Baseline_Prediction": baseline_predictions.to_numpy(),
        "Random_Forest_Prediction": predictions.round(2)
    })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results_path = REPORTS_DIR / "volume_forecast_test_results.csv"
    results.to_csv(results_path, index=False)

    # Plot test-period predictions
    plt.figure(figsize=(14, 6))
    plt.plot(
        results["Open_Date"],
        results["Actual_Tickets"],
        label="Actual tickets"
    )
    plt.plot(
        results["Open_Date"],
        results["Baseline_Prediction"],
        label="Previous-week baseline"
    )
    plt.plot(
        results["Open_Date"],
        results["Random_Forest_Prediction"],
        label="Random Forest"
    )

    plt.title("ITSM Daily Ticket Forecast: Test Period")
    plt.xlabel("Date")
    plt.ylabel("Ticket count")
    plt.legend()
    plt.tight_layout()

    chart_path = FIGURES_DIR / "volume_forecast_test.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nTest results saved to: {results_path}")
    print(f"Forecast chart saved to: {chart_path}")
    print("\nModel training and evaluation completed.")


if __name__ == "__main__":
    train_volume_forecast()
