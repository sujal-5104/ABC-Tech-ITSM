
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor



PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "daily_forecasting_data.csv"


def evaluate_forecast_models():
    if not DATA_FILE.exists():
        print(f"Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE, parse_dates=["Open_Date"])
    df = df.sort_values("Open_Date").reset_index(drop=True)

    # Calendar features
    df["Day_Of_Week"] = df["Open_Date"].dt.day_of_week
    df["Is_Weekend"] = (df["Day_Of_Week"] >= 5).astype(int)

    # Historical features
    df["Lag_1"] = df["Ticket_Count"].shift(1)
    df["Lag_7"] = df["Ticket_Count"].shift(7)
    df["Rolling_Mean_7"] = (
        df["Ticket_Count"].shift(1).rolling(7).mean()
    )

    model_df = df.dropna().reset_index(drop=True)

    features = [
        "Day_Of_Week",
        "Is_Weekend",
        "Lag_1",
        "Lag_7",
        "Rolling_Mean_7",
    ]

    split_index = int(len(model_df) * 0.80)

    train = model_df.iloc[:split_index]
    test = model_df.iloc[split_index:]

    X_train = train[features]
    y_train = train["Ticket_Count"]
    X_test = test[features]
    y_test = test["Ticket_Count"]

    # Model 1: Previous-week baseline
    baseline_predictions = test["Lag_7"].to_numpy()

    # Model 2: Weekday-aware median prediction
    weekday_medians = train.groupby("Day_Of_Week")[
        "Ticket_Count"
    ].median()

    weekday_predictions = (
        test["Day_Of_Week"].map(weekday_medians).to_numpy()
    )

    # Model 3: Random Forest
    forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    forest_predictions = forest.predict(X_test)

    predictions = {
        "Previous-week baseline": baseline_predictions,
        "Weekday median": weekday_predictions,
        "Random Forest": forest_predictions,
    }

    results = []

    print("=" * 65)
    print("ITSM FORECAST MODEL COMPARISON")
    print("=" * 65)
    print(f"Training observations: {len(train)}")
    print(f"Testing observations: {len(test)}")
    print(
        f"Test period: {test['Open_Date'].min().date()} to "
        f"{test['Open_Date'].max().date()}"
    )

    for model_name, model_predictions in predictions.items():
        mae = mean_absolute_error(y_test, model_predictions)
        rmse = mean_squared_error(
            y_test, model_predictions
        ) ** 0.5

        results.append({
            "Model": model_name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
        })

        print(f"\n{model_name}")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")

    results_df = pd.DataFrame(results).sort_values("MAE")

    output_file = (
        PROJECT_ROOT / "reports" / "forecast_model_comparison.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 65)
    print("MODEL RANKING (lower MAE is better)")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    evaluate_forecast_models()
