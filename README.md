<<<<<<< HEAD
# ABC Tech ITSM AI — Intelligent IT Service Management

An end-to-end machine learning project for analyzing IT service management data, forecasting daily incident volumes, and identifying potentially high-priority IT tickets.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Objectives](#2-project-objectives)
3. [Project Components](#3-project-components)
4. [Dataset Description](#4-dataset-description)
5. [Component 1: Daily Ticket Volume Forecasting](#5-component-1-daily-ticket-volume-forecasting)
6. [Component 2: High-Priority Incident Classification](#6-component-2-high-priority-incident-classification)
7. [Project Results](#7-project-results)
8. [Project Structure](#8-project-structure)
9. [Installation and Setup](#9-installation-and-setup)
10. [Running the Project](#10-running-the-project)
11. [Technologies Used](#11-technologies-used)
12. [Limitations](#12-limitations)
13. [Future Improvements](#13-future-improvements)
14. [Conclusion](#14-conclusion)

## 1. Project Overview

The ABC Tech ITSM AI project applies machine learning and data analysis techniques to historical IT service management data.

IT support teams handle incidents with different levels of urgency and workload. Analyzing historical incident data can help organizations understand ticket patterns, anticipate support demand, and identify incidents that may require additional attention.

This project currently includes two machine learning components:

- **Daily Ticket Volume Forecasting:** Estimates future ticket volumes using historical ticket counts, calendar patterns, and forecasting baselines.
- **High-Priority Incident Classification:** Predicts whether an incident belongs to the high-priority category using selected ticket attributes.

The project emphasizes reproducible data preparation, chronological model evaluation, performance comparison, and transparent reporting of model limitations.

## 2. Project Objectives

- Analyze historical ITSM incident data.
- Validate and prepare datasets for machine learning.
- Explore daily and weekly ticket-volume patterns.
- Forecast ticket volumes using historical observations.
- Compare machine learning models with simple baselines.
- Identify potentially high-priority incidents.
- Evaluate classification performance using precision, recall, F1-score, and average precision.
- Save predictions, comparison tables, and visualizations.
- Establish a foundation for future dashboard development and ITSM decision support.

## 3. Project Components

| Component | Purpose | Current status |
|---|---|---|
| Data extraction and preparation | Prepare historical ITSM data for analysis | Completed |
| Daily ticket forecasting | Estimate future ticket volume | Evaluation completed |
| Forecast model comparison | Compare three forecasting methods | Completed |
| High-priority classification | Identify potentially high-priority tickets | Chronological evaluation completed |
| Threshold comparison | Examine precision–recall trade-offs | Exploratory analysis completed |
| Interactive dashboard | Present forecasts and predictions through a user interface | Planned |

## 4. Dataset Description

The project uses historical ITSM incident records containing ticket attributes, priority information, timestamps, configuration-item details, and incident-management fields.

### Source dataset

The exported dataset is stored at:

`data/itsm_dataset.csv`

The original dataset contains **46,606 records and 25 columns**.

Relevant fields include:

- `Incident_ID`
- `Status`
- `Impact`
- `Urgency`
- `Priority`
- `CI_Cat`
- `CI_Subcat`
- `Category`
- `Open_Time`
- `Resolved_Time`
- `Close_Time`
- `Handle_Time_hrs`
- `No_of_Reassignments`
- `No_of_Related_Interactions`

### Data preparation

The workflow includes:

- Loading and inspecting the source data.
- Converting priority values into numeric form.
- Handling missing values in selected features.
- Processing incident opening timestamps.
- Creating calendar-based features.
- Preparing separate datasets for forecasting and classification.
- Preserving chronological order for time-based evaluation.

The forecasting workflow uses a prepared daily dataset, while the classification workflow uses records with known priority labels.

## 5. Component 1: Daily Ticket Volume Forecasting

### 5.1 Objective

Forecast daily IT ticket volume to support workload estimation and IT support resource planning.

### 5.2 Forecasting dataset

The prepared dataset, `data/daily_forecasting_data.csv`, contains **182 daily observations** from October 1, 2013, through March 31, 2014.

Initial validation identified:

- 182 daily observations.
- No missing values in the selected forecasting period.
- No duplicate dates.
- No missing dates in the selected period.
- Two days with zero recorded tickets.

| Statistic | Result |
|---|---:|
| Average daily ticket count | 249.76 |
| Median daily ticket count | 317 |
| Maximum daily ticket count | 569 |

These statistics describe the selected historical period and should not be interpreted as current operational demand.

### 5.3 Feature engineering

The forecasting workflow uses the following features:

| Feature | Description |
|---|---|
| `Day_Of_Week` | Day of the week |
| `Is_Weekend` | Indicates a weekend |
| `Lag_1` | Ticket count from the previous day |
| `Lag_7` | Ticket count from seven days earlier |
| `Rolling_Mean_7` | Seven-day rolling average of historical ticket counts |

Lagged and rolling features provide historical context for predicting daily demand. They must be constructed using only information available at the forecast origin.

### 5.4 Forecasting models

Three forecasting methods were compared.

**Previous-Week Baseline**

Uses the ticket count from seven days earlier as the prediction for the corresponding day.

**Weekday Median**

Predicts the median ticket count for the relevant weekday, calculated from the training data.

**Random Forest Regressor**

Uses calendar features and lag-based observations to estimate ticket volume. The configured model uses 300 trees, a maximum depth of 8, a minimum of 3 samples per leaf, and random seed 42.

### 5.5 Evaluation methodology

The dataset was split chronologically, without random shuffling.

After preparing lag-based features, the evaluation dataset contained 175 usable observations:

- Training observations: 140
- Testing observations: 35
- Test period: February 25, 2014, to March 31, 2014

The models were evaluated using:

- **Mean Absolute Error (MAE):** Average absolute difference between actual and predicted ticket counts.
- **Root Mean Squared Error (RMSE):** Error measure that penalizes larger mistakes more strongly.
- **R² score:** Additional measure of how well the regression model explains variation in the test observations.

Lower MAE and RMSE indicate better performance.

### 5.6 Forecast model comparison

| Forecasting method | MAE | RMSE |
|---|---:|---:|
| Previous-week baseline | 24.77 | 35.61 |
| Random Forest Regressor | 31.61 | 41.72 |
| Weekday median | 42.83 | 61.09 |

The previous-week baseline achieved the lowest MAE and RMSE among the three tested approaches.

The Random Forest Regressor's MAE was approximately 27.6% higher than the previous-week baseline's MAE.

These findings demonstrate that a simple baseline can outperform a more complex model when historical weekly patterns are informative.

### 5.7 Forecasting findings

The historical data showed differences between weekday and weekend ticket volumes. The Random Forest feature-importance analysis assigned the following importance values:

| Feature | Importance |
|---|---:|
| `Lag_7` | 0.3524 |
| `Is_Weekend` | 0.3110 |
| `Day_Of_Week` | 0.2786 |

`Rolling_Mean_7` and `Lag_1` contributed less in the fitted model.

Feature importance describes how the fitted model used its inputs; it does not establish causal relationships.

### 5.8 Demonstration forecast

The forecasting script generated the following historical demonstration:

| Field | Result |
|---|---|
| Latest available date | March 31, 2014 |
| Observed ticket count | 217 |
| Forecast date | April 7, 2014 |
| Predicted ticket count | 217 |
| Forecasting method | Previous-week baseline |

The forecast was saved to `reports/next_week_volume_forecast.csv`.

**Important:** The script named `predict_next_day_volume.py` forecasts seven days after the latest available observation. The generated April 7, 2014, estimate is a historical demonstration, not a forecast for the present day.

## 6. Component 2: High-Priority Incident Classification

### 6.1 Objective

Classify ITSM incidents into two categories:

- **High priority:** Original priority values 1 or 2.
- **Other priority:** Original priority values 3, 4, or 5.

The model is intended to help identify incidents that may deserve additional review. It should not replace established incident-management rules or human triage.

### 6.2 Classification dataset

The prepared dataset, `data/priority_training_data.csv`, contains **45,226 incidents with known priority labels**.

| Category | Records |
|---|---:|
| High-priority incidents | 700 |
| Other-priority incidents | 44,526 |
| Total labelled incidents | 45,226 |

High-priority incidents represent approximately 1.55% of labelled records, making this an imbalanced classification problem.

### 6.3 Features

The model excludes `Impact` and `Urgency` because diagnostic analysis found that their combinations almost completely determined the original priority labels in this dataset. Including them produced suspiciously perfect classification results and could simply reproduce the existing priority assignment rules.

The evaluated model uses:

- `CI_Cat`
- `CI_Subcat`
- `Category`
- `Open_Weekday`
- `Open_Hour`

Categorical variables are imputed where necessary and encoded using one-hot encoding. A class-weighted Logistic Regression classifier is used to help address the imbalance between high-priority and other-priority incidents.

### 6.4 Chronological evaluation

The labelled records were sorted by opening timestamp and divided chronologically.

| Evaluation detail | Result |
|---|---|
| Training records | 36,180 |
| Testing records | 9,046 |
| Training period | February 5, 2012 – February 18, 2014 |
| Testing period | February 18, 2014 – March 31, 2014 |
| High-priority training records | 591 |
| High-priority test records | 109 |

This evaluation uses earlier records for training and later records for testing. It is more representative of future prediction than a random split, although it remains limited to the available historical period.

### 6.5 Classification results

At the default decision threshold of 0.50, the chronological test produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | Approximately 95% |
| High-priority precision | 15.23% |
| High-priority recall | 61.47% |
| High-priority F1-score | 0.2441 |
| Average precision | 0.2406 |

The confusion matrix was:

| Actual / Predicted | Other priority | High priority |
|---|---:|---:|
| Other priority | 8,564 | 373 |
| High priority | 42 | 67 |

The model identified 67 of the 109 high-priority incidents and missed 42. It also incorrectly flagged 373 other-priority incidents.

Although overall accuracy is high, the minority-class precision and F1-score reveal that the classifier still has important limitations. Accuracy alone is not sufficient to assess this model.

### 6.6 Decision-threshold analysis

The classifier's default decision threshold was compared with alternative thresholds using the saved chronological test predictions.

| Threshold | Precision | Recall | F1-score | False positives |
|---:|---:|---:|---:|---:|
| 0.50 | 15.2% | 61.5% | 0.244 | 373 |
| 0.60 | 19.5% | 58.7% | 0.292 | 265 |
| 0.70 | 22.9% | 53.2% | 0.320 | 195 |
| 0.80 | 30.0% | 52.3% | 0.381 | 133 |
| 0.90 | 40.4% | 50.5% | 0.449 | 81 |

Increasing the threshold reduced false positives but also reduced recall.

The 0.80 threshold is a candidate for further investigation because it reduces false alarms while retaining some ability to detect high-priority incidents. The 0.90 threshold achieved the highest F1-score among the thresholds tested.

**Evaluation caveat:** These threshold results were calculated on the same chronological test set used for the original evaluation. They are exploratory, not an independent final performance estimate. A separate chronological validation period should be used to choose the threshold before reporting final test performance.

## 7. Project Results

The completed experiments demonstrate the following:

1. The forecasting workflow successfully prepared daily ticket data and evaluated three forecasting methods.
2. The previous-week baseline outperformed the Random Forest Regressor and weekday-median method on the selected forecasting test period.
3. The classification workflow identified some high-priority incidents but generated a substantial number of false alarms.
4. Removing `Impact` and `Urgency` avoided relying on features that almost directly encoded the original priority labels.
5. Threshold analysis demonstrated the trade-off between precision and recall.
6. CSV reports and a forecasting visualization were generated for reproducibility and further analysis.

These are results from historical experiments and should not be interpreted as evidence of production readiness.

## 8. Project Structure

```text
ABC-Tech-ITSM-AI/
│
├── data/
│   ├── itsm_dataset.csv
│   ├── daily_forecasting_data.csv
│   └── priority_training_data.csv
│
├── src/
│   ├── database.py
│   ├── export_dataset.py
│   ├── prepare_priority_data.py
│   ├── check_priority_rules.py
│   ├── check_priority_timeline.py
│   ├── train_priority_model.py
│   ├── evaluate_priority_chronologically.py
│   ├── tune_priority_threshold.py
│   ├── train_volume_forecast.py
│   ├── compare_forecast_models.py
│   └── predict_next_day_volume.py
│
├── reports/
│   ├── figures/
│   │   └── volume_forecast_test.png
│   ├── volume_forecast_test_results.csv
│   ├── forecast_model_comparison.csv
│   ├── next_week_volume_forecast.csv
│   ├── priority_model_comparison.csv
│   ├── priority_chronological_predictions.csv
│   └── priority_threshold_comparison.csv
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

This is a representative project structure based on the implemented workflow. Keep only files that actually exist in your repository, and update this tree as additional components are developed.

## 9. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies.
- PyCharm or another Python IDE.
- Git, if version control is required.
- Access to the approved ITSM dataset or database export.

### Step 1: Clone the repository

If the project is hosted on GitHub, clone your repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ABC-Tech-ITSM-AI
```

Replace the placeholder with your actual repository URL.

Alternatively, open your existing project folder in PyCharm.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

Install the packages listed in your project's `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

Ensure the requirements file contains the packages actually used by your scripts, including pandas and the relevant scikit-learn dependencies.

### Step 4: Configure environment variables

If database access is required, create a local `.env` file containing the required database settings.

Do not commit `.env`, passwords, access tokens, or other credentials to GitHub. Keep the original database read-only unless you have explicit authorization to modify it.

## 10. Running the Project

Run commands from the project root while the virtual environment is active.

### A. Prepare the priority dataset

```bash
python src/prepare_priority_data.py
```

### B. Inspect priority rules and historical patterns

```bash
python src/check_priority_rules.py
python src/check_priority_timeline.py
```

### C. Train and compare the priority classifier

```bash
python src/train_priority_model.py
```

### D. Evaluate the classifier chronologically

```bash
python src/evaluate_priority_chronologically.py
```

### E. Compare decision thresholds

```bash
python src/tune_priority_threshold.py
```

Threshold selection should subsequently be repeated using a separate validation period.

### F. Train and evaluate forecasting models

```bash
python src/train_volume_forecast.py
python src/compare_forecast_models.py
```

### G. Generate the demonstration forecast

```bash
python src/predict_next_day_volume.py
```

Check the `reports/` directory for generated CSV files and visualizations.

## 11. Technologies Used

- **Python** — primary programming language.
- **Pandas** — data loading, cleaning, transformation, and analysis.
- **Scikit-learn** — classification, regression, preprocessing, and evaluation.
- **NumPy** — numerical operations where required.
- **Matplotlib** — visualization of forecasting results.
- **MySQL** — source-data access where configured.
- **PyCharm** — development environment.
- **Git and GitHub** — version control and project hosting.

## 12. Limitations

### Historical data limitations

The dataset covers historical records from 2012–2014. It may not represent present-day ticket patterns, service volumes, priority policies, or operational processes.

The data also exhibits a sharp increase in ticket volume around September–October 2013. The cause has not been established and should be investigated before assuming that the entire period represents a consistent operational process.

### Forecasting limitations

The forecasting test set contains only 35 observations. The previous-week baseline performed best among the methods evaluated, but its future performance has not been established.

### Classification limitations

High-priority incidents account for a small proportion of labelled data. The classifier has limited precision, and its threshold analysis has not yet been independently validated.

Historical labels may reflect existing operational rules rather than independently verified urgency. Predictions should therefore be reviewed by qualified support personnel.

### Deployment limitations

The current project is an experimental machine learning workflow. It has not yet demonstrated reliable performance on current data or production integration with an ITSM platform.

## 13. Future Improvements

- Investigate the historical ticket-volume transition and validate zero-ticket days.
- Evaluate forecasting models with rolling-origin time-series validation.
- Test additional forecasting approaches where justified by the data.
- Validate model performance on newer ITSM records.
- Select the priority decision threshold using a separate chronological validation period.
- Evaluate precision–recall curves and class-specific performance.
- Examine model calibration and operational false-alarm costs.
- Save the complete classification preprocessing and model pipeline.
- Add automated tests and reproducible experiment configuration.
- Develop a Streamlit dashboard for forecasts, ticket trends, and classification results.
- Integrate predictions with ITSM workflows only after suitable validation and authorization.

## 14. Conclusion

The ABC Tech ITSM AI project has established a reproducible foundation for two machine learning tasks: daily ticket volume forecasting and high-priority incident classification.

For forecasting, the previous-week baseline achieved the lowest error among the three tested methods. For classification, chronological evaluation showed that the model could detect a portion of high-priority incidents, but false positives and class imbalance remain significant challenges.

The project demonstrates the importance of chronological evaluation, comparison with simple baselines, feature-leakage checks, and transparent reporting of limitations.

Further validation on separate time periods and more recent data is necessary before these models can be considered for operational use.
=======
# ABC-Tech-ITSM
>>>>>>> origin/main
