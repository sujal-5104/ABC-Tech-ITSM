# ABC-Tech-ITSM-AI
### AI-Powered IT Service Management Analytics and Incident Priority Prediction

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-blueviolet?style=for-the-badge)](https://abc-tech-itsm.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?style=flat-square&logo=scikitlearn)](https://scikit-learn.org/)

**Live application:** https://abc-tech-itsm.onrender.com  
**GitHub repository:** https://github.com/sujal-5104/ABC-Tech-ITSM

## Overview

ABC-Tech-ITSM-AI is a machine-learning and data analytics project designed to explore IT Service Management (ITSM) incident data. It combines an interactive dashboard, incident priority classification, and historical incident-volume forecasting experiments in a Python application.

The project demonstrates data preprocessing, exploratory data analysis, time-aware model evaluation, model serialization, and deployment using Streamlit and Render.

> **Important:** The models use historical data from 2012–2014. Their outputs are experimental and should not be treated as current operational forecasts or confirmed incident priorities.

## Features

- **Incident Dashboard:** View incident totals, priority distribution, top incident categories, and monthly incident trends.
- **Priority Prediction:** Estimate the probability that an incident belongs to the high-priority group (P1/P2).
- **Interactive Inputs:** Select configuration-item category, subcategory, incident category, weekday, and opening hour.
- **Historical Volume Forecasting:** Compare forecasting methods using chronological evaluation and a previous-week baseline.
- **Model Evaluation:** Review precision, recall, F1-score, confusion matrices, and average precision.
- **Cloud Deployment:** Run the Streamlit application through Render.

## Technology Stack

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Dashboard and visualization:** Streamlit, Plotly
- **Model persistence:** Joblib
- **Version control:** Git and GitHub
- **Deployment:** Render

## Project Architecture

```text
ABC-Tech-ITSM-AI/
├── app.py
├── main.py
├── requirements-render.txt
├── requirements.txt
├── .python-version
├── .gitignore
├── README.md
├── data/
│   ├── itsm_dataset.csv
│   ├── daily_forecasting_data.csv
│   └── priority_training_data.csv
├── models/
│   └── priority_classifier.joblib
├── reports/
│   ├── figures/
│   ├── forecast_model_comparison.csv
│   ├── priority_final_test_predictions.csv
│   ├── priority_threshold_comparison.csv
│   ├── priority_validation_thresholds.csv
│   └── volume_forecast_test_results.csv
└── src/
    ├── prepare_priority_data.py
    ├── train_priority_model.py
    ├── evaluate_priority_chronologically.py
    ├── validate_priority_threshold.py
    ├── evaluate_final_priority_model.py
    ├── save_priority_pipeline.py
    ├── compare_forecast_models.py
    └── predict_next_day_volume.py
```

*The tree lists key project files; exact report files may vary. Never commit `.env`, credentials, or files you are not authorized to publish.*

## Machine Learning

### Incident Priority Classification

The classifier estimates whether an incident belongs to the high-priority class:

- **High Priority:** P1/P2
- **Other Priority:** P3/P4/P5

Input features include configuration-item category, subcategory, incident category, opening weekday, and opening hour. Existing Impact and Urgency fields are excluded from the model's predictive inputs to avoid simply reproducing the source priority rules.

The final evaluation used a chronological holdout set with a probability threshold selected on a separate validation period.

| Metric | Final test result |
|---|---:|
| Accuracy | 98.51% |
| High-priority precision | 40.44% |
| High-priority recall | 50.46% |
| High-priority F1-score | 44.90% |
| Average precision | 0.2406 |

The dataset is imbalanced, so accuracy alone is not sufficient. On the final test set, the model identified 55 of 109 high-priority incidents and missed 54. Review the full precision, recall, false-positive, and false-negative trade-offs before using such a model operationally.

### Incident-Volume Forecasting

Several forecasting approaches were compared on a chronological test period:

| Method | MAE | RMSE |
|---|---:|---:|
| Previous-week baseline | 24.77 | 35.61 |
| Weekday median baseline | 42.83 | 61.09 |
| Random Forest Regressor | 31.61 | 41.72 |

The previous-week baseline performed best among these tested methods. This is a useful reminder that a simpler baseline can outperform a more complex model.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/sujal-5104/ABC-Tech-ITSM.git
cd ABC-Tech-ITSM
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements-render.txt
```

### 4. Ensure required files are present

Place the authorized ITSM dataset at `data/itsm_dataset.csv` and confirm the saved model exists at `models/priority_classifier.joblib`.

### 5. Start the dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Deploy on Render

1. Push the project to GitHub.
2. In the Render dashboard, create a **Web Service** and connect this repository.
3. Set the branch to `main` and leave the root directory blank.
4. Use the following commands:

**Build command**

```bash
pip install -r requirements-render.txt
```

**Start command**

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

5. Set the Python version using the repository's `.python-version` file.
6. Deploy and inspect the logs if the build fails.
7. Confirm the model and dataset paths are available to the application.

**Data and security:** Publish the dataset only if you have permission. Do not upload `.env`, database passwords, API keys, or other credentials.

## Limitations

- The data represents a historical period (2012–2014), not current ITSM activity.
- High-priority examples are relatively rare, affecting model precision and recall.
- The probability threshold changes the trade-off between false alarms and missed high-priority incidents.
- Predictions are experimental decision support, not a replacement for organizational incident-management policies.
- Forecasting results describe the historical test period and should not be interpreted as validated current forecasts.

## Future Improvements

- Test on more recent, representative ITSM data.
- Improve minority-class detection and calibrate predicted probabilities.
- Add model monitoring, data-quality checks, and drift detection.
- Expand automated tests and deployment checks.
- Improve accessibility, dashboard documentation, and visual reporting.

## Author

**Sujal**

- GitHub: https://github.com/sujal-5104

---

## Screenshots

### Dashboard Overview
![Dashboard Overview](screenshots/dashboard-overview.png)

### Priority Prediction
![Priority Prediction](screenshots/priority-prediction.png)