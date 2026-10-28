
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_ROOT / "models" / "priority_classifier.joblib"
DATA_FILE = PROJECT_ROOT / "data" / "itsm_dataset.csv"

st.set_page_config(
    page_title="ABC Tech ITSM AI",
    page_icon="🎫",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_category_options():
    df = pd.read_csv(
        DATA_FILE,
        usecols=["CI_Cat", "CI_Subcat", "Category"],
        low_memory=False,
    )

    options = {}

    for column in ["CI_Cat", "CI_Subcat", "Category"]:
        values = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )
        options[column] = sorted(values.unique().tolist())

        if "Unknown" not in options[column]:
            options[column].insert(0, "Unknown")

    return options


st.title("🎫 ABC Tech ITSM AI")
st.subheader("Incident Priority Prediction")
st.write(
    "Select incident details to estimate whether a ticket "
    "belongs to the high-priority class."
)

if not MODEL_FILE.exists():
    st.error("Model not found. Run python src/save_priority_pipeline.py first.")
    st.stop()

if not DATA_FILE.exists():
    st.error("Dataset not found at data/itsm_dataset.csv.")
    st.stop()

try:
    bundle = load_model()
    pipeline = bundle["pipeline"]
    threshold = float(bundle["threshold"])
    options = load_category_options()
except Exception as exc:
    st.error(f"Could not load the model or dataset: {exc}")
    st.stop()


with st.form("incident_form"):
    st.markdown("### Incident details")

    col1, col2 = st.columns(2)

    with col1:
        ci_category = st.selectbox(
            "Configuration item category",
            options=options["CI_Cat"],
        )

        ci_subcategory = st.selectbox(
            "Configuration item subcategory",
            options=options["CI_Subcat"],
        )

        incident_category = st.selectbox(
            "Incident category",
            options=options["Category"],
        )

    with col2:
        weekday_options = [
            ("Monday", "0"),
            ("Tuesday", "1"),
            ("Wednesday", "2"),
            ("Thursday", "3"),
            ("Friday", "4"),
            ("Saturday", "5"),
            ("Sunday", "6"),
        ]

        selected_weekday = st.selectbox(
            "Opening weekday",
            options=weekday_options,
            format_func=lambda item: item[0],
        )

        selected_hour = st.slider(
            "Opening hour",
            min_value=0,
            max_value=23,
            value=10,
        )

    submitted = st.form_submit_button(
        "Predict priority",
        type="primary",
        use_container_width=True,
    )


if submitted:
    incident = pd.DataFrame(
        [{
            "CI_Cat": ci_category,
            "CI_Subcat": ci_subcategory,
            "Category": incident_category,
            "Open_Weekday": selected_weekday[1],
            "Open_Hour": str(selected_hour),
        }],
        columns=bundle["feature_columns"],
    )

    try:
        probability = float(pipeline.predict_proba(incident)[0, 1])
        is_high_priority = probability >= threshold

        st.divider()
        st.markdown("### Prediction result")

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Estimated high-priority probability",
            f"{probability:.1%}",
        )
        metric2.metric(
            "Decision threshold",
            f"{threshold:.0%}",
        )

        st.progress(probability)

        if is_high_priority:
            st.error("⚠️ Predicted: High priority (P1/P2)")
        else:
            st.success("Predicted: Other priority (P3/P4/P5)")

        st.caption(
            "This is an experimental model estimate, not a confirmed priority. "
            "The model uses historical 2012–2014 data. Review each incident "
            "using your organization's ITSM rules."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


with st.expander("About this model"):
    st.write("**Model:** Class-weighted Logistic Regression")
    st.write(f"**Training records:** {bundle['training_records']:,}")
    st.write(f"**Decision threshold:** {threshold:.0%}")
    st.write("**High priority definition:** Original Priority 1 or 2")
    st.write(
        "Impact and Urgency are excluded from the model's input features."
    )
