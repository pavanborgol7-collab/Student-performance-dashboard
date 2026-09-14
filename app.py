from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance", layout="wide")
st.title("Student Performance Dashboard")

def first_existing(*paths):
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return Path(paths[0])


MODEL_PATH = first_existing(
    "models/pass_fail_pipeline.joblib",
    "pass_fail_pipeline.joblib",
)
DATA_PATH = first_existing(
    "DATA/Student_performance_10k.csv",
    "Student_performance_10k.csv",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_students():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["grade"])
    df["result"] = df["grade"].isin(["A", "B", "C"]).map({True: "Pass", False: "Fail"})
    return df


if not MODEL_PATH.exists():
    st.error(f"Model file not found at `{MODEL_PATH}`.")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Data file not found at `{DATA_PATH}`.")
    st.stop()

model = load_model()
df = load_students()

st.success("Model and student data loaded.")

pass_count = (df["result"] == "Pass").sum()
fail_count = (df["result"] == "Fail").sum()

col1, col2, col3 = st.columns(3)
col1.metric("Total students", len(df))
col2.metric("Pass", int(pass_count))
col3.metric("Fail", int(fail_count))

st.subheader("Student records")
st.dataframe(df.head(50), use_container_width=True)

st.subheader("Predict Pass or Fail")
st.write("Enter one student's details, then click Predict.")

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["female", "male"])
        race = st.selectbox(
            "Race / ethnicity",
            ["group A", "group B", "group C", "group D", "group E"],
        )
        education = st.selectbox(
            "Parental education",
            [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree",
            ],
        )
        lunch = st.selectbox("Lunch (1 = standard, 0 = free/reduced)", [1.0, 0.0])
        test_prep = st.selectbox("Test preparation (1 = completed, 0 = none)", [1.0, 0.0])
    with c2:
        math_score = st.number_input("Math score", min_value=0.0, max_value=100.0, value=65.0)
        reading_score = st.number_input("Reading score", min_value=0.0, max_value=100.0, value=70.0)
        writing_score = st.number_input("Writing score", min_value=0.0, max_value=100.0, value=68.0)
        science_score = st.number_input("Science score", min_value=0.0, max_value=100.0, value=72.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    new_student = pd.DataFrame(
        [
            {
                "gender": gender,
                "race_ethnicity": race,
                "parental_level_of_education": education,
                "lunch": lunch,
                "test_preparation_course": test_prep,
                "math_score": math_score,
                "reading_score": reading_score,
                "writing_score": writing_score,
                "science_score": science_score,
            }
        ]
    )
    prediction = model.predict(new_student)[0]
    st.markdown(f"### Result: **{prediction}**")
