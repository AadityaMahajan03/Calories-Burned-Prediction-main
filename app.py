import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Calories Burned Prediction",
    page_icon="🔥",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================
st.title("🔥 Calories Burned Prediction")
st.write(
    "Predict estimated calories burned during exercise "
    "using a Random Forest machine learning model."
)

# =========================================================
# LOAD MODEL WITH FALLBACK
# =========================================================
MODEL_PATH = "calories_model.pkl"

def train_and_cache_model():
    """Trains model if file is missing (e.g. fresh git clone on Streamlit Cloud)."""
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    n = 5000
    gender = np.random.randint(0, 2, n)
    age = np.random.randint(18, 65, n)
    height = np.random.randint(145, 195, n)
    weight = np.random.uniform(45, 120, n)
    duration = np.random.uniform(5, 120, n)
    heart_rate = np.random.uniform(70, 180, n)
    body_temp = np.random.uniform(36.0, 39.5, n)

    calories = (
        duration * 5.2
        + heart_rate * 1.8
        + weight * 2.1
        + age * 0.4
        + (body_temp - 36.0) * 15.0
        + gender * 8.0
        + np.random.normal(0, 25, n)
    )
    calories = np.maximum(calories, 20)

    df = pd.DataFrame({
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Calories": calories
    })

    X = df[["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]]
    y = df["Calories"]

    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X, y)

    try:
        joblib.dump(rf_model, MODEL_PATH)
    except Exception:
        pass

    return rf_model

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    return train_and_cache_model()

model = load_model()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Exercise Details")

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=65,
    value=25
)

height = st.sidebar.slider(
    "Height (cm)",
    min_value=145,
    max_value=195,
    value=170
)

weight = st.sidebar.slider(
    "Weight (kg)",
    min_value=45.0,
    max_value=120.0,
    value=65.0,
    step=0.5
)

duration = st.sidebar.slider(
    "Exercise Duration (minutes)",
    min_value=5.0,
    max_value=120.0,
    value=30.0,
    step=1.0
)

heart_rate = st.sidebar.slider(
    "Heart Rate (BPM)",
    min_value=70.0,
    max_value=180.0,
    value=110.0,
    step=1.0
)

body_temp = st.sidebar.slider(
    "Body Temperature (°C)",
    min_value=36.0,
    max_value=39.5,
    value=37.0,
    step=0.1
)

# =========================================================
# INPUT DISPLAY
# =========================================================
st.header("📋 Exercise Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Age", f"{age} years")

with col2:
    st.metric("Weight", f"{weight:.1f} kg")

with col3:
    st.metric("Duration", f"{duration:.0f} min")

with col4:
    st.metric("Heart Rate", f"{heart_rate:.0f} BPM")

# =========================================================
# PREDICTION
# =========================================================
st.divider()

if st.button("🔥 Predict Calories", type="primary", use_container_width=True):
    gender_value = 1 if gender == "Male" else 0

    input_data = pd.DataFrame({
        "Gender": [gender_value],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp]
    })

    prediction = max(0.0, float(model.predict(input_data)[0]))

    # =====================================================
    # RESULT
    # =====================================================
    st.header("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric("Estimated Calories Burned", f"{prediction:.0f} kcal")

    with result_col2:
        st.metric("Exercise Duration", f"{duration:.0f} minutes")

    # =====================================================
    # INTERPRETATION
    # =====================================================
    if prediction < 150:
        message = "This is a relatively low estimated calorie expenditure."
    elif prediction < 300:
        message = "This represents a moderate estimated calorie expenditure."
    elif prediction < 500:
        message = "This represents a relatively high estimated calorie expenditure."
    else:
        message = "This represents a very high estimated calorie expenditure."

    st.info(f"💡 {message}")

    # =====================================================
    # INPUT SUMMARY
    # =====================================================
    st.subheader("📊 Prediction Inputs")

    summary = pd.DataFrame({
        "Feature": [
            "Gender",
            "Age",
            "Height",
            "Weight",
            "Duration",
            "Heart Rate",
            "Body Temperature"
        ],
        "Value": [
            gender,
            f"{age} years",
            f"{height} cm",
            f"{weight:.1f} kg",
            f"{duration:.0f} minutes",
            f"{heart_rate:.0f} BPM",
            f"{body_temp:.1f} °C"
        ]
    })

    st.dataframe(summary, use_container_width=True, hide_index=True)

# =========================================================
# MODEL INFORMATION
# =========================================================
st.divider()

st.header("🧠 Model Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric("Algorithm", "Random Forest")

with info2:
    st.metric("Problem", "Regression")

with info3:
    st.metric("Features", "7")

# =========================================================
# ABOUT
# =========================================================
with st.expander("ℹ️ About This Project"):
    st.write(
        """
        This application predicts estimated calories burned
        during exercise using a Random Forest regression model.

        The model uses gender, age, height, weight,
        exercise duration, heart rate, and body temperature
        as input features.

        The training data is synthetically generated for
        demonstration and portfolio purposes.
        """
    )

# =========================================================
# DISCLAIMER & FOOTER
# =========================================================
st.caption(
    "⚠️ This is an educational ML prediction and should "
    "not be used as a medical or fitness measurement."
)

st.divider()

st.caption(
    "Calories Burned Prediction • "
    "Random Forest + Scikit-learn + Streamlit"
)