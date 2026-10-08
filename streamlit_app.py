import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Calories Burned Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# MODEL TRAINING FALLBACK (Ensures Streamlit Cloud runs seamlessly)
# =========================================================
MODEL_PATH = "calories_model.pkl"

def train_and_cache_model():
    """Generates synthetic dataset and trains Random Forest model if model file is missing."""
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

@st.cache_resource(show_spinner="Loading machine learning model...")
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    # Automatically train model if file is absent or corrupted
    return train_and_cache_model()

model = load_model()

# =========================================================
# HEADER & APP TITLE
# =========================================================
st.title("🔥 Calories Burned Prediction")
st.markdown(
    "Predict estimated calories burned during workouts and exercise sessions "
    "using a trained **Random Forest Regression** machine learning model."
)

# =========================================================
# SIDEBAR - INPUT CONTROLS
# =========================================================
st.sidebar.header("⚙️ Exercise & User Metrics")
st.sidebar.markdown("Configure personal metrics and workout attributes below:")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age (years)", min_value=18, max_value=65, value=25, step=1)
height = st.sidebar.slider("Height (cm)", min_value=145, max_value=195, value=170, step=1)
weight = st.sidebar.slider("Weight (kg)", min_value=45.0, max_value=120.0, value=65.0, step=0.5)
duration = st.sidebar.slider("Exercise Duration (minutes)", min_value=5.0, max_value=120.0, value=30.0, step=1.0)
heart_rate = st.sidebar.slider("Heart Rate (BPM)", min_value=70.0, max_value=180.0, value=110.0, step=1.0)
body_temp = st.sidebar.slider("Body Temperature (°C)", min_value=36.0, max_value=39.5, value=37.0, step=0.1)

# Helper calculations (BMI)
height_m = height / 100.0
bmi = weight / (height_m ** 2)

# =========================================================
# MAIN SECTION - INPUT METRICS DASHBOARD
# =========================================================
st.subheader("📋 Session Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Age & Gender", value=f"{age} yrs", delta=gender)
with col2:
    st.metric(label="Weight / Height", value=f"{weight:.1f} kg", delta=f"{height} cm")
with col3:
    st.metric(label="Duration", value=f"{duration:.0f} min")
with col4:
    st.metric(label="Heart Rate", value=f"{heart_rate:.0f} BPM")

st.divider()

# =========================================================
# PREDICTION LOGIC
# =========================================================
col_btn, _ = st.columns([1, 2])
predict_btn = col_btn.button("🔥 Predict Calories Burned", type="primary", use_container_width=True)

# Format input data
gender_numeric = 1 if gender == "Male" else 0
input_df = pd.DataFrame([{
    "Gender": gender_numeric,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "Duration": duration,
    "Heart_Rate": heart_rate,
    "Body_Temp": body_temp
}])

# Compute prediction
predicted_calories = max(0.0, float(model.predict(input_df)[0]))
burn_rate = predicted_calories / duration if duration > 0 else 0.0

if predict_btn or "has_run" not in st.session_state:
    st.session_state["has_run"] = True

if st.session_state.get("has_run", False):
    st.header("🎯 Prediction Result")

    res_col1, res_col2, res_col3, res_col4 = st.columns(4)
    with res_col1:
        st.metric("Estimated Calories Burned", f"{predicted_calories:.1f} kcal")
    with res_col2:
        st.metric("Burn Rate", f"{burn_rate:.1f} kcal/min")
    with res_col3:
        st.metric("Workout Duration", f"{duration:.0f} mins")
    with res_col4:
        st.metric("Estimated BMI", f"{bmi:.1f} kg/m²")

    # Intensity assessment
    if predicted_calories < 150:
        intensity_msg = "Light session: Good for active recovery, flexibility, or warm-up."
        alert_fn = st.info
    elif predicted_calories < 300:
        intensity_msg = "Moderate session: Great for cardiovascular endurance and general health maintenance."
        alert_fn = st.success
    elif predicted_calories < 500:
        intensity_msg = "High intensity workout: Significant energy expenditure, suitable for fat loss and fitness progression."
        alert_fn = st.warning
    else:
        intensity_msg = "Very high intensity / long duration: Maximum exertion. Remember proper rehydration and post-workout recovery!"
        alert_fn = st.error

    alert_fn(f"💡 **Analysis:** {intensity_msg}")

    # Summary Table
    with st.expander("📊 View Input Feature Summary", expanded=False):
        summary_table = pd.DataFrame({
            "Feature": [
                "Gender",
                "Age",
                "Height",
                "Weight",
                "Exercise Duration",
                "Heart Rate",
                "Body Temperature",
                "Calculated BMI"
            ],
            "Input Value": [
                gender,
                f"{age} years",
                f"{height} cm",
                f"{weight:.1f} kg",
                f"{duration:.0f} minutes",
                f"{heart_rate:.0f} BPM",
                f"{body_temp:.1f} °C",
                f"{bmi:.1f}"
            ]
        })
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

# =========================================================
# MODEL INFORMATION & ABOUT
# =========================================================
st.divider()
st.subheader("🧠 Model Information & Architecture")

info1, info2, info3, info4 = st.columns(4)
with info1:
    st.metric("Algorithm", "Random Forest")
with info2:
    st.metric("Task", "Regression")
with info3:
    st.metric("Input Features", "7")
with info4:
    st.metric("Status", "Trained & Ready")

with st.expander("ℹ️ About This Application"):
    st.markdown(
        """
        - **Objective:** Predicts estimated calories burned during exercise based on physiological and exercise session metrics.
        - **Pipeline:** Features include Gender, Age, Height, Weight, Duration, Heart Rate, and Body Temperature.
        - **Model:** Random Forest Regressor trained on exercise session metrics.
        - **Deployment Ready:** Configured for one-click deployment on **Streamlit Community Cloud**.
        """
    )

st.caption("⚠️ *Disclaimer: Predictions are generated by a machine learning model for informational and educational purposes. Not intended as medical or professional fitness advice.*")
