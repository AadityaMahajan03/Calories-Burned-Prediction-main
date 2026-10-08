import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# CREATE TRAINING DATA
# =========================================================

np.random.seed(42)

N = 5000

gender = np.random.randint(0, 2, N)
age = np.random.randint(18, 65, N)
height = np.random.randint(145, 195, N)
weight = np.random.uniform(45, 120, N)

duration = np.random.uniform(5, 120, N)
heart_rate = np.random.uniform(70, 180, N)
body_temp = np.random.uniform(36.0, 39.5, N)


# =========================================================
# CALORIE FORMULA
# =========================================================

calories = (
    duration * 5.2
    + heart_rate * 1.8
    + weight * 2.1
    + age * 0.4
    + (body_temp - 36) * 15
    + gender * 8
)


# Add realistic variation

noise = np.random.normal(
    0,
    25,
    N
)

calories = calories + noise

calories = np.maximum(
    calories,
    20
)


# =========================================================
# DATAFRAME
# =========================================================

data = pd.DataFrame({

    "Gender": gender,

    "Age": age,

    "Height": height,

    "Weight": weight,

    "Duration": duration,

    "Heart_Rate": heart_rate,

    "Body_Temp": body_temp,

    "Calories": calories
})


print(
    "Dataset created successfully."
)

print(
    f"Total samples: {len(data)}"
)


# =========================================================
# FEATURES
# =========================================================

X = data[
    [
        "Gender",
        "Age",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp"
    ]
]

y = data[
    "Calories"
]


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# =========================================================
# MODEL
# =========================================================

model = RandomForestRegressor(

    n_estimators=150,

    max_depth=15,

    random_state=42,

    n_jobs=-1
)


print(
    "\nTraining Random Forest..."
)


model.fit(
    X_train,
    y_train
)


# =========================================================
# EVALUATION
# =========================================================

predictions = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print(
    "\n=============================="
)

print(
    "MODEL PERFORMANCE"
)

print(
    "=============================="
)

print(
    f"MAE: {mae:.2f} calories"
)

print(
    f"R² Score: {r2:.4f}"
)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "calories_model.pkl"
)


print(
    "\nModel saved as calories_model.pkl"
)

print(
    "Training completed successfully! 🎉"
)