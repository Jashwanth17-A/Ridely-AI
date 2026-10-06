from pathlib import Path
import sys

import pandas as pd
import joblib


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORT PROJECT MODULE
# ============================================================

from src.features import add_features
# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "ml_models.joblib"


# ============================================================
# LOAD MODELS
# ============================================================

saved_models = joblib.load(
    MODEL_PATH
)


# ============================================================
# FEATURE CREATION
# ============================================================

def create_prediction_features(
    input_df
):

    df = add_features(
        input_df
    )

    feature_cols = [
        "Pickup",
        "Drop",
        "Traffic",
        "Weather",
        "Ride_Type",

        "Distance_km",
        "Duration_min",
        "Demand_Index",
        "Driver_Availability",

        "Weekend",
        "Peak_Hour",
        "Holiday",

        "Uber_Fare",
        "Ola_Fare",
        "Rapido_Fare",

        "Hour",
        "Minute",
        "Day_of_Week",

        "Demand_Supply_Ratio",

        "Hour_sin",
        "Hour_cos",
        "Dow_sin",
        "Dow_cos"
    ]

    return df[feature_cols]


# ============================================================
# PREDICTION
# ============================================================

def predict_fares(input_data):

    if isinstance(input_data, dict):

        input_df = pd.DataFrame(
            [input_data]
        )

    else:

        input_df = input_data.copy()

    X = create_prediction_features(
        input_df
    )

    predictions = {}

    for platform in [
        "Uber",
        "Ola",
        "Rapido"
    ]:

        # Linear Regression is currently
        # the best-performing model
        model = saved_models[
            platform
        ]["Linear Regression"]

        predicted = model.predict(X)

        values = predicted[0]

        values = [
            max(0, float(value))
            for value in values
        ]

        predictions[platform] = values

    return predictions


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_input = {

        "Pickup": "Gachibowli",
        "Drop": "Hitech City",

        "Distance_km": 12,
        "Duration_min": 35,

        "Traffic": "High",
        "Weather": "Sunny",

        "Ride_Type": "Sedan",

        "Demand_Index": 8,
        "Driver_Availability": 4,

        "Weekend": 0,
        "Peak_Hour": 1,
        "Holiday": 0,

        "Uber_Fare": 280,
        "Ola_Fare": 260,
        "Rapido_Fare": 230,

        "Date": "2026-10-06",
        "Time": "19:00"
    }

    result = predict_fares(
        test_input
    )

    print("\nFuture Fare Predictions")
    print("-" * 50)

    for platform, values in result.items():

        print(f"\n{platform}")

        print(
            f"+15 min : ₹{values[0]:.2f}"
        )

        print(
            f"+30 min : ₹{values[1]:.2f}"
        )

        print(
            f"+45 min : ₹{values[2]:.2f}"
        )

        print(
            f"+60 min : ₹{values[3]:.2f}"
        )