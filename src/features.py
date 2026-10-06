import numpy as np
import pandas as pd


# ============================================================
# 1. FEATURE GROUPS
# ============================================================

CATEGORICAL = [
    "Pickup",
    "Drop",
    "Traffic",
    "Weather",
    "Ride_Type"
]

NUMERIC = [
    "Distance_km",
    "Duration_min",
    "Demand_Index",
    "Driver_Availability",
    "Weekend",
    "Peak_Hour",
    "Holiday",
    "Uber_Fare",
    "Ola_Fare",
    "Rapido_Fare"
]


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Convert Date + Time into datetime
    # --------------------------------------------------------
    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce"
    )

    # --------------------------------------------------------
    # Time-based features
    # --------------------------------------------------------
    df["Hour"] = dt.dt.hour
    df["Minute"] = dt.dt.minute
    df["Day_of_Week"] = dt.dt.dayofweek

    # --------------------------------------------------------
    # Demand / Supply ratio
    # --------------------------------------------------------
    df["Demand_Supply_Ratio"] = (
        df["Demand_Index"]
        / df["Driver_Availability"].clip(lower=1)
    )

    # --------------------------------------------------------
    # Cyclical representation of Hour
    #
    # This helps the model understand that:
    # 23:00 and 00:00 are close to each other.
    # --------------------------------------------------------
    df["Hour_sin"] = np.sin(
        2 * np.pi * df["Hour"] / 24
    )

    df["Hour_cos"] = np.cos(
        2 * np.pi * df["Hour"] / 24
    )

    # --------------------------------------------------------
    # Cyclical representation of Day of Week
    # --------------------------------------------------------
    df["Dow_sin"] = np.sin(
        2 * np.pi * df["Day_of_Week"] / 7
    )

    df["Dow_cos"] = np.cos(
        2 * np.pi * df["Day_of_Week"] / 7
    )

    return df


# ============================================================
# 3. CREATE ML INPUT AND TARGET MATRICES
# ============================================================

def make_ml_matrices(
    df: pd.DataFrame,
    platform: str
):

    # Add engineered features
    df = add_features(df)

    # --------------------------------------------------------
    # Future fare prediction targets
    # --------------------------------------------------------
    target_cols = [
        f"{platform}_Future_15",
        f"{platform}_Future_30",
        f"{platform}_Future_45",
        f"{platform}_Future_60",
    ]

    # --------------------------------------------------------
    # Features used by ML models
    # --------------------------------------------------------
    feature_cols = CATEGORICAL + NUMERIC + [
        "Hour",
        "Minute",
        "Day_of_Week",
        "Demand_Supply_Ratio",
        "Hour_sin",
        "Hour_cos",
        "Dow_sin",
        "Dow_cos"
    ]

    # Keep only columns that actually exist
    feature_cols = [
        col for col in feature_cols
        if col in df.columns
    ]

    # Input features
    X = df[feature_cols].copy()

    # Prediction targets
    y = df[target_cols].copy()

    return X, y, feature_cols