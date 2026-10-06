import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from src.features import (
    CATEGORICAL,
    NUMERIC,
    make_ml_matrices
)


# ============================================================
# 2. FILE PATHS
# ============================================================

DATA = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ridely_processed.csv"
)

MODEL_OUT = os.path.join(
    PROJECT_ROOT,
    "models",
    "ml_models.joblib"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs"
)

os.makedirs(
    os.path.dirname(MODEL_OUT),
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("RIDELY AI - ML MODEL TRAINING")
print("=" * 70)

print("\nLoading processed dataset...")

if not os.path.exists(DATA):

    raise FileNotFoundError(
        f"\nProcessed dataset not found:\n{DATA}\n\n"
        "Run preprocessing.py first."
    )

df = pd.read_csv(DATA)

print("\nDataset loaded successfully!")

print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# 4. CREATE DATETIME FOR CHRONOLOGICAL SPLIT
# ============================================================

df["Datetime"] = pd.to_datetime(
    df["Date"].astype(str)
    + " "
    + df["Time"].astype(str),
    errors="coerce"
)

df = df.sort_values(
    "Datetime"
).reset_index(drop=True)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

split = int(
    len(df) * 0.80
)

train_df = df.iloc[
    :split
].copy()

test_df = df.iloc[
    split:
].copy()

print("\nData Split")
print("-" * 40)

print(
    "Training samples:",
    len(train_df)
)

print(
    "Testing samples :",
    len(test_df)
)

print(
    "\nUsing chronological 80/20 split."
)


# ============================================================
# 6. DEFINE THREE ML MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        MultiOutputRegressor(
            XGBRegressor(
                n_estimators=350,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=4
            )
        )
}


# ============================================================
# 7. STORAGE
# ============================================================

saved = {}

metric_rows = []


# ============================================================
# 8. TRAIN MODELS
# ============================================================

platforms = [
    "Uber",
    "Ola",
    "Rapido"
]


for platform in platforms:

    print("\n")
    print("=" * 70)
    print(
        f"TRAINING MODELS FOR {platform.upper()}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Create X and y
    # --------------------------------------------------------

    X_train, y_train, feature_cols = make_ml_matrices(
        train_df,
        platform
    )

    X_test, y_test, _ = make_ml_matrices(
        test_df,
        platform
    )


    # --------------------------------------------------------
    # Display target columns
    # --------------------------------------------------------

    print("\nTarget columns:")

    for target in y_train.columns:
        print(
            " -",
            target
        )


    # --------------------------------------------------------
    # Separate categorical and numerical features
    # --------------------------------------------------------

    categorical_features = [
        feature
        for feature in CATEGORICAL
        if feature in feature_cols
    ]

    numerical_features = [
        feature
        for feature in feature_cols
        if feature not in categorical_features
    ]


    # --------------------------------------------------------
    # Preprocessing
    #
    # Categorical:
    # One-Hot Encoding
    #
    # Numerical:
    # Standard Scaling
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "cat",

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),

                categorical_features
            ),

            (
                "num",

                StandardScaler(),

                numerical_features
            )
        ]
    )


    saved[platform] = {}


    # ========================================================
    # TRAIN EACH MODEL
    # ========================================================

    for model_name, estimator in models.items():

        print("\n")
        print("-" * 60)

        print(
            f"Training {model_name}..."
        )

        print("-" * 60)


        # ----------------------------------------------------
        # Create pipeline
        # ----------------------------------------------------

        pipe = Pipeline(
            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    estimator
                )
            ]
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipe.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        pred = pipe.predict(
            X_test
        )


        # ----------------------------------------------------
        # Overall Metrics
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        )

        r2 = r2_score(
            y_test,
            pred,
            multioutput="uniform_average"
        )


        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print(
            f"\n{model_name} Results:"
        )

        print(
            f"MAE  : {mae:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"R²   : {r2:.4f}"
        )


        # ----------------------------------------------------
        # Store overall metrics
        # ----------------------------------------------------

        metric_rows.append({

            "Platform":
                platform,

            "Model":
                model_name,

            "MAE":
                mae,

            "RMSE":
                rmse,

            "R2":
                r2
        })


        # ----------------------------------------------------
        # Horizon-wise metrics
        # ----------------------------------------------------

        print(
            "\nHorizon-wise Results:"
        )

        for i, target in enumerate(
            y_test.columns
        ):

            target_actual = y_test.iloc[
                :, i
            ]

            target_prediction = pred[
                :, i
            ]


            target_mae = mean_absolute_error(
                target_actual,
                target_prediction
            )


            target_rmse = np.sqrt(
                mean_squared_error(
                    target_actual,
                    target_prediction
                )
            )


            target_r2 = r2_score(
                target_actual,
                target_prediction
            )


            print(
                f"{target}: "
                f"MAE={target_mae:.3f}, "
                f"RMSE={target_rmse:.3f}, "
                f"R²={target_r2:.3f}"
            )


        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        saved[platform][
            model_name
        ] = pipe


# ============================================================
# 9. CREATE METRICS DATAFRAME
# ============================================================

metrics = pd.DataFrame(
    metric_rows
)


# ============================================================
# 10. SAVE METRICS
# ============================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "ml_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 11. SAVE ALL MODELS
# ============================================================

joblib.dump(
    saved,
    MODEL_OUT
)


# ============================================================
# 12. PRINT MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    metrics.to_string(
        index=False
    )
)


# ============================================================
# 13. FIND BEST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("BEST MODEL FOR EACH PLATFORM")
print("=" * 70)


for platform in platforms:

    platform_metrics = metrics[
        metrics["Platform"] == platform
    ]

    best_index = platform_metrics[
        "RMSE"
    ].idxmin()

    best = platform_metrics.loc[
        best_index
    ]


    print(
        f"\n{platform}"
    )

    print(
        f"Best Model : {best['Model']}"
    )

    print(
        f"MAE        : {best['MAE']:.4f}"
    )

    print(
        f"RMSE       : {best['RMSE']:.4f}"
    )

    print(
        f"R²         : {best['R2']:.4f}"
    )


# ============================================================
# 14. MODEL COMPARISON PLOT
# ============================================================

avg = (
    metrics
    .groupby("Model")["MAE"]
    .mean()
    .sort_values()
)


plt.figure(
    figsize=(9, 5)
)

avg.plot(
    kind="bar"
)

plt.ylabel(
    "MAE (₹)"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.title(
    "Ridely AI - ML Model Comparison"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()


plot_path = os.path.join(
    OUTPUT_DIR,
    "ml_comparison.png"
)

plt.savefig(
    plot_path,
    dpi=160
)

plt.close()


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nModels saved to:"
    f"\n{MODEL_OUT}"
)

print(
    f"\nMetrics saved to:"
    f"\n{metrics_path}"
)

print(
    f"\nComparison graph saved to:"
    f"\n{plot_path}"
)

print("\nRidely AI ML training finished!")
