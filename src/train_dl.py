import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(BASE_DIR)

from src.features import (
    CATEGORICAL,
    NUMERIC,
    add_features
)


DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ridely_processed.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

PLATFORMS = [
    "Uber",
    "Ola",
    "Rapido"
]

SEQ_LEN = 8
HORIZON = 4

EPOCHS = 35
BATCH_SIZE = 64

RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("RIDELY AI - MULTIVARIATE RNN & LSTM TRAINING")
print("=" * 70)

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"\nProcessed dataset not found:\n{DATA_PATH}\n\n"
        "Run preprocessing.py first."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("Rows    :", len(df))
print("Columns :", len(df.columns))


# ============================================================
# 4. SORT DATA CHRONOLOGICALLY
# ============================================================

df["Datetime"] = pd.to_datetime(
    df["Date"].astype(str)
    + " "
    + df["Time"].astype(str),
    errors="coerce"
)

df = df.dropna(
    subset=["Datetime"]
)

df = df.sort_values(
    "Datetime"
).reset_index(drop=True)


# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================

df = add_features(df)


# ============================================================
# 6. FEATURE LIST
# ============================================================

ENGINEERED_NUMERIC = [
    "Hour",
    "Minute",
    "Day_of_Week",
    "Demand_Supply_Ratio",
    "Hour_sin",
    "Hour_cos",
    "Dow_sin",
    "Dow_cos"
]

FEATURE_COLUMNS = (
    CATEGORICAL
    + NUMERIC
    + ENGINEERED_NUMERIC
)


# Keep only columns that exist
FEATURE_COLUMNS = [
    col
    for col in FEATURE_COLUMNS
    if col in df.columns
]


print("\nFeatures used by RNN/LSTM:")

for feature in FEATURE_COLUMNS:
    print(" -", feature)


# ============================================================
# 7. CHECK REQUIRED TARGET COLUMNS
# ============================================================

for platform in PLATFORMS:

    targets = [
        f"{platform}_Future_15",
        f"{platform}_Future_30",
        f"{platform}_Future_45",
        f"{platform}_Future_60"
    ]

    for target in targets:

        if target not in df.columns:

            raise ValueError(
                f"Missing target column: {target}"
            )


# ============================================================
# 8. CREATE SEQUENCES
# ============================================================

def create_sequences(
    features,
    targets,
    seq_len
):
    """
    Create time-series sequences.

    For every prediction point:

    Previous 8 observations
            ↓
    [t-7, t-6, ..., t-1, t]
            ↓
    Predict:
    [15 min, 30 min, 45 min, 60 min]
    """

    X = []
    y = []

    for i in range(
        seq_len - 1,
        len(features)
    ):

        X.append(
            features[
                i - seq_len + 1:i + 1
            ]
        )

        y.append(
            targets[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# 9. BUILD RNN
# ============================================================

def build_rnn(
    input_shape
):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=input_shape
        ),

        tf.keras.layers.SimpleRNN(
            64,
            activation="tanh",
            return_sequences=True
        ),

        tf.keras.layers.Dropout(
            0.2
        ),

        tf.keras.layers.SimpleRNN(
            32,
            activation="tanh"
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            HORIZON
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# 10. BUILD LSTM
# ============================================================

def build_lstm(
    input_shape
):

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=input_shape
        ),

        tf.keras.layers.LSTM(
            64,
            activation="tanh",
            return_sequences=True
        ),

        tf.keras.layers.Dropout(
            0.2
        ),

        tf.keras.layers.LSTM(
            32,
            activation="tanh"
        ),

        tf.keras.layers.Dropout(
            0.2
        ),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            HORIZON
        )
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model


# ============================================================
# 11. STORAGE
# ============================================================

results = []

horizon_results = []

preprocessors = {}


# ============================================================
# 12. TRAIN EACH PLATFORM
# ============================================================

for platform in PLATFORMS:

    print("\n")
    print("=" * 70)

    print(
        f"TRAINING {platform.upper()} RNN & LSTM"
    )

    print("=" * 70)


    # ========================================================
    # 12.1 TARGET COLUMNS
    # ========================================================

    target_columns = [

        f"{platform}_Future_15",

        f"{platform}_Future_30",

        f"{platform}_Future_45",

        f"{platform}_Future_60"
    ]


    # ========================================================
    # 12.2 CREATE X AND Y
    # ========================================================

    X_df = df[
        FEATURE_COLUMNS
    ].copy()

    y_df = df[
        target_columns
    ].copy()


    # ========================================================
    # 12.3 HANDLE MISSING VALUES
    # ========================================================

    # Categorical missing values
    for col in CATEGORICAL:

        if col in X_df.columns:

            X_df[col] = X_df[col].fillna(
                X_df[col].mode()[0]
            )


    # Numerical missing values
    numerical_columns = [
        col
        for col in FEATURE_COLUMNS
        if col not in CATEGORICAL
    ]

    for col in numerical_columns:

        X_df[col] = X_df[col].fillna(
            X_df[col].median()
        )


    # Targets
    valid_rows = ~y_df.isna().any(
        axis=1
    )

    X_df = X_df.loc[
        valid_rows
    ].reset_index(drop=True)

    y_df = y_df.loc[
        valid_rows
    ].reset_index(drop=True)


    # ========================================================
    # 12.4 CHRONOLOGICAL SPLIT
    # ========================================================

    split = int(
        len(X_df) * 0.80
    )

    X_train_df = X_df.iloc[
        :split
    ].copy()

    X_test_df = X_df.iloc[
        split:
    ].copy()

    y_train_df = y_df.iloc[
        :split
    ].copy()

    y_test_df = y_df.iloc[
        split:
    ].copy()


    print(
        "\nTraining rows:",
        len(X_train_df)
    )

    print(
        "Testing rows :",
        len(X_test_df)
    )


    # ========================================================
    # 12.5 PREPROCESS INPUT FEATURES
    # ========================================================

    categorical_features = [
        col
        for col in CATEGORICAL
        if col in FEATURE_COLUMNS
    ]

    numerical_features = [
        col
        for col in FEATURE_COLUMNS
        if col not in categorical_features
    ]


    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),

                categorical_features
            ),

            (
                "numerical",

                StandardScaler(),

                numerical_features
            )
        ]

    )


    # Fit ONLY on training data
    X_train_processed = preprocessor.fit_transform(
        X_train_df
    )

    X_test_processed = preprocessor.transform(
        X_test_df
    )


    # ========================================================
    # 12.6 SCALE TARGETS
    # ========================================================

    target_scaler = StandardScaler()

    y_train_scaled = target_scaler.fit_transform(
        y_train_df
    )

    y_test_scaled = target_scaler.transform(
        y_test_df
    )


    # ========================================================
    # 12.7 SAVE PREPROCESSOR
    # ========================================================

    preprocessors[platform] = {
        "preprocessor": preprocessor,
        "target_scaler": target_scaler,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": target_columns
    }


    # ========================================================
    # 12.8 CREATE SEQUENCES
    # ========================================================

    X_train, y_train = create_sequences(

        X_train_processed,

        y_train_scaled,

        SEQ_LEN
    )


    # For testing, we need the previous SEQ_LEN-1
    # observations from the training period.
    X_test_combined = np.vstack([

        X_train_processed[
            -(SEQ_LEN - 1):
        ],

        X_test_processed

    ])

    y_test_combined = np.vstack([

        y_train_scaled[
            -(SEQ_LEN - 1):
        ],

        y_test_scaled

    ])


    X_test, y_test = create_sequences(

        X_test_combined,

        y_test_combined,

        SEQ_LEN
    )


    print(
        "\nSequence shapes:"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    print(
        "y_test :",
        y_test.shape
    )


    # ========================================================
    # 12.9 VALIDATION SPLIT
    # ========================================================

    val_size = int(
        len(X_train) * 0.15
    )

    X_tr = X_train[
        :-val_size
    ]

    X_val = X_train[
        -val_size:
    ]

    y_tr = y_train[
        :-val_size
    ]

    y_val = y_train[
        -val_size:
    ]


    # ========================================================
    # 12.10 MODEL BUILDERS
    # ========================================================

    model_builders = {

        "RNN": build_rnn,

        "LSTM": build_lstm
    }


    # ========================================================
    # 12.11 TRAIN RNN AND LSTM
    # ========================================================

    for model_name, builder in model_builders.items():

        print("\n")
        print("-" * 65)

        print(
            f"{platform} - {model_name}"
        )

        print("-" * 65)


        # Reset seed
        tf.random.set_seed(
            RANDOM_SEED
        )


        # Build model
        model = builder(
            (
                X_train.shape[1],
                X_train.shape[2]
            )
        )


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        early_stopping = tf.keras.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=6,

            restore_best_weights=True
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        history = model.fit(

            X_tr,

            y_tr,

            validation_data=(
                X_val,
                y_val
            ),

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            callbacks=[
                early_stopping
            ],

            verbose=1
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        pred_scaled = model.predict(
            X_test,
            verbose=0
        )


        # ====================================================
        # INVERSE TRANSFORM
        # ====================================================

        pred = target_scaler.inverse_transform(
            pred_scaled
        )

        actual = target_scaler.inverse_transform(
            y_test
        )


        # ====================================================
        # OVERALL METRICS
        # ====================================================

        mae = mean_absolute_error(
            actual,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                pred
            )
        )

        r2 = r2_score(
            actual,
            pred,
            multioutput="uniform_average"
        )


        print("\nResults:")

        print(
            f"MAE  : {mae:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"R²   : {r2:.4f}"
        )


        # ====================================================
        # STORE OVERALL RESULTS
        # ====================================================

        results.append({

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


        # ====================================================
        # HORIZON-WISE RESULTS
        # ====================================================

        print(
            "\nHorizon-wise results:"
        )


        for i, horizon in enumerate(
            [15, 30, 45, 60]
        ):

            actual_h = actual[
                :, i
            ]

            pred_h = pred[
                :, i
            ]


            h_mae = mean_absolute_error(
                actual_h,
                pred_h
            )


            h_rmse = np.sqrt(
                mean_squared_error(
                    actual_h,
                    pred_h
                )
            )


            h_r2 = r2_score(
                actual_h,
                pred_h
            )


            print(
                f"{horizon} min -> "
                f"MAE={h_mae:.3f}, "
                f"RMSE={h_rmse:.3f}, "
                f"R²={h_r2:.3f}"
            )


            horizon_results.append({

                "Platform":
                    platform,

                "Model":
                    model_name,

                "Horizon":
                    f"{horizon} min",

                "MAE":
                    h_mae,

                "RMSE":
                    h_rmse,

                "R2":
                    h_r2
            })


        # ====================================================
        # SAVE MODEL
        # ====================================================

        model_filename = (

            f"{platform.lower()}_"

            f"{model_name.lower()}_model.keras"
        )


        model_path = os.path.join(

            MODEL_DIR,

            model_filename
        )


        model.save(
            model_path
        )


        print(
            "\nModel saved:"
        )

        print(
            model_path
        )


        # ====================================================
        # SAVE TRAINING HISTORY
        # ====================================================

        history_filename = (

            f"{platform.lower()}_"

            f"{model_name.lower()}_history.csv"
        )


        history_path = os.path.join(

            OUTPUT_DIR,

            history_filename
        )


        pd.DataFrame(
            history.history
        ).to_csv(

            history_path,

            index=False
        )


# ============================================================
# 13. SAVE PREPROCESSORS
# ============================================================

preprocessor_path = os.path.join(

    MODEL_DIR,

    "dl_preprocessors.joblib"
)


joblib.dump(

    preprocessors,

    preprocessor_path
)


print(
    "\nDL preprocessors saved:"
)

print(
    preprocessor_path
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

metrics = pd.DataFrame(
    results
)


metrics_path = os.path.join(

    OUTPUT_DIR,

    "dl_metrics.csv"
)


metrics.to_csv(

    metrics_path,

    index=False
)


# ============================================================
# 15. SAVE HORIZON RESULTS
# ============================================================

horizon_metrics = pd.DataFrame(
    horizon_results
)


horizon_path = os.path.join(

    OUTPUT_DIR,

    "dl_horizon_metrics.csv"
)


horizon_metrics.to_csv(

    horizon_path,

    index=False
)


# ============================================================
# 16. DISPLAY FINAL RESULTS
# ============================================================

print("\n")
print("=" * 70)

print(
    "MULTIVARIATE DL RESULTS"
)

print("=" * 70)


print(
    metrics.to_string(
        index=False
    )
)


# ============================================================
# 17. BEST DL MODEL
# ============================================================

print("\n")
print("=" * 70)

print(
    "BEST DL MODEL FOR EACH PLATFORM"
)

print("=" * 70)


for platform in PLATFORMS:

    platform_results = metrics[
        metrics["Platform"] == platform
    ]


    if platform_results.empty:
        continue


    best_index = platform_results[
        "RMSE"
    ].idxmin()


    best = platform_results.loc[
        best_index
    ]


    print(
        f"\n{platform}:"
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
# 18. CREATE COMPARISON GRAPH
# ============================================================

average_mae = (

    metrics

    .groupby("Model")["MAE"]

    .mean()

    .sort_values()
)


plt.figure(
    figsize=(8, 5)
)


average_mae.plot(
    kind="bar"
)


plt.ylabel(
    "MAE (₹)"
)


plt.xlabel(
    "Deep Learning Model"
)


plt.title(
    "Ridely AI - Multivariate DL Model Comparison"
)


plt.xticks(
    rotation=0
)


plt.tight_layout()


comparison_path = os.path.join(

    OUTPUT_DIR,

    "dl_comparison.png"
)


plt.savefig(

    comparison_path,

    dpi=160
)


plt.close()


# ============================================================
# 19. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)

print(
    "MULTIVARIATE DL TRAINING COMPLETED"
)

print("=" * 70)


print(
    "\nModels saved in:"
)

print(
    MODEL_DIR
)


print(
    "\nPreprocessors saved to:"
)

print(
    preprocessor_path
)


print(
    "\nMetrics saved to:"
)

print(
    metrics_path
)


print(
    "\nHorizon metrics saved to:"
)

print(
    horizon_path
)


print(
    "\nComparison graph saved to:"
)

print(
    comparison_path
)