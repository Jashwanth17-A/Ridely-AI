import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/raw/ridely_historical.csv"
OUTPUT_FILE = "data/processed/ridely_processed.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(file_path):
    """
    Load the raw Ridely dataset.
    """

    print("\n" + "=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully.")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


# ============================================================
# 2. BASIC DATA INFORMATION
# ============================================================

def inspect_data(df):
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 60)
    print("2. DATA INFORMATION")
    print("=" * 60)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nStatistical summary:")
    print(df.describe(include="all").T)


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

def check_missing_values(df):
    """
    Check missing values in every column.
    """

    print("\n" + "=" * 60)
    print("3. MISSING VALUE CHECK")
    print("=" * 60)

    missing = df.isnull().sum()

    missing_table = pd.DataFrame({
        "Column": missing.index,
        "Missing_Count": missing.values
    })

    missing_table["Missing_Percentage"] = (
        missing_table["Missing_Count"]
        / len(df)
        * 100
    )

    print(missing_table.to_string(index=False))

    total_missing = missing.sum()

    print(f"\nTotal missing values: {total_missing}")

    return missing_table


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):
    """
    Handle missing values.

    Numerical columns:
        Fill using median.

    Categorical columns:
        Fill using mode.
    """

    print("\n" + "=" * 60)
    print("4. HANDLING MISSING VALUES")
    print("=" * 60)

    df = df.copy()

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    # Numerical columns
    for column in numerical_columns:

        if df[column].isnull().sum() > 0:

            median_value = df[column].median()

            df[column] = df[column].fillna(
                median_value
            )

            print(
                f"{column}: filled missing values "
                f"with median = {median_value}"
            )

    # Categorical columns
    for column in categorical_columns:

        if df[column].isnull().sum() > 0:

            mode_value = df[column].mode()[0]

            df[column] = df[column].fillna(
                mode_value
            )

            print(
                f"{column}: filled missing values "
                f"with mode = {mode_value}"
            )

    print("\nMissing values handled.")

    return df


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    """
    Remove duplicate records.
    """

    print("\n" + "=" * 60)
    print("5. DUPLICATE CHECK")
    print("=" * 60)

    before = len(df)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows found: {duplicate_count}")

    df = df.drop_duplicates()

    after = len(df)

    print(f"Rows before removing duplicates: {before}")
    print(f"Rows after removing duplicates : {after}")
    print(f"Rows removed                   : {before - after}")

    return df


# ============================================================
# 6. DATA TYPE CONVERSION
# ============================================================

def convert_data_types(df):
    """
    Convert columns to appropriate data types.
    """

    print("\n" + "=" * 60)
    print("6. DATA TYPE CONVERSION")
    print("=" * 60)

    df = df.copy()

    # Convert date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Convert numerical columns
    numerical_columns = [
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

    for column in numerical_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    print("Data types converted successfully.")

    return df


# ============================================================
# 7. HANDLE INVALID VALUES
# ============================================================

def handle_invalid_values(df):
    """
    Replace physically/logically invalid values.
    """

    print("\n" + "=" * 60)
    print("7. INVALID VALUE CHECK")
    print("=" * 60)

    df = df.copy()

    # Distance cannot be zero or negative
    invalid_distance = (df["Distance_km"] <= 0).sum()

    if invalid_distance > 0:

        print(
            f"Invalid distance records: "
            f"{invalid_distance}"
        )

        df.loc[
            df["Distance_km"] <= 0,
            "Distance_km"
        ] = np.nan

    # Duration cannot be zero or negative
    invalid_duration = (df["Duration_min"] <= 0).sum()

    if invalid_duration > 0:

        print(
            f"Invalid duration records: "
            f"{invalid_duration}"
        )

        df.loc[
            df["Duration_min"] <= 0,
            "Duration_min"
        ] = np.nan

    # Demand should be between 1 and 10
    df["Demand_Index"] = df["Demand_Index"].clip(
        lower=1,
        upper=10
    )

    # Driver availability should be between 1 and 10
    df["Driver_Availability"] = df[
        "Driver_Availability"
    ].clip(
        lower=1,
        upper=10
    )

    # Fares should not be negative
    fare_columns = [
        "Uber_Fare",
        "Ola_Fare",
        "Rapido_Fare"
    ]

    future_columns = [
        column
        for column in df.columns
        if "Future_" in column
    ]

    for column in fare_columns + future_columns:

        if column in df.columns:

            df[column] = df[column].clip(
                lower=0
            )

    # Fill values made invalid with median
    df["Distance_km"] = df[
        "Distance_km"
    ].fillna(
        df["Distance_km"].median()
    )

    df["Duration_min"] = df[
        "Duration_min"
    ].fillna(
        df["Duration_min"].median()
    )

    print("Invalid values handled.")

    return df


# ============================================================
# 8. DATE + TIME FEATURES
# ============================================================

def create_time_features(df):
    """
    Extract useful information from Date and Time.
    """

    print("\n" + "=" * 60)
    print("8. CREATING TIME FEATURES")
    print("=" * 60)

    df = df.copy()

    # Combine date and time
    df["Datetime"] = pd.to_datetime(
        df["Date"].astype(str)
        + " "
        + df["Time"].astype(str),
        errors="coerce"
    )

    # Hour
    df["Hour"] = df["Datetime"].dt.hour

    # Minute
    df["Minute"] = df["Datetime"].dt.minute

    # Day of week
    df["Day_of_Week"] = (
        df["Datetime"].dt.dayofweek
    )

    # Day name
    df["Day_Name"] = (
        df["Datetime"].dt.day_name()
    )

    # Month
    df["Month"] = (
        df["Datetime"].dt.month
    )

    # Week of year
    df["Week_of_Year"] = (
        df["Datetime"].dt.isocalendar().week
        .astype(int)
    )

    print("Created:")
    print("  Hour")
    print("  Minute")
    print("  Day_of_Week")
    print("  Day_Name")
    print("  Month")
    print("  Week_of_Year")

    return df


# ============================================================
# 9. PEAK HOUR FEATURE
# ============================================================

def create_peak_hour_feature(df):
    """
    Create peak-hour indicator.

    Morning peak:
        07:00 - 10:00

    Evening peak:
        17:00 - 21:00
    """

    print("\n" + "=" * 60)
    print("9. CREATING PEAK HOUR FEATURE")
    print("=" * 60)

    df = df.copy()

    df["Peak_Hour"] = (
        (
            (df["Hour"] >= 7)
            & (df["Hour"] <= 10)
        )
        |
        (
            (df["Hour"] >= 17)
            & (df["Hour"] <= 21)
        )
    ).astype(int)

    print(
        "Peak hour feature created."
    )

    print(
        df["Peak_Hour"].value_counts()
    )

    return df


# ============================================================
# 10. WEEKEND FEATURE
# ============================================================

def create_weekend_feature(df):
    """
    Saturday and Sunday = 1
    Other days = 0
    """

    print("\n" + "=" * 60)
    print("10. CREATING WEEKEND FEATURE")
    print("=" * 60)

    df = df.copy()

    df["Weekend"] = (
        df["Day_of_Week"] >= 5
    ).astype(int)

    print(
        df["Weekend"].value_counts()
    )

    return df


# ============================================================
# 11. TIME OF DAY
# ============================================================

def create_time_of_day(df):
    """
    Convert hour into meaningful time categories.
    """

    print("\n" + "=" * 60)
    print("11. CREATING TIME OF DAY")
    print("=" * 60)

    df = df.copy()

    def classify_time(hour):

        if 5 <= hour < 12:
            return "Morning"

        elif 12 <= hour < 17:
            return "Afternoon"

        elif 17 <= hour < 22:
            return "Evening"

        else:
            return "Night"

    df["Time_of_Day"] = (
        df["Hour"].apply(classify_time)
    )

    print(
        df["Time_of_Day"].value_counts()
    )

    return df


# ============================================================
# 12. TRAFFIC NUMERIC ENCODING
# ============================================================

def encode_traffic(df):
    """
    Convert traffic category into numerical score.

    Low    = 0
    Medium = 1
    High   = 2
    """

    print("\n" + "=" * 60)
    print("12. ENCODING TRAFFIC")
    print("=" * 60)

    df = df.copy()

    traffic_mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    df["Traffic_Score"] = (
        df["Traffic"]
        .map(traffic_mapping)
    )

    print(
        df[
            ["Traffic", "Traffic_Score"]
        ].drop_duplicates()
    )

    return df


# ============================================================
# 13. WEATHER ENCODING
# ============================================================

def encode_weather(df):
    """
    Convert weather into numerical representation.
    """

    print("\n" + "=" * 60)
    print("13. ENCODING WEATHER")
    print("=" * 60)

    df = df.copy()

    weather_mapping = {
        "Sunny": 0,
        "Cloudy": 1,
        "Rainy": 2
    }

    df["Weather_Score"] = (
        df["Weather"]
        .map(weather_mapping)
    )

    print(
        df[
            ["Weather", "Weather_Score"]
        ].drop_duplicates()
    )

    return df


# ============================================================
# 14. DEMAND / SUPPLY RATIO
# ============================================================

def create_demand_supply_ratio(df):
    """
    Calculate demand relative to driver availability.
    """

    print("\n" + "=" * 60)
    print("14. CREATING DEMAND/SUPPLY RATIO")
    print("=" * 60)

    df = df.copy()

    df["Demand_Supply_Ratio"] = (
        df["Demand_Index"]
        /
        df["Driver_Availability"].clip(
            lower=1
        )
    )

    print(
        df[
            "Demand_Supply_Ratio"
        ].describe()
    )

    return df


# ============================================================
# 15. AVERAGE PLATFORM FARE
# ============================================================

def create_average_fare(df):
    """
    Calculate average current fare across platforms.
    """

    print("\n" + "=" * 60)
    print("15. CREATING AVERAGE FARE")
    print("=" * 60)

    df = df.copy()

    fare_columns = [
        "Uber_Fare",
        "Ola_Fare",
        "Rapido_Fare"
    ]

    df["Average_Current_Fare"] = (
        df[fare_columns].mean(axis=1)
    )

    return df


# ============================================================
# 16. CHEAPEST PLATFORM
# ============================================================

def create_cheapest_platform(df):
    """
    Determine the platform with the lowest current fare.
    """

    print("\n" + "=" * 60)
    print("16. CREATING CHEAPEST PLATFORM")
    print("=" * 60)

    df = df.copy()

    fare_columns = [
        "Uber_Fare",
        "Ola_Fare",
        "Rapido_Fare"
    ]

    df["Cheapest_Platform"] = (
        df[fare_columns]
        .idxmin(axis=1)
        .str.replace(
            "_Fare",
            "",
            regex=False
        )
    )

    print(
        df[
            "Cheapest_Platform"
        ].value_counts()
    )

    return df


# ============================================================
# 17. FARE DIFFERENCE
# ============================================================

def create_fare_difference_features(df):
    """
    Create fare difference features between platforms.
    """

    print("\n" + "=" * 60)
    print("17. CREATING FARE DIFFERENCE FEATURES")
    print("=" * 60)

    df = df.copy()

    df["Uber_Ola_Difference"] = (
        df["Uber_Fare"]
        -
        df["Ola_Fare"]
    )

    df["Uber_Rapido_Difference"] = (
        df["Uber_Fare"]
        -
        df["Rapido_Fare"]
    )

    df["Ola_Rapido_Difference"] = (
        df["Ola_Fare"]
        -
        df["Rapido_Fare"]
    )

    return df


# ============================================================
# 18. CYCLICAL TIME FEATURES
# ============================================================

def create_cyclical_features(df):
    """
    Represent time cyclically.

    This allows the model to understand that:
    23:00 and 00:00 are close in time.
    """

    print("\n" + "=" * 60)
    print("18. CREATING CYCLICAL TIME FEATURES")
    print("=" * 60)

    df = df.copy()

    # Hour cycle
    df["Hour_sin"] = np.sin(
        2 * np.pi * df["Hour"] / 24
    )

    df["Hour_cos"] = np.cos(
        2 * np.pi * df["Hour"] / 24
    )

    # Day-of-week cycle
    df["Day_sin"] = np.sin(
        2 * np.pi * df["Day_of_Week"] / 7
    )

    df["Day_cos"] = np.cos(
        2 * np.pi * df["Day_of_Week"] / 7
    )

    print(
        "Created Hour_sin, Hour_cos, "
        "Day_sin and Day_cos."
    )

    return df


# ============================================================
# 19. DISTANCE CATEGORIES
# ============================================================

def create_distance_category(df):
    """
    Convert distance into short/medium/long categories.
    """

    print("\n" + "=" * 60)
    print("19. CREATING DISTANCE CATEGORY")
    print("=" * 60)

    df = df.copy()

    def classify_distance(distance):

        if distance <= 5:
            return "Short"

        elif distance <= 15:
            return "Medium"

        else:
            return "Long"

    df["Distance_Category"] = (
        df["Distance_km"]
        .apply(classify_distance)
    )

    print(
        df[
            "Distance_Category"
        ].value_counts()
    )

    return df


# ============================================================
# 20. SPEED FEATURE
# ============================================================

def create_speed_feature(df):
    """
    Estimate average travel speed.

    Speed = Distance / Time
    """

    print("\n" + "=" * 60)
    print("20. CREATING AVERAGE SPEED")
    print("=" * 60)

    df = df.copy()

    df["Average_Speed_kmph"] = (
        df["Distance_km"]
        /
        (df["Duration_min"] / 60)
    )

    # Avoid unrealistic values
    df["Average_Speed_kmph"] = (
        df["Average_Speed_kmph"]
        .clip(lower=1, upper=120)
    )

    return df


# ============================================================
# 21. REMOVE OUTLIERS
# ============================================================

def remove_outliers(df):
    """
    Remove extreme outliers from selected numerical columns
    using the IQR method.
    """

    print("\n" + "=" * 60)
    print("21. OUTLIER HANDLING")
    print("=" * 60)

    df = df.copy()

    columns = [
        "Distance_km",
        "Duration_min",
        "Demand_Index",
        "Driver_Availability",
        "Uber_Fare",
        "Ola_Fare",
        "Rapido_Fare"
    ]

    before = len(df)

    for column in columns:

        if column not in df.columns:
            continue

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[
            (df[column] >= lower)
            &
            (df[column] <= upper)
        ]

    after = len(df)

    print(
        f"Rows before outlier removal: {before}"
    )

    print(
        f"Rows after outlier removal : {after}"
    )

    print(
        f"Rows removed               : "
        f"{before - after}"
    )

    return df


# ============================================================
# 22. SORT DATA CHRONOLOGICALLY
# ============================================================

def sort_data(df):
    """
    Sort data by datetime.

    Important for time-series forecasting.
    """

    print("\n" + "=" * 60)
    print("22. SORTING DATA")
    print("=" * 60)

    df = df.copy()

    df = df.sort_values(
        by="Datetime"
    )

    df = df.reset_index(
        drop=True
    )

    print("Dataset sorted chronologically.")

    return df


# ============================================================
# 23. SAVE PROCESSED DATA
# ============================================================

def save_processed_data(df, file_path):
    """
    Save final processed dataset.
    """

    print("\n" + "=" * 60)
    print("23. SAVING PROCESSED DATA")
    print("=" * 60)

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    df.to_csv(
        file_path,
        index=False
    )

    print(
        f"Processed dataset saved to:\n"
        f"{file_path}"
    )

    print(
        f"Final rows    : {df.shape[0]}"
    )

    print(
        f"Final columns : {df.shape[1]}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("        RIDELY AI - DATA PREPROCESSING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    df = load_data(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    inspect_data(
        df
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    check_missing_values(
        df
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    df = handle_missing_values(
        df
    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    df = remove_duplicates(
        df
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    df = convert_data_types(
        df
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    df = handle_invalid_values(
        df
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    df = create_time_features(
        df
    )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    df = create_peak_hour_feature(
        df
    )

    # --------------------------------------------------------
    # STEP 10
    # --------------------------------------------------------

    df = create_weekend_feature(
        df
    )

    # --------------------------------------------------------
    # STEP 11
    # --------------------------------------------------------

    df = create_time_of_day(
        df
    )

    # --------------------------------------------------------
    # STEP 12
    # --------------------------------------------------------

    df = encode_traffic(
        df
    )

    # --------------------------------------------------------
    # STEP 13
    # --------------------------------------------------------

    df = encode_weather(
        df
    )

    # --------------------------------------------------------
    # STEP 14
    # --------------------------------------------------------

    df = create_demand_supply_ratio(
        df
    )

    # --------------------------------------------------------
    # STEP 15
    # --------------------------------------------------------

    df = create_average_fare(
        df
    )

    # --------------------------------------------------------
    # STEP 16
    # --------------------------------------------------------

    df = create_cheapest_platform(
        df
    )

    # --------------------------------------------------------
    # STEP 17
    # --------------------------------------------------------

    df = create_fare_difference_features(
        df
    )

    # --------------------------------------------------------
    # STEP 18
    # --------------------------------------------------------

    df = create_cyclical_features(
        df
    )

    # --------------------------------------------------------
    # STEP 19
    # --------------------------------------------------------

    df = create_distance_category(
        df
    )

    # --------------------------------------------------------
    # STEP 20
    # --------------------------------------------------------

    df = create_speed_feature(
        df
    )

    # --------------------------------------------------------
    # STEP 21
    # --------------------------------------------------------

    df = remove_outliers(
        df
    )

    # --------------------------------------------------------
    # STEP 22
    # --------------------------------------------------------

    df = sort_data(
        df
    )

    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL DATASET CHECK")
    print("=" * 60)

    print("\nShape:")
    print(df.shape)

    print("\nMissing values:")
    print(
        df.isnull().sum()
        .sum()
    )

    print("\nFinal columns:")
    for i, column in enumerate(
        df.columns,
        start=1
    ):
        print(
            f"{i:02d}. {column}"
        )

    print("\nSample:")
    print(
        df.head().to_string()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_processed_data(
        df,
        OUTPUT_FILE
    )

    print("\n")
    print("=" * 70)
    print("       PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()