import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)


# ============================================================
# IMPORTS
# ============================================================

from src.route_service import calculate_route

from src.weather_service import get_weather

from src.auto_features import (
    generate_automatic_features
)

from src.fare_estimator import (
    estimate_current_fares
)

from src.predict import (
    predict_fares
)

from src.recommender import (
    generate_recommendation
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Ridely AI",
    page_icon="🚕",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🚕 Ridely AI")

st.subheader(
    "Future Fare Prediction & Smart Booking Advisor"
)

st.write(
    "Enter your pickup and drop locations. "
    "Ridely AI automatically calculates route information, "
    "weather, traffic and contextual features before "
    "predicting future fares."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("📍 Enter Your Journey")

col1, col2 = st.columns(2)

with col1:

    pickup = st.text_input(
        "Pickup Location",
        placeholder="Example: Gachibowli"
    )

with col2:

    drop = st.text_input(
        "Drop Location",
        placeholder="Example: Hitech City"
    )


ride_type = st.selectbox(
    "🚗 Ride Type",
    [
        "Bike",
        "Auto",
        "Mini",
        "Sedan"
    ]
)


predict_button = st.button(
    "🔮 Predict Fare",
    type="primary",
    use_container_width=True
)


# ============================================================
# MAIN PROCESS
# ============================================================

if predict_button:

    if not pickup.strip():

        st.error(
            "Please enter the pickup location."
        )

        st.stop()

    if not drop.strip():

        st.error(
            "Please enter the drop location."
        )

        st.stop()

    try:

        # ====================================================
        # STEP 1 — ROUTE
        # ====================================================

        with st.spinner(
            "📍 Finding locations and calculating route..."
        ):

            route = calculate_route(
                pickup.strip(),
                drop.strip()
            )

        distance_km = route[
            "distance_km"
        ]

        duration_min = route[
            "duration_min"
        ]

        pickup_coordinates = route[
            "pickup"
        ]

        drop_coordinates = route[
            "drop"
        ]


        # ====================================================
        # STEP 2 — WEATHER
        # ====================================================

        with st.spinner(
            "🌦️ Checking weather..."
        ):

            weather_data = get_weather(
                pickup_coordinates["latitude"],
                pickup_coordinates["longitude"]
            )

        weather = weather_data[
            "category"
        ]


        # ====================================================
        # STEP 3 — AUTOMATIC FEATURES
        # ====================================================

        now = datetime.now()

        automatic = generate_automatic_features(
            distance_km=distance_km,
            duration_min=duration_min,
            weather=weather,
            now=now
        )


        # ====================================================
        # STEP 4 — CURRENT FARE ESTIMATION
        # ====================================================

        current_fares = estimate_current_fares(

            distance_km=distance_km,

            duration_min=duration_min,

            demand=automatic[
                "Demand_Index"
            ],

            traffic=automatic[
                "Traffic"
            ],

            ride_type=ride_type
        )


        # ====================================================
        # STEP 5 — BUILD MODEL INPUT
        # ====================================================

        model_input = {

            "Pickup": pickup,

            "Drop": drop,

            "Distance_km":
                distance_km,

            "Duration_min":
                duration_min,

            "Traffic":
                automatic["Traffic"],

            "Weather":
                automatic["Weather"],

            "Ride_Type":
                ride_type,

            "Demand_Index":
                automatic["Demand_Index"],

            "Driver_Availability":
                automatic[
                    "Driver_Availability"
                ],

            "Weekend":
                automatic["Weekend"],

            "Peak_Hour":
                automatic["Peak_Hour"],

            "Holiday":
                automatic["Holiday"],

            "Uber_Fare":
                current_fares["Uber"],

            "Ola_Fare":
                current_fares["Ola"],

            "Rapido_Fare":
                current_fares["Rapido"],

            "Date":
                now.strftime("%Y-%m-%d"),

            "Time":
                now.strftime("%H:%M")
        }


        # ====================================================
        # STEP 6 — ML PREDICTION
        # ====================================================

        with st.spinner(
            "🤖 Predicting future fares..."
        ):

            predictions = predict_fares(
                model_input
            )


        # ====================================================
        # STEP 7 — RECOMMENDATION
        # ====================================================

        recommendation = (
            generate_recommendation(
                predictions
            )
        )


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "Fare prediction completed successfully!"
        )


        # ====================================================
        # ROUTE INFORMATION
        # ====================================================

        st.header(
            "🗺️ Automatically Calculated Trip Information"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Distance",
                f"{distance_km:.2f} km"
            )

        with c2:

            st.metric(
                "Duration",
                f"{duration_min:.0f} min"
            )

        with c3:

            st.metric(
                "Traffic",
                automatic["Traffic"]
            )

        with c4:

            st.metric(
                "Weather",
                weather
            )


        # ====================================================
        # CONTEXTUAL FEATURES
        # ====================================================

        with st.expander(
            "🔍 View automatically calculated AI features"
        ):

            feature_df = pd.DataFrame({

                "Feature": [
                    "Demand Index",
                    "Driver Availability",
                    "Peak Hour",
                    "Weekend",
                    "Ride Type"
                ],

                "Value": [
                    automatic[
                        "Demand_Index"
                    ],

                    automatic[
                        "Driver_Availability"
                    ],

                    "Yes"
                    if automatic[
                        "Peak_Hour"
                    ]
                    else "No",

                    "Yes"
                    if automatic[
                        "Weekend"
                    ]
                    else "No",

                    ride_type
                ]
            })

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # CURRENT ESTIMATED FARES
        # ====================================================

        st.header(
            "💰 Estimated Current Fares"
        )

        current_df = pd.DataFrame({

            "Platform": [
                "Uber",
                "Ola",
                "Rapido"
            ],

            "Estimated Current Fare": [

                f"₹{current_fares['Uber']:.2f}",

                f"₹{current_fares['Ola']:.2f}",

                f"₹{current_fares['Rapido']:.2f}"
            ]
        })

        st.dataframe(
            current_df,
            use_container_width=True,
            hide_index=True
        )


        st.caption(
            "Current fares shown above are model-input "
            "baseline estimates, not live ride-hailing prices."
        )


        # ====================================================
        # FUTURE PREDICTION TABLE
        # ====================================================

        st.header(
            "📊 Future Fare Prediction"
        )

        rows = []

        for platform, values in predictions.items():

            rows.append({

                "Platform":
                    platform,

                "+15 min":
                    f"₹{values[0]:.2f}",

                "+30 min":
                    f"₹{values[1]:.2f}",

                "+45 min":
                    f"₹{values[2]:.2f}",

                "+60 min":
                    f"₹{values[3]:.2f}"
            })


        prediction_df = pd.DataFrame(
            rows
        )

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # CHART
        # ====================================================

        st.header(
            "📈 Predicted Fare Trend"
        )

        chart_data = pd.DataFrame({

            "Uber": predictions["Uber"],

            "Ola": predictions["Ola"],

            "Rapido": predictions["Rapido"]

        }, index=[

            "+15 min",
            "+30 min",
            "+45 min",
            "+60 min"

        ])

        st.line_chart(
            chart_data
        )


        # ====================================================
        # RECOMMENDATION
        # ====================================================

        st.divider()

        st.header(
            "🏆 Smart Booking Recommendation"
        )

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Recommended Platform",
                recommendation[
                    "recommended_platform"
                ]
            )

        with r2:

            st.metric(
                "Recommended Time",
                recommendation[
                    "recommended_time"
                ]
            )

        with r3:

            st.metric(
                "Expected Fare",
                f"₹{recommendation['expected_fare']:.2f}"
            )


        st.info(
            recommendation["reason"]
        )


        # ====================================================
        # LOCATION DETAILS
        # ====================================================

        with st.expander(
            "📍 Location details"
        ):

            st.write(
                "**Pickup:**",
                pickup_coordinates[
                    "display_name"
                ]
            )

            st.write(
                "**Drop:**",
                drop_coordinates[
                    "display_name"
                ]
            )


    except Exception as e:

        st.error(
            "Unable to calculate the journey."
        )

        st.exception(e)


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        "Enter a pickup location and drop location "
        "to begin."
    )

    st.markdown(
        """
        ### How Ridely AI works

        **1.** Enter pickup and drop locations.

        **2.** Ridely AI automatically finds the locations.

        **3.** Road distance and estimated journey duration
        are calculated.

        **4.** Current weather is obtained automatically.

        **5.** Traffic, demand and driver availability
        are estimated from contextual information.

        **6.** The trained ML model predicts fares for
        the next 15, 30, 45 and 60 minutes.

        **7.** Uber, Ola and Rapido are compared.

        **8.** The Smart Booking Advisor recommends
        the best platform and booking time.
        """
    )