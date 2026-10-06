from datetime import datetime


# ============================================================
# TRAFFIC ESTIMATION
# ============================================================

def estimate_traffic(hour, day_of_week):

    # Weekday rush hours
    weekday = day_of_week < 5

    if weekday:

        if hour in [8, 9, 10, 17, 18, 19, 20]:
            return "High"

        if hour in [7, 11, 16, 21]:
            return "Medium"

    else:

        if hour in [12, 13, 18, 19, 20]:
            return "Medium"

    return "Low"


# ============================================================
# DEMAND ESTIMATION
# ============================================================

def estimate_demand(
    hour,
    traffic,
    weekend
):

    score = 4

    # Peak hours
    if hour in [8, 9, 10, 17, 18, 19, 20]:
        score += 2

    # Traffic
    if traffic == "High":
        score += 2

    elif traffic == "Medium":
        score += 1

    # Weekend
    if weekend == 1:
        score += 1

    return max(1, min(10, score))


# ============================================================
# DRIVER AVAILABILITY
# ============================================================

def estimate_driver_availability(
    demand,
    traffic
):

    availability = 10 - demand + 2

    if traffic == "High":
        availability -= 1

    return max(
        1,
        min(10, availability)
    )


# ============================================================
# PEAK HOUR
# ============================================================

def calculate_peak_hour(hour):

    return int(
        hour in [
            7, 8, 9, 10,
            17, 18, 19, 20
        ]
    )


# ============================================================
# AUTOMATIC FEATURES
# ============================================================

def generate_automatic_features(
    distance_km,
    duration_min,
    weather,
    now=None
):

    if now is None:
        now = datetime.now()

    hour = now.hour
    minute = now.minute

    day_of_week = now.weekday()

    weekend = int(
        day_of_week >= 5
    )

    peak_hour = calculate_peak_hour(
        hour
    )

    traffic = estimate_traffic(
        hour,
        day_of_week
    )

    demand = estimate_demand(
        hour,
        traffic,
        weekend
    )

    availability = estimate_driver_availability(
        demand,
        traffic
    )

    holiday = 0

    return {
        "Hour": hour,
        "Minute": minute,
        "Day_of_Week": day_of_week,
        "Weekend": weekend,
        "Peak_Hour": peak_hour,
        "Traffic": traffic,
        "Demand_Index": demand,
        "Driver_Availability": availability,
        "Holiday": holiday,
        "Distance_km": distance_km,
        "Duration_min": duration_min,
        "Weather": weather
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    result = generate_automatic_features(
        distance_km=12,
        duration_min=30,
        weather="Sunny"
    )

    print("\nAutomatic Features")
    print("-" * 40)

    for key, value in result.items():
        print(f"{key}: {value}")