# ============================================================
# BASELINE CURRENT FARE ESTIMATOR
# ============================================================

def estimate_current_fares(
    distance_km,
    duration_min,
    demand,
    traffic,
    ride_type
):

    # Base fare
    base = {
        "Uber": 50,
        "Ola": 45,
        "Rapido": 35
    }

    # Per km
    per_km = {
        "Uber": 16,
        "Ola": 15,
        "Rapido": 11
    }

    # Per minute
    per_min = {
        "Uber": 2.5,
        "Ola": 2.3,
        "Rapido": 1.8
    }

    # Ride type multiplier
    multiplier = {
        "Bike": 0.75,
        "Auto": 0.85,
        "Mini": 0.95,
        "Sedan": 1.10
    }

    # Traffic multiplier
    traffic_multiplier = {
        "Low": 1.00,
        "Medium": 1.08,
        "High": 1.18
    }

    # Demand multiplier
    demand_multiplier = (
        1 + (demand - 5) * 0.04
    )

    results = {}

    for platform in [
        "Uber",
        "Ola",
        "Rapido"
    ]:

        fare = (
            base[platform]
            + distance_km * per_km[platform]
            + duration_min * per_min[platform]
        )

        fare *= multiplier[ride_type]

        fare *= traffic_multiplier[
            traffic
        ]

        fare *= demand_multiplier

        results[platform] = round(
            max(fare, 50),
            2
        )

    return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    fares = estimate_current_fares(
        distance_km=12,
        duration_min=30,
        demand=8,
        traffic="High",
        ride_type="Sedan"
    )

    print("\nEstimated Current Fares")

    for platform, fare in fares.items():

        print(
            f"{platform}: ₹{fare}"
        )