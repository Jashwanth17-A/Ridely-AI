import os
import numpy as np
import pandas as pd

np.random.seed(42)

OUT = "data/raw/ridely_historical.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

routes = [
    ("LB Nagar", "Gachibowli", 18.0),
    ("Miyapur", "Hitech City", 9.5),
    ("Kukatpally", "Secunderabad", 16.0),
    ("Mehdipatnam", "Madhapur", 10.0),
    ("Dilsukhnagar", "Begumpet", 12.0),
    ("Uppal", "Hitech City", 14.0),
    ("Ameerpet", "Gachibowli", 13.0),
    ("Kondapur", "Secunderabad", 19.0),
    ("Charminar", "Banjara Hills", 8.5),
    ("Kompally", "Hitech City", 25.0),
]

ride_types = {
    "Bike": (35, 7.0),
    "Auto": (55, 10.0),
    "Mini": (80, 13.0),
    "Sedan": (110, 16.0),
}
platform_factor = {"Uber": 1.04, "Ola": 1.00, "Rapido": 0.93}
weather_effect = {"Sunny": 0, "Cloudy": 5, "Rainy": 25}
traffic_effect = {"Low": 0, "Medium": 18, "High": 40}

# 20,000 rides across chronological dates.
n = 20000
start = pd.Timestamp("2026-01-01 00:00:00")
times = pd.date_range(start, periods=n, freq="15min")

rows = []

for i, ts in enumerate(times):
    pickup, drop, base_distance = routes[i % len(routes)]
    distance = max(1.0, np.random.normal(base_distance, base_distance * 0.08))

    hour = ts.hour
    weekday = ts.dayofweek
    weekend = int(weekday >= 5)

    peak = int((7 <= hour <= 10) or (17 <= hour <= 21))
    night = int(hour >= 22 or hour < 6)

    if peak:
        traffic_probs = [0.10, 0.35, 0.55]
        traffic = np.random.choice(["Low", "Medium", "High"], p=traffic_probs)
    else:
        traffic = np.random.choice(
            ["Low", "Medium", "High"], p=[0.45, 0.40, 0.15]
        )

    weather = np.random.choice(
        ["Sunny", "Cloudy", "Rainy"], p=[0.60, 0.28, 0.12]
    )

    demand = 3.5 + 4.0 * peak + 1.5 * weekend + np.random.normal(0, 1.2)
    demand = float(np.clip(demand, 1, 10))

    availability = 8.0 - 2.8 * peak + np.random.normal(0, 1.0)
    availability = float(np.clip(availability, 1, 10))

    duration = distance * (2.2 + 0.55 * {"Low": 0, "Medium": 1, "High": 2}[traffic])
    duration += np.random.normal(0, 3)
    duration = max(5, duration)

    ride_type = np.random.choice(list(ride_types), p=[0.20, 0.25, 0.30, 0.25])
    base_fare, per_km = ride_types[ride_type]

    holiday = int(np.random.random() < 0.025)
    demand_ratio = demand / availability

    # Common latent market pressure creates correlated platform prices.
    market_pressure = (
        1
        + 0.025 * (demand - 5)
        + 0.018 * demand_ratio
        + 0.035 * peak
        + 0.025 * (weather == "Rainy")
    )

    fares = {}
    for platform in platform_factor:
        raw = (
            base_fare
            + distance * per_km
            + duration * 0.8
            + traffic_effect[traffic]
            + weather_effect[weather]
            + 6 * demand
            + 18 * holiday
        )
        noise = np.random.normal(0, 8)
        fare = raw * platform_factor[platform] * market_pressure + noise
        fares[platform] = max(40, round(fare, 2))

    # Future fares are generated from a plausible changing market process.
    # These are synthetic targets, not real platform prices.
    future = {}
    for platform in fares:
        current = fares[platform]
        values = []
        for step in range(1, 5):
            drift = (
                2.5 * peak
                + 1.4 * (demand - 5)
                - 1.2 * (availability - 5)
                + (5 if weather == "Rainy" else 0)
            )
            mean_reversion = (current - (base_fare + distance * per_km)) * 0.06
            shock = np.random.normal(0, 7)
            next_fare = current + drift - mean_reversion + shock
            current = max(40, next_fare)
            values.append(round(current, 2))
        future[platform] = values

    rows.append({
        "Ride_ID": f"R{i+1:05d}",
        "Date": ts.date().isoformat(),
        "Time": ts.strftime("%H:%M"),
        "Pickup": pickup,
        "Drop": drop,
        "Distance_km": round(distance, 2),
        "Duration_min": round(duration, 2),
        "Traffic": traffic,
        "Weather": weather,
        "Ride_Type": ride_type,
        "Demand_Index": round(demand, 2),
        "Driver_Availability": round(availability, 2),
        "Weekend": weekend,
        "Peak_Hour": peak,
        "Holiday": holiday,
        "Uber_Fare": fares["Uber"],
        "Ola_Fare": fares["Ola"],
        "Rapido_Fare": fares["Rapido"],
        "Uber_Future_15": future["Uber"][0],
        "Uber_Future_30": future["Uber"][1],
        "Uber_Future_45": future["Uber"][2],
        "Uber_Future_60": future["Uber"][3],
        "Ola_Future_15": future["Ola"][0],
        "Ola_Future_30": future["Ola"][1],
        "Ola_Future_45": future["Ola"][2],
        "Ola_Future_60": future["Ola"][3],
        "Rapido_Future_15": future["Rapido"][0],
        "Rapido_Future_30": future["Rapido"][1],
        "Rapido_Future_45": future["Rapido"][2],
        "Rapido_Future_60": future["Rapido"][3],
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

print(f"Created {OUT}")
print(f"Rows: {len(df):,}")
print(df.head())
