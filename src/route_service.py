import time
import requests
from functools import lru_cache


# ============================================================
# CONFIGURATION
# ============================================================

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

OSRM_URLS = [
    "https://router.project-osrm.org/route/v1/driving",
    "https://routing.openstreetmap.de/routed-car/route/v1/driving"
]

HEADERS = {
    "User-Agent": "RidelyAI/1.0 (academic fare prediction project)"
}


# ============================================================
# GEOCODING
# ============================================================

@lru_cache(maxsize=100)
def geocode_place(place):

    params = {
        "q": f"{place}, Hyderabad, Telangana, India",
        "format": "jsonv2",
        "limit": 1,
        "countrycodes": "in"
    }

    response = requests.get(
        NOMINATIM_URL,
        params=params,
        headers=HEADERS,
        timeout=15
    )

    response.raise_for_status()

    results = response.json()

    if not results:
        raise ValueError(
            f"Could not find location: {place}"
        )

    result = results[0]

    return {
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
        "display_name": result["display_name"]
    }


# ============================================================
# ROUTE USING OSRM
# ============================================================

def get_osrm_route(
    pickup_data,
    drop_data
):

    pickup_lon = pickup_data["longitude"]
    pickup_lat = pickup_data["latitude"]

    drop_lon = drop_data["longitude"]
    drop_lat = drop_data["latitude"]

    coordinates = (
        f"{pickup_lon},{pickup_lat};"
        f"{drop_lon},{drop_lat}"
    )

    params = {
        "overview": "false",
        "steps": "false"
    }

    for server in OSRM_URLS:

        url = f"{server}/{coordinates}"

        try:

            print(
                f"Trying routing server: {server}"
            )

            response = requests.get(
                url,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            if data.get("code") != "Ok":
                continue

            route = data["routes"][0]

            distance_km = (
                route["distance"] / 1000
            )

            duration_min = (
                route["duration"] / 60
            )

            return {
                "distance_km":
                    round(distance_km, 2),

                "duration_min":
                    round(duration_min, 1),

                "source":
                    "OSRM"
            }

        except requests.exceptions.RequestException as e:

            print(
                f"Routing server failed: {e}"
            )

            continue

    return None


# ============================================================
# FALLBACK ESTIMATE
# ============================================================

def fallback_route_estimate(
    pickup_data,
    drop_data
):

    from math import radians, sin, cos, sqrt, atan2

    lat1 = radians(
        pickup_data["latitude"]
    )

    lon1 = radians(
        pickup_data["longitude"]
    )

    lat2 = radians(
        drop_data["latitude"]
    )

    lon2 = radians(
        drop_data["longitude"]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        +
        cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    earth_radius_km = 6371

    straight_distance = (
        earth_radius_km * c
    )

    # Road distance is generally longer
    # than straight-line distance.
    road_distance = (
        straight_distance * 1.25
    )

    # Approximate urban driving speed.
    average_speed = 30

    duration_hours = (
        road_distance / average_speed
    )

    duration_min = (
        duration_hours * 60
    )

    return {
        "distance_km":
            round(road_distance, 2),

        "duration_min":
            round(duration_min, 1),

        "source":
            "Fallback Estimate"
    }


# ============================================================
# COMPLETE ROUTE CALCULATION
# ============================================================

@lru_cache(maxsize=100)
def calculate_route(
    pickup,
    drop
):

    pickup_data = geocode_place(
        pickup
    )

    # Nominatim public service should not
    # be queried rapidly.
    time.sleep(1.1)

    drop_data = geocode_place(
        drop
    )

    route = get_osrm_route(
        pickup_data,
        drop_data
    )

    if route is None:

        print(
            "\nOSRM unavailable."
        )

        print(
            "Using fallback route estimate."
        )

        route = fallback_route_estimate(
            pickup_data,
            drop_data
        )

    return {

        "pickup":
            pickup_data,

        "drop":
            drop_data,

        "distance_km":
            route["distance_km"],

        "duration_min":
            route["duration_min"],

        "source":
            route["source"]
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pickup = "Gachibowli"

    drop = "Hitech City"

    print(
        "\nCalculating route..."
    )

    print(
        f"Pickup: {pickup}"
    )

    print(
        f"Drop: {drop}"
    )

    try:

        result = calculate_route(
            pickup,
            drop
        )

        print(
            "\nRoute Result"
        )

        print(
            "-" * 40
        )

        print(
            f"Distance: "
            f"{result['distance_km']} km"
        )

        print(
            f"Duration: "
            f"{result['duration_min']} minutes"
        )

        print(
            f"Source: "
            f"{result['source']}"
        )

        print(
            f"\nPickup coordinates: "
            f"{result['pickup']['latitude']}, "
            f"{result['pickup']['longitude']}"
        )

        print(
            f"Drop coordinates: "
            f"{result['drop']['latitude']}, "
            f"{result['drop']['longitude']}"
        )

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(e)