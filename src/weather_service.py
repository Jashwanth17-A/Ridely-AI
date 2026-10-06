import requests


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# WEATHER CODE → PROJECT WEATHER CATEGORY
# ============================================================

def weather_code_to_category(code):

    code = int(code)

    # Clear
    if code == 0:
        return "Sunny"

    # Mainly clear / partly cloudy
    if code in [1, 2]:
        return "Cloudy"

    # Overcast
    if code == 3:
        return "Cloudy"

    # Fog
    if code in [45, 48]:
        return "Cloudy"

    # Drizzle / Rain
    if code in [
        51, 53, 55,
        56, 57,
        61, 63, 65,
        66, 67,
        80, 81, 82
    ]:
        return "Rainy"

    # Snow
    if code in [71, 73, 75, 77, 85, 86]:
        return "Rainy"

    # Thunderstorm
    if code in [95, 96, 99]:
        return "Rainy"

    return "Cloudy"


# ============================================================
# GET CURRENT WEATHER
# ============================================================

def get_weather(latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,precipitation,weather_code",
        "timezone": "auto"
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    current = data["current"]

    weather_code = current["weather_code"]

    category = weather_code_to_category(
        weather_code
    )

    return {
        "temperature": current["temperature_2m"],
        "precipitation": current["precipitation"],
        "weather_code": weather_code,
        "category": category
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Example: Hyderabad
    latitude = 17.4065
    longitude = 78.4772

    result = get_weather(
        latitude,
        longitude
    )

    print("\nWeather")
    print("-" * 40)

    print(
        f"Temperature: "
        f"{result['temperature']} °C"
    )

    print(
        f"Precipitation: "
        f"{result['precipitation']} mm"
    )

    print(
        f"Weather: "
        f"{result['category']}"
    )