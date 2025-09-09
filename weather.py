import requests
import datetime

def get_seasonal_forecast(lat, lon, start_month, end_month):
    today = datetime.date.today()
    year = today.year

    # Handle crop seasons that cross over the year-end (e.g. Nov–Apr)
    if start_month > end_month:
        start_date = datetime.date(year, start_month, 1)
        end_date = datetime.date(year + 1, end_month, 28)
    else:
        start_date = datetime.date(year, start_month, 1)
        end_date = datetime.date(year, end_month, 28)

    # ✅ Climate API instead of forecast
    url = (
        f"https://climate-api.open-meteo.com/v1/climate?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        "&timezone=auto"
    )

    res = requests.get(url).json()
    print("DEBUG URL:", url)       # Debugging line
    print("DEBUG RESPONSE:", res)  # Debugging line

    if "daily" not in res:
        return {"error": "No daily data found", "response": res}

    temps = res["daily"]["temperature_2m_max"]
    rainfall = res["daily"]["precipitation_sum"]

    return {
        "avg_temp": round(sum(temps) / len(temps)),
        "total_rainfall": round(sum(rainfall))
    }
