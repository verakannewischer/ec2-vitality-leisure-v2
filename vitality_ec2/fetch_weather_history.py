"""
fetch_weather_history.py
Run ONCE locally before retraining:
    python3 fetch_weather_history.py

Fetches daily temperature (°C) and precipitation (mm) for 2008–2025
from the Open-Meteo archive API and saves to weather_history.csv.
"""

import requests
import pandas as pd
from datetime import date

LAT, LON = 52.0833, 8.75  # Leisure City

def fetch_chunk(start: str, end: str) -> pd.DataFrame:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start}&end_date={end}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode"
        "&timezone=Europe%2FBerlin"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(d)
    df["date"] = pd.to_datetime(df["time"])
    df["temp_c"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
    df["precip_mm"] = df["precipitation_sum"].fillna(0)
    return df[["date", "temp_c", "precip_mm", "weathercode"]]

# Fetch in yearly chunks to avoid timeout
chunks = []
for year in range(2008, 2026):
    start = f"{year}-01-01"
    end   = f"{year}-12-31" if year < 2025 else "2025-11-23"
    print(f"Fetching {year}...")
    try:
        chunks.append(fetch_chunk(start, end))
    except Exception as e:
        print(f"  Error for {year}: {e}")

weather_df = pd.concat(chunks).reset_index(drop=True)
weather_df.to_csv("weather_history.csv", index=False)
print(f"\nSaved weather_history.csv — {len(weather_df)} rows")
print(weather_df.head(3))
