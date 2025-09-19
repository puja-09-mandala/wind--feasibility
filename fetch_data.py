from datetime import datetime
from meteostat import Stations, Hourly
import pandas as pd

def fetch_hourly_by_coords(lat, lon, start, end):
    stations = Stations()
    station = stations.nearby(lat, lon).fetch(1)
    st_id = station.index[0]
    data = Hourly(st_id, start, end).fetch()
    return data[['wspd', 'wdir', 'temp']].reset_index()

if __name__ == "__main__":
    start = datetime(2023, 1, 1)
    end   = datetime(2023, 12, 31)
    df = fetch_hourly_by_coords(12.9716, 77.5946, start, end)  # Example: Bangalore
    df.to_csv("data/bangalore_hourly.csv", index=False)
    print("✅ Data saved to data/bangalore_hourly.csv")
