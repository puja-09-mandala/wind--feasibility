import pandas as pd

def power_from_wind(v, rated_kw, v_cutin=3.5, v_rated=11, v_cutout=25):
    """Power output of a wind turbine at speed v (m/s)."""
    if v < v_cutin or v >= v_cutout:
        return 0.0
    if v_cutin <= v < v_rated:
        return rated_kw * ((v - v_cutin) / (v_rated - v_cutin))**3
    return rated_kw

def compute_energy(df, rated_kw=100):
    df['power_kw'] = df['wspd'].apply(lambda v: power_from_wind(v, rated_kw))
    annual_energy_kwh = df['power_kw'].sum()
    return annual_energy_kwh, df

if __name__ == "__main__":
    df = pd.read_csv("data/bangalore_hourly_clean.csv")
    energy, df = compute_energy(df, rated_kw=100)
    print(f"✅ Estimated annual energy: {energy:.2f} kWh")
