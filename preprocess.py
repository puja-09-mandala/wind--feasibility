import pandas as pd

def clean_and_resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort, set time index, resample hourly, and fill gaps.
    """
    df = df.sort_values('time').set_index('time')
    # Ensure hourly frequency
    df = df.resample('1H').mean()
    # Fill missing values
    df['wspd'] = df['wspd'].interpolate(limit=6)  # fill gaps up to 6 hours
    return df.reset_index()

if __name__ == "__main__":
    # Load the data saved earlier
    df = pd.read_csv("data/bangalore_hourly.csv", parse_dates=['time'])
    df = clean_and_resample_hourly(df)
    df.to_csv("data/bangalore_hourly_clean.csv", index=False)
    print("✅ Cleaned data saved to data/bangalore_hourly_clean.csv")
