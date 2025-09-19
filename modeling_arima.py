import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(series, steps=168, order=(5,1,0)):
    """
    ARIMA forecast for given series.
    steps=168 → 7 days hourly forecast
    """
    model = ARIMA(series, order=order)
    fit = model.fit()
    forecast = fit.get_forecast(steps=steps)
    mean = forecast.predicted_mean
    conf = forecast.conf_int()
    return mean, conf

if __name__ == "__main__":
    df = pd.read_csv("data/bangalore_hourly_clean.csv", parse_dates=['time'])
    series = df.set_index('time')['wspd']
    mean, conf = arima_forecast(series, steps=168)
    print("✅ Forecasted next 168 hours:")
    print(mean.head())
