import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("/Users/ramko/Downloads/Nat_Gas.csv")
df['Date'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df.set_index('Date', inplace=True)
monthly = df['Prices'].asfreq('ME')

decompose = seasonal_decompose(monthly, model='additive', period=12, two_sided=False)
trend = decompose.trend.dropna()
seasonal = decompose.seasonal

origin = trend.index.min()
x_trend = (trend.index - origin).days.values
y_trend = trend.values

smoothing = 0.1 * np.var(y_trend) * len(y_trend)
spline = UnivariateSpline(x_trend, y_trend, k=3, s=smoothing)

seasonal_extended = seasonal.copy()
seasonal_extended = seasonal_extended.asfreq('D')
seasonal_daily = seasonal_extended.interpolate(method='cubic')

loop_start = seasonal_daily.index.min()
loop_end = seasonal_daily.index.max()
loop_days = pd.date_range(start=loop_end + pd.Timedelta(days=1), periods=365, freq='D')
repeating_values = seasonal_daily.loc[seasonal_daily.index[-365:]].values
seasonal_daily = pd.concat([
    seasonal_daily,
    pd.Series(repeating_values, index=loop_days)
])

def predict_price(date):
    """
    Estimate the price on any date by summing:
      • spline-trend(date)
      • seasonal_index[month]
    """
    dt = pd.to_datetime(date)
    t = (dt - origin).days
    trend_est = float(spline(t))
    if dt in seasonal_daily.index:
        season_est = seasonal_daily[dt]
    else:
        offset = (dt - seasonal_daily.index[0]).days
        season_est = seasonal_daily.iloc[offset % len(seasonal_daily)]
    return trend_est + season_est

start, end = monthly.index.min(), monthly.index.max() + pd.DateOffset(years=1)
all_days = pd.date_range(start, end, freq='D')
prediction = [predict_price(d) for d in all_days]

plt.figure(figsize=(12, 6))
plt.plot(monthly.index, monthly.values, 'o', label='Observed (month-end)')
plt.plot(all_days, prediction, '-', label='Model fit & extrapolation')
plt.axvline(monthly.index.max(), color='red', linestyle='--', label='Forecast start')
plt.title("Natural Gas Price: Trend + Seasonality Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()
