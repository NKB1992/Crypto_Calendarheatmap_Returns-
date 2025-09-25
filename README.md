# Calendar heatmap_ % returns

import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

# Binance API endpoint
BASE_URL = "https://api.binance.com/api/v3/klines"

# Parameters
symbol = "BTCUSDT"   # Bitcoin in USDT
interval = "1d"      # Daily candles
limit = 1000         # Max rows per request (Binance limits this)

# Function to fetch historical daily data
def fetch_binance_data(symbol, interval, start_str, end_str=None):
    url = BASE_URL
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(pd.to_datetime(start_str).timestamp() * 1000),
        "limit": limit
    }
    if end_str:
        params["endTime"] = int(pd.to_datetime(end_str).timestamp() * 1000)
    r = requests.get(url, params=params)
    data = r.json()
    if "code" in data:
        raise Exception(f"Error fetching data: {data}")
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close"] = df["close"].astype(float)
    df.set_index("date", inplace=True)
    return df[["close"]]

# Fetch 10 years of BTC data
start_date = "2015-01-01"
today = datetime.today().strftime("%Y-%m-%d")

btc_df = fetch_binance_data(symbol, "1d", start_date, today)

# Resample to monthly and calculate returns
monthly_prices = btc_df["close"].resample("M").last()
monthly_returns = monthly_prices.pct_change().dropna() * 100

# Create interactive bar chart
fig = px.bar(
    monthly_returns,
    x=monthly_returns.index,
    y=monthly_returns.values,
    title="📈 Bitcoin Monthly Returns (Last 10 Years)",
    labels={"x": "Date", "y": "Monthly Return (%)"},
    color=monthly_returns.values,
    color_continuous_scale="RdYlGn"
)

fig.update_layout(
    xaxis=dict(showgrid=False, rangeslider=dict(visible=True)),
    yaxis=dict(showgrid=True, title="Return (%)"),
    template="plotly_dark",
    title_x=0.5,
    height=600
)

fig.show()
