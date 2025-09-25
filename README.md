# Calendar heatmap_ % returns

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib

cryptos = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "SUIUSDT"]
interval = "1d"
limit = 1000

def get_binance_klines(symbol, interval, start_str, end_str=None, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": start_str}
    if end_str:
        params["endTime"] = end_str
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_full_history(symbol, interval="1d", years=5):
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now().timestamp() - years*365*24*3600) * 1000)
    while start_time < end_time:
        data = get_binance_klines(symbol, interval, start_time, limit=limit)
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][0] + 1
    return all_data

def analyze_crypto(symbol, years=5, show_avg_row=False):
    # Fetch daily history
    data = fetch_full_history(symbol, "1d", years=years)
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(data, columns=cols)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("date", inplace=True)
    df = df.sort_index()

    # Only complete months (month-end stamps)
    month_end = df.index.to_period("M").to_timestamp("M")
    df_complete = df[df.index == month_end]

    # Month-end returns
    monthly_returns = df_complete["close"].resample("M").last().pct_change() * 100
    monthly_table = monthly_returns.to_frame(name="Return")
    monthly_table["Year"] = monthly_table.index.year
    monthly_table["Month"] = monthly_table.index.strftime("%b")

    order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot_table = monthly_table.pivot(index="Year", columns="Month", values="Return").reindex(columns=order)

    # Optional averages row
    if show_avg_row and not pivot_table.empty:
        avg_row = pivot_table.mean(axis=0, skipna=True).to_frame().T
        avg_row.index = ["Avg"]
        pivot_table = pd.concat([pivot_table, avg_row], axis=0)

    # Heatmap numeric data
    data_num = pivot_table.round(1)
    data_vals = data_num.to_numpy()

    # Colormap (no deprecation)
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")  # replaces deprecated get_cmap [web:45]

    # Draw heatmap WITHOUT seaborn's built-in annot
    plt.figure(figsize=(14, 7))
    ax = sns.heatmap(
        data_num,
        annot=False,          # turn off built-in annot to avoid notebook bug [web:23]
        cmap=cmap,
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Monthly Return %"}
    )

    # Manually draw text labels per cell (robust across backends)
    # Compute symmetric normalization for luminance-based text color
    if np.isfinite(data_vals).any():
        vmin = np.nanmin(data_vals); vmax = np.nanmax(data_vals)
        vr = max(abs(vmin), abs(vmax))
        norm = matplotlib.colors.Normalize(vmin=-vr, vmax=vr)

    nrows, ncols = data_vals.shape
    for r in range(nrows):
        for c in range(ncols):
            v = data_vals[r, c]
            if np.isfinite(v):
                label = f"{v:.1f}%"
                # Choose black/white for contrast vs background color
                r_, g_, b_, _ = cmap(norm(v)) if np.isfinite(v) else (1,1,1,1)
                lum = 0.299*r_ + 0.587*g_ + 0.114*b_
                color = "black" if lum > 0.6 else "white"
                ax.text(c + 0.5, r + 0.5, label,
                        ha="center", va="center",
                        color=color, fontsize=10, fontweight="bold")  # visible text [web:3]

    plt.title(f"{symbol} Monthly Returns (%) - Last {years} Years", fontsize=16, weight="bold")
    plt.ylabel("Year")
    plt.xlabel("Month")
    plt.tight_layout()
    plt.show()

    # Stats per calendar month
    print(f"\n{symbol} Monthly Return Statistics (Last {years} Years)\n")
    print(monthly_table.groupby("Month")["Return"].agg(["mean","median","std"]).reindex(order).round(2))

# Run all
for c in cryptos:
    analyze_crypto(c, years=5, show_avg_row=False)
