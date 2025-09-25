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

    ## Save to Excel

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

def analyze_crypto(symbol, writer, years=5, show_avg_row=False):
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

    # ---------------- Heatmap with readable labels ----------------
    data_num = pivot_table.round(1)
    data_vals = data_num.to_numpy()
    cmap = matplotlib.colormaps.get_cmap("RdYlGn")

    plt.figure(figsize=(14, 7))
    ax = sns.heatmap(
        data_num,
        annot=False,
        cmap=cmap,
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Monthly Return %"}
    )

    # Ensure readable text in each cell
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
                r_, g_, b_, _ = cmap(norm(v)) if np.isfinite(v) else (1,1,1,1)
                lum = 0.299*r_ + 0.587*g_ + 0.114*b_
                color = "black" if lum > 0.6 else "white"
                ax.text(c + 0.5, r + 0.5, label,
                        ha="center", va="center",
                        color=color, fontsize=9.5, fontweight="bold")

    plt.title(f"{symbol} Monthly Returns (%) - Last {years} Years", fontsize=16, weight="bold")
    plt.ylabel("Year")
    plt.xlabel("Month")
    plt.tight_layout()
    plt.show()

    # ---------------- Monthly Statistics ----------------
    monthly_stats = monthly_table.groupby("Month")["Return"].agg(["mean","median","std"]).reindex(order).round(2)

    print(f"\n{symbol} Monthly Return Statistics (Last {years} Years)\n")
    print(monthly_stats)

    # ---------------- Save to Excel ----------------
    # Write pivot and stats each under same sheet
    pivot_table.to_excel(writer, sheet_name=symbol, startrow=0)
    monthly_stats.to_excel(writer, sheet_name=symbol, startrow=pivot_table.shape[0] + 3)

# ---------------- Master Run ----------------
excel_file = "Crypto_Monthly_Returns.xlsx"
with pd.ExcelWriter(excel_file, engine="xlsxwriter") as writer:
    for c in cryptos:
        analyze_crypto(c, writer, years=5, show_avg_row=False)

print(f"\n✅ All results saved to {excel_file}")

## Run bell-sahpe distrubition with 1 SD on price & monthly returns in bar diagram

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import norm
import matplotlib

sns.set(style="whitegrid")

# ---------- Parameters ----------
symbol = "BTCUSDT"
interval = "1d"
years = 5
limit = 1000

# ---------- Helpers ----------
def get_binance_klines(symbol, interval, start_ms, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit, "startTime": start_ms}
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

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

# ---------- Fetch BTC data ----------
print("Fetching BTC history (this may take a few seconds)...")
data = fetch_full_history(symbol, interval, years=years)

cols = ["open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"]
df = pd.DataFrame(data, columns=cols)
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["date"] = pd.to_datetime(df["open_time"], unit="ms")
df.set_index("date", inplace=True)
df.sort_index(inplace=True)

# ---------- Monthly returns ----------
# Take month-end close points (ensures complete months)
month_end = df.index.to_period("M").to_timestamp("M")
df_complete = df[df.index == month_end]

monthly_returns = df_complete["close"].resample("M").last().pct_change() * 100
monthly_table = monthly_returns.to_frame(name="Return").dropna()
monthly_table["Month"] = monthly_table.index.strftime("%b")
order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------- 1) Professional Average Monthly Bar Chart ----------
monthly_mean = monthly_table.groupby("Month")["Return"].mean().reindex(order)

# Create color shades based on magnitude
pos_vals = monthly_mean[monthly_mean > 0].abs()
neg_vals = monthly_mean[monthly_mean < 0].abs()
max_pos = pos_vals.max() if not pos_vals.empty else 1.0
max_neg = neg_vals.max() if not neg_vals.empty else 1.0

colors = []
for v in monthly_mean.values:
    if v >= 0:
        # Greens colormap: lighter -> darker by magnitude
        intensity = 0.35 + 0.65 * (abs(v) / max_pos) if max_pos != 0 else 0.5
        colors.append(matplotlib.cm.Greens(intensity))
    else:
        intensity = 0.35 + 0.65 * (abs(v) / max_neg) if max_neg != 0 else 0.5
        colors.append(matplotlib.cm.Reds(intensity))

plt.figure(figsize=(12,6))
bars = plt.bar(monthly_mean.index, monthly_mean.values, color=colors, edgecolor="black", linewidth=0.6)
plt.axhline(0, color="black", linewidth=1)

# Labels on top (positive above, negative below)
for bar, val in zip(bars, monthly_mean.values):
    y = val
    # offset a bit so labels don't overlap the bar edge
    offset = 0.4 if val >= 0 else -0.4
    va = "bottom" if val >= 0 else "top"
    plt.text(bar.get_x() + bar.get_width()/2, y + offset, f"{val:.2f}%", ha="center", va=va,
             fontsize=10, fontweight="bold", color="black")

plt.title("BTC — Average Monthly Return (%) (Last 5 years)", fontsize=16, weight="bold")
plt.ylabel("Average Return (%)", fontsize=12)
plt.xlabel("Month", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ---------- 2) Bell Curve for October — price-based, centered at today's price ----------
oct_returns = monthly_table[monthly_table["Month"] == "Oct"]["Return"].dropna()

if oct_returns.empty:
    print("Not enough October data to build distribution.")
else:
    mean_oct = oct_returns.mean()
    std_oct = oct_returns.std(ddof=1)  # sample std

    today_price = df["close"].iloc[-1]
    sigma_price = today_price * (std_oct / 100.0)     # 1σ in price units
    # center at today's price (user requested), ±1σ = today ± sigma_price
    price_center = today_price
    price_low = price_center - sigma_price
    price_high = price_center + sigma_price

    # Build normal pdf in PRICE domain with mean=today_price and sd=sigma_price
    x = np.linspace(price_center - 4*sigma_price, price_center + 4*sigma_price, 500)
    y = norm.pdf(x, loc=price_center, scale=sigma_price)

    plt.figure(figsize=(12,6))
    plt.plot(x, y, color="#225ea8", lw=2)
    plt.fill_between(x, y, where=(x >= price_low) & (x <= price_high), color="#99d8c9", alpha=0.6,
                     label="±1σ price range (~68%)")

    # vertical lines for mean and ±1σ (price terms)
    plt.axvline(price_center, color="black", linestyle="--", lw=1.5, label=f"Today: {price_center:,.2f} USD")
    plt.axvline(price_low, color="#de2d26", linestyle="--", lw=1.2, label=f"-1σ: {price_low:,.0f} USD ({-std_oct:.2f}%)")
    plt.axvline(price_high, color="#31a354", linestyle="--", lw=1.2, label=f"+1σ: {price_high:,.0f} USD ({std_oct:.2f}%)")

    # Annotate sigma percentage and price range on plot
    plt.text(price_center, max(y)*0.95,
             f"σ (price) = {sigma_price:,.2f} USD  ≈  {std_oct:.2f}% of spot\n±1σ price range: {price_low:,.0f} — {price_high:,.0f} USD\nProbability ≈ 68.27%",
             ha="center", va="top", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    plt.title("BTC — October Returns (price distribution mapped to today's spot)", fontsize=14, weight="bold")
    plt.xlabel("BTC Price (USD)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # Console summary
    print("BTC October stats (historical):")
    print(f"  mean return (Oct): {mean_oct:.2f}%")
    print(f"  std dev (Oct):     {std_oct:.2f}%")
    print(f"  today spot price:  {today_price:,.2f} USD")
    print(f"  ±1σ price range:   {price_low:,.2f} — {price_high:,.2f} USD (≈68%)")

