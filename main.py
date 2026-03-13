import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from decomp_cache import decompose_mstl_cached


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Coerce to numeric
    for c in ["year", "day", "hour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["VOLUME_KWH"] = pd.to_numeric(df["VOLUME_KWH"], errors="coerce")

    # Drop invalid
    df = df.dropna(subset=["year", "day", "hour", "VOLUME_KWH"])
    df["year"] = df["year"].astype(int)
    df["day"] = df["day"].astype(int)
    df["hour"] = df["hour"].astype(int)

    df = df[df["day"].between(1, 366) & df["hour"].between(0, 23)]

    # Only group if duplicates exist
    if df.duplicated(["year", "day", "hour"]).any():
        df = df.groupby(["year", "day", "hour"], as_index=False)["VOLUME_KWH"].mean()

    # Build timestamp index safely
    base = pd.to_datetime(df["year"].astype(str), format="%Y", errors="coerce")
    df["time"] = base + pd.to_timedelta(df["day"] - 1, unit="D") + pd.to_timedelta(df["hour"], unit="h")
    df = df.dropna(subset=["time"]).sort_values("time").set_index("time")

    # Enforce hourly grid (IMPORTANT: pandas wants "h", not "H" in your setup)
    df = df.asfreq("h")

    # Fill missing values
    df["VOLUME_KWH"] = df["VOLUME_KWH"].interpolate("time")

    return df


def decompose_time_series_two_stage(df: pd.DataFrame):
    """
    Robust + fast alternative to MSTL:
    1) STL daily (24)
    2) STL weekly (168) on remainder after removing daily seasonal
    Returns arrays: trend, seasonal_day, seasonal_week, resid
    """
    y = df["VOLUME_KWH"].sort_index().asfreq("h").interpolate("time")

    if len(y) < 200:  # guardrail
        raise ValueError(f"Time series too short for STL: len(y)={len(y)}. Check cleaning/indexing.")

    # Daily
    r_day = STL(y, period=24, robust=True).fit()
    y_minus_day = y - r_day.seasonal

    # Weekly on remainder
    r_week = STL(y_minus_day, period=168, robust=True).fit()

    seasonal_day = r_day.seasonal.to_numpy()
    seasonal_week = r_week.seasonal.to_numpy()
    trend = r_week.trend.to_numpy()
    resid = r_week.resid.to_numpy()

    return trend, seasonal_day, seasonal_week, resid


def add_stacked_to_existing_axes(ax, df, trend, seasonal_day, seasonal_week, i0=0, i1=None):
    y = df["VOLUME_KWH"].sort_index().asfreq("h")

    if i1 is None:
        i1 = len(y)

    y = y.iloc[i0:i1]

    trend_s = pd.Series(np.asarray(trend)[i0:i1], index=y.index, dtype=float)
    day_s = pd.Series(np.asarray(seasonal_day)[i0:i1], index=y.index, dtype=float)
    week_s = pd.Series(np.asarray(seasonal_week)[i0:i1], index=y.index, dtype=float)

    x = y.index
    # Convert datetime index to matplotlib date numbers for reliable fill_between
    x_num = mdates.date2num(pd.to_datetime(x).to_pydatetime())

    c1 = trend_s.values
    c2 = (trend_s + day_s).values
    c3 = (trend_s + day_s + week_s).values

    # Original line
    ax.plot(x_num, y.values, label="Original", alpha=0.5, zorder=3)

    # Stacked fills
    ax.fill_between(x_num, 0, c1, label="Trend", alpha=0.25, zorder=1)
    ax.fill_between(x_num, c1, c2, label="Daily seasonal", alpha=0.25, zorder=1)
    ax.fill_between(x_num, c2, c3, label="Weekly seasonal", alpha=0.25, zorder=1)

    # Year-only ticks
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(x_num[0], x_num[-1])

    return ax

def add_daily_weekly_only(ax, df, seasonal_day, seasonal_week, i0=0, i1=None, plot_original=True):
    """
    Adds ONLY daily + weekly seasonal components as stacked fills to an existing Axes.
    - Does NOT call plt.show()
    - x-limits controlled by i0/i1 (integer iloc slice into df)
    - Year-only x ticks
    """
    y = df["VOLUME_KWH"].sort_index().asfreq("h")

    if i1 is None:
        i1 = len(y)

    y = y.iloc[i0:i1]

    day_s  = pd.Series(np.asarray(seasonal_day)[i0:i1], index=y.index, dtype=float)
    week_s = pd.Series(np.asarray(seasonal_week)[i0:i1], index=y.index, dtype=float)

    x = y.index
    x_num = mdates.date2num(pd.to_datetime(x).to_pydatetime())

    c1 = day_s.values
    c2 = (day_s + week_s).values

    if plot_original:
        ax.plot(x_num, y.values, label="Original", alpha=0.5, zorder=3)

    ax.fill_between(x_num, 0,  c1, label="Daily seasonal", alpha=0.25, zorder=1)
    ax.fill_between(x_num, c1, c2, label="Weekly seasonal", alpha=0.25, zorder=1)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(x_num[0], x_num[-1])

    return ax

def main():
    df = pd.read_csv("processed_data.csv")
    # df = clean_data(df)

    # Build your hourly series (make sure df is indexed by time before this)
    y = df["VOLUME_KWH"]
    # If you already have df indexed by time, great.
    # Otherwise: y needs a DatetimeIndex, not year/day/hour columns.

    seasonal, trend, resid, meta = decompose_mstl_cached(
        y,
        periods=[24, 168],
        cache_dir="cache",
    )

    seasonal_day = seasonal[:, 0]
    seasonal_week = seasonal[:, 1]

    print("Loaded/computed MSTL:", meta)

    # Plot example
    plt.figure(figsize=(12, 6))
    plt.plot(y.index, y.values, alpha=0.5, label="Original")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()