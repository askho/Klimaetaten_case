# plotting_current.py
from __future__ import annotations

from typing import Optional, Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import ewma

def _parse_start_timestamp_from_tid(tid: pd.Series) -> pd.Series:
    """
    Parse start timestamp from strings like '01.01.2025 00:00 - 01:00'.
    Returns a pandas Series of datetimes (NaT for failures).
    """
    s = tid.astype(str).str.strip()
    start = s.str.split(" - ", n=1, expand=True)[0].str.strip()

    dt = pd.to_datetime(start, format="%d.%m.%Y %H:%M", errors="coerce")
    if dt.isna().any():
        dt2 = pd.to_datetime(start, dayfirst=True, errors="coerce")
        dt = dt.fillna(dt2)
    return dt

def plot_flex_diagnostics(
    *,
    df_main: pd.DataFrame,
    P_avg: np.ndarray,
    P_peak: np.ndarray,
    T_out: np.ndarray,
    P_star: np.ndarray,
    P_temp: np.ndarray,
    other_avg: Optional[Dict[str, Any]] = None,
    q: float = 0.99,
    top_n_peaks: int = 48,
    periods: Optional[List[Tuple[str, str]]] = None,
    title_prefix: str = "Flex diagnostics",
    show: bool = True,
) -> dict[str, plt.Figure]:
    """
    Works with your current implementation:
      - df_main is the raw dataframe returned by fetch_and_clean_df (contains 'Tid (Time)' OR datetime index)
      - P_avg/P_peak/T_out/P_star/P_temp are numpy arrays computed from df_main columns in row-order
      - other_avg can be:
          * dict[str, pd.Series] with DatetimeIndex
          * dict[str, np.ndarray] in the same row-order as df_main
    Produces:
      1) overlay plots for selected periods
      2) stacked attribution for top peak hours
      3) scatter P* vs HDH (using EWMA smoothed outdoor temp)
    """

    # --- Build a DatetimeIndex and a consistent keep-mask ---
    if isinstance(df_main.index, pd.DatetimeIndex):
        dt = pd.Series(df_main.index)
    elif "Tid (Time)" in df_main.columns:
        dt = _parse_start_timestamp_from_tid(df_main["Tid (Time)"])
    else:
        raise ValueError("df_main must have a DatetimeIndex or contain a 'Tid (Time)' column.")

    # Keep valid & unique timestamps, then apply the same mask to all arrays
    keep = dt.notna() & (~dt.duplicated(keep="first"))
    keep_np = keep.to_numpy()

    idx = pd.DatetimeIndex(dt[keep].to_numpy())

    def _to_series(arr: np.ndarray, name: str) -> pd.Series:
        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] != keep_np.shape[0]:
            raise ValueError(f"{name} length {arr.shape[0]} does not match df_main length {keep_np.shape[0]}.")
        return pd.Series(arr[keep_np], index=idx, name=name)

    P_avg_s = _to_series(P_avg, "P_avg")
    P_peak_s = _to_series(P_peak, "P_peak")
    T_out_s = _to_series(T_out, "T_out")
    P_star_s = _to_series(P_star, "P_star")
    P_temp_s = _to_series(P_temp, "P_temp")

    # --- Prepare other meters for plotting (align to idx) ---
    other_plot: Dict[str, pd.Series] = {}
    if other_avg:
        for name, s in other_avg.items():
            if isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex):
                other_plot[name] = s.reindex(idx).astype(float)
            else:
                arr = np.asarray(s, dtype=float)
                if arr.shape[0] == keep_np.shape[0]:
                    other_plot[name] = pd.Series(arr[keep_np], index=idx, name=str(name))
                else:
                    # last resort: trim to min length
                    n = min(arr.shape[0], len(idx))
                    other_plot[name] = pd.Series(arr[:n], index=idx[:n], name=str(name))

    # --- Auto periods if none provided: first 7 days + last 7 days ---
    if periods is None:
        start = idx.min()
        end = start + pd.Timedelta(days=7)
        end2 = idx.max()
        start2 = end2 - pd.Timedelta(days=7)
        periods = [(start.isoformat(), end.isoformat()), (start2.isoformat(), end2.isoformat())]

    figs: dict[str, plt.Figure] = {}

    # ========= 1) Overlay plots =========
    overlay_figs = []
    for i, (start, end) in enumerate(periods, 1):
        w = slice(pd.to_datetime(start), pd.to_datetime(end))
        fig = plt.figure(figsize=(14, 5))
        plt.plot(P_avg_s.loc[w].index, P_avg_s.loc[w].to_numpy(), label="P_avg (kW)")
        plt.plot(P_peak_s.loc[w].index, P_peak_s.loc[w].to_numpy(), label="P_peak (kW)")
        plt.plot(P_temp_s.loc[w].index, P_temp_s.loc[w].to_numpy(), label="P_temp (kW)")
        plt.title(f"{title_prefix} – Overlay {i}: {start} → {end}")
        plt.xlabel("Time")
        plt.ylabel("kW")
        plt.legend()
        plt.tight_layout()
        overlay_figs.append(fig)
    figs["overlay"] = overlay_figs[-1]  # keep last handle; all are shown anyway

    # ========= 2) Peak-hour attribution stacked =========
    if other_plot:
        df_stack = pd.DataFrame({"total": P_avg_s, "peak_ref": P_peak_s, **other_plot}).dropna().sort_index()
        thr = float(df_stack["peak_ref"].quantile(q))
        peak_df = df_stack.loc[df_stack["peak_ref"] >= thr].sort_values("peak_ref", ascending=False).head(top_n_peaks)

        meter_names = list(other_plot.keys())
        known_sum = peak_df[meter_names].sum(axis=1)
        peak_df["UNACCOUNTED_OTHER"] = (peak_df["total"] - known_sum).clip(lower=0.0)

        fig = plt.figure(figsize=(14, 6))
        x = np.arange(len(peak_df))
        bottom = np.zeros(len(peak_df), dtype=float)

        for name in meter_names + ["UNACCOUNTED_OTHER"]:
            y = peak_df[name].to_numpy(dtype=float)
            plt.bar(x, y, bottom=bottom, label=name)
            bottom += y

        plt.xticks(x, [ts.strftime("%Y-%m-%d %H:%M") for ts in peak_df.index], rotation=90)
        plt.title(f"{title_prefix} – Peak attribution (stacked)\nq={q}, threshold={thr:.3f}, top {len(peak_df)} hours")
        plt.xlabel("Peak hours (sorted by P_peak)")
        plt.ylabel("kW (avg)")
        plt.legend(ncols=2, fontsize=9)
        plt.tight_layout()
        figs["peak_attribution"] = fig

    # ========= 3) Scatter: P* vs HDH (EWMA temp) =========
    # EWMA smoothing (simple thermal inertia)
    alpha = 0.2
    T_smooth = ewma(T_out_s.to_numpy(dtype=float), alpha=alpha)

    T_b = 18.0
    HDH = np.maximum(0.0, T_b - T_smooth)

    fig = plt.figure(figsize=(7, 6))
    plt.scatter(HDH, P_star_s.to_numpy(dtype=float), s=8, alpha=0.4)
    plt.title(f"{title_prefix} – P* vs HDH (EWMA α={alpha}, T_b={T_b}°C)")
    plt.xlabel("HDH (K)")
    plt.ylabel("P* (kW)")
    plt.tight_layout()
    figs["scatter"] = fig

    if show:
        plt.show()

    return figs