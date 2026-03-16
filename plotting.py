from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ewma


def _parse_start_timestamp_from_tid(tid: pd.Series) -> pd.Series:
    """
    Parse start timestamps from strings like '01.01.2025 00:00 - 01:00'.
    """
    cleaned = tid.astype(str).str.strip()
    start_text = cleaned.str.split(" - ", n=1, expand=True)[0].str.strip()

    parsed = pd.to_datetime(start_text, format="%d.%m.%Y %H:%M", errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(start_text, dayfirst=True, errors="coerce")
        parsed = parsed.fillna(fallback)

    return parsed


def plot_flex_diagnostics(
    *,
    df_main: pd.DataFrame,
    P_avg: np.ndarray,
    P_peak: np.ndarray,
    T_out: np.ndarray,
    P_star: np.ndarray,
    P_temp: np.ndarray,
    other_avg: dict[str, Any] | None = None,
    q: float = 0.99,
    top_n_peaks: int = 48,
    periods: list[tuple[str, str]] | None = None,
    title_prefix: str = "Flex diagnostics",
    show: bool = True,
) -> dict[str, plt.Figure]:
    """
    Plot the main diagnostics used to understand peak behavior and the weather
    contribution estimate.
    """
    plot_index, keep_mask = _build_plot_index(df_main)
    main_series = _build_main_plot_series(
        index=plot_index,
        keep_mask=keep_mask,
        P_avg=P_avg,
        P_peak=P_peak,
        T_out=T_out,
        P_star=P_star,
        P_temp=P_temp,
    )
    other_series = _build_other_meter_plot_series(
        other_avg=other_avg,
        index=plot_index,
        keep_mask=keep_mask,
    )

    if periods is None:
        periods = _default_plot_periods(plot_index)

    figures: dict[str, plt.Figure] = {}
    figures["overlay"] = _plot_overlay_periods(main_series, periods, title_prefix)

    peak_attribution_figure = _plot_peak_attribution(
        main_series=main_series,
        other_series=other_series,
        q=q,
        top_n_peaks=top_n_peaks,
        title_prefix=title_prefix,
    )
    if peak_attribution_figure is not None:
        figures["peak_attribution"] = peak_attribution_figure

    figures["scatter"] = _plot_temperature_scatter(
        P_star_series=main_series["P_star"],
        temperature_series=main_series["T_out"],
        title_prefix=title_prefix,
    )

    if show:
        plt.show()

    return figures


def _build_plot_index(df_main: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if isinstance(df_main.index, pd.DatetimeIndex):
        timestamps = pd.Series(df_main.index)
    elif "Tid (Time)" in df_main.columns:
        timestamps = _parse_start_timestamp_from_tid(df_main["Tid (Time)"])
    else:
        raise ValueError("df_main must have a DatetimeIndex or contain a 'Tid (Time)' column.")

    keep_mask = (timestamps.notna() & ~timestamps.duplicated(keep="first")).to_numpy()
    index = pd.DatetimeIndex(timestamps[keep_mask].to_numpy())
    return index, keep_mask


def _build_main_plot_series(
    *,
    index: pd.DatetimeIndex,
    keep_mask: np.ndarray,
    P_avg: np.ndarray,
    P_peak: np.ndarray,
    T_out: np.ndarray,
    P_star: np.ndarray,
    P_temp: np.ndarray,
) -> dict[str, pd.Series]:
    return {
        "P_avg": _array_to_series(P_avg, "P_avg", index, keep_mask),
        "P_peak": _array_to_series(P_peak, "P_peak", index, keep_mask),
        "T_out": _array_to_series(T_out, "T_out", index, keep_mask),
        "P_star": _array_to_series(P_star, "P_star", index, keep_mask),
        "P_temp": _array_to_series(P_temp, "P_temp", index, keep_mask),
    }


def _array_to_series(
    values: np.ndarray,
    name: str,
    index: pd.DatetimeIndex,
    keep_mask: np.ndarray,
) -> pd.Series:
    array = np.asarray(values, dtype=float)
    if array.shape[0] != keep_mask.shape[0]:
        raise ValueError(f"{name} length {array.shape[0]} does not match df_main length {keep_mask.shape[0]}.")

    return pd.Series(array[keep_mask], index=index, name=name)


def _build_other_meter_plot_series(
    *,
    other_avg: dict[str, Any] | None,
    index: pd.DatetimeIndex,
    keep_mask: np.ndarray,
) -> dict[str, pd.Series]:
    other_series: dict[str, pd.Series] = {}
    if not other_avg:
        return other_series

    for meter_name, values in other_avg.items():
        other_series[meter_name] = _normalize_other_meter_series(
            values=values,
            meter_name=meter_name,
            index=index,
            keep_mask=keep_mask,
        )

    return other_series


def _normalize_other_meter_series(
    *,
    values: Any,
    meter_name: str,
    index: pd.DatetimeIndex,
    keep_mask: np.ndarray,
) -> pd.Series:
    if isinstance(values, pd.Series) and isinstance(values.index, pd.DatetimeIndex):
        return values.reindex(index).astype(float)

    array = np.asarray(values, dtype=float)
    if array.shape[0] == keep_mask.shape[0]:
        return pd.Series(array[keep_mask], index=index, name=str(meter_name))

    trimmed_length = min(array.shape[0], len(index))
    return pd.Series(array[:trimmed_length], index=index[:trimmed_length], name=str(meter_name))


def _default_plot_periods(index: pd.DatetimeIndex) -> list[tuple[str, str]]:
    first_start = index.min()
    first_end = first_start + pd.Timedelta(days=7)
    last_end = index.max()
    last_start = last_end - pd.Timedelta(days=7)

    return [
        (first_start.isoformat(), first_end.isoformat()),
        (last_start.isoformat(), last_end.isoformat()),
    ]


def _plot_overlay_periods(
    main_series: dict[str, pd.Series],
    periods: list[tuple[str, str]],
    title_prefix: str,
) -> plt.Figure:
    overlay_figures: list[plt.Figure] = []

    for period_number, (start, end) in enumerate(periods, start=1):
        window = slice(pd.to_datetime(start), pd.to_datetime(end))
        figure = plt.figure(figsize=(14, 5))
        plt.plot(main_series["P_avg"].loc[window].index, main_series["P_avg"].loc[window].to_numpy(), label="P_avg (kW)")
        plt.plot(main_series["P_peak"].loc[window].index, main_series["P_peak"].loc[window].to_numpy(), label="P_peak (kW)")
        plt.plot(main_series["P_temp"].loc[window].index, main_series["P_temp"].loc[window].to_numpy(), label="P_temp (kW)")
        plt.title(f"{title_prefix} - Overlay {period_number}: {start} to {end}")
        plt.xlabel("Time")
        plt.ylabel("kW")
        plt.legend()
        plt.tight_layout()
        overlay_figures.append(figure)

    return overlay_figures[-1]


def _plot_peak_attribution(
    *,
    main_series: dict[str, pd.Series],
    other_series: dict[str, pd.Series],
    q: float,
    top_n_peaks: int,
    title_prefix: str,
) -> plt.Figure | None:
    if not other_series:
        return None

    stacked = pd.DataFrame(
        {
            "total": main_series["P_avg"],
            "peak_ref": main_series["P_peak"],
            **other_series,
        }
    ).dropna().sort_index()

    threshold = float(stacked["peak_ref"].quantile(q))
    peak_hours = (
        stacked.loc[stacked["peak_ref"] >= threshold]
        .sort_values("peak_ref", ascending=False)
        .head(top_n_peaks)
    )

    meter_names = list(other_series.keys())
    known_sum = peak_hours[meter_names].sum(axis=1)
    peak_hours["UNACCOUNTED_OTHER"] = (peak_hours["total"] - known_sum).clip(lower=0.0)

    figure = plt.figure(figsize=(14, 6))
    x_positions = np.arange(len(peak_hours))
    stacked_bottom = np.zeros(len(peak_hours), dtype=float)

    for meter_name in meter_names + ["UNACCOUNTED_OTHER"]:
        values = peak_hours[meter_name].to_numpy(dtype=float)
        plt.bar(x_positions, values, bottom=stacked_bottom, label=meter_name)
        stacked_bottom += values

    plt.xticks(x_positions, [timestamp.strftime("%Y-%m-%d %H:%M") for timestamp in peak_hours.index], rotation=90)
    plt.title(
        f"{title_prefix} - Peak attribution (stacked)\n"
        f"q={q}, threshold={threshold:.3f}, top {len(peak_hours)} hours"
    )
    plt.xlabel("Peak hours (sorted by P_peak)")
    plt.ylabel("kW (avg)")
    plt.legend(ncols=2, fontsize=9)
    plt.tight_layout()
    return figure


def _plot_temperature_scatter(
    *,
    P_star_series: pd.Series,
    temperature_series: pd.Series,
    title_prefix: str,
) -> plt.Figure:
    alpha = 0.2
    balance_temperature = 18.0

    # The smoothed temperature acts as a simple stand-in for thermal inertia, so
    # the scatter is easier to interpret than using raw outdoor temperature.
    smoothed_temperature = ewma(temperature_series.to_numpy(dtype=float), alpha=alpha)
    heating_degree_hours = np.maximum(0.0, balance_temperature - smoothed_temperature)

    figure = plt.figure(figsize=(7, 6))
    plt.scatter(heating_degree_hours, P_star_series.to_numpy(dtype=float), s=8, alpha=0.4)
    plt.title(
        f"{title_prefix} - P* vs HDH "
        f"(EWMA alpha={alpha}, T_b={balance_temperature}C)"
    )
    plt.xlabel("HDH (K)")
    plt.ylabel("P* (kW)")
    plt.tight_layout()
    return figure
