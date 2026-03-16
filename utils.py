from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from statsmodels.tsa.seasonal import MSTL


def mstl_decomposition(series: np.ndarray, periods: list[int]) -> dict[int | str, np.ndarray]:
    if series.ndim != 1:
        raise ValueError("series must be 1D")
    if not periods:
        raise ValueError("periods must contain at least one integer")
    if np.isnan(series).any():
        raise ValueError("series contains NaNs; clean/interpolate missing data before MSTL")

    model = MSTL(
        series,
        periods=tuple(periods),
        stl_kwargs={"robust": True},
    )
    result = model.fit()

    decomposition: dict[int | str, np.ndarray] = {}
    seasonal = np.asarray(result.seasonal)

    if seasonal.ndim == 1:
        decomposition[model.periods[0]] = seasonal
    else:
        for column_index, period in enumerate(model.periods):
            decomposition[period] = seasonal[:, column_index]

    decomposition["res"] = np.asarray(result.resid)
    decomposition["trend"] = np.asarray(result.trend)
    return decomposition


def ewma(values: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Simple exponential moving average used for temperature smoothing."""
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]

    for index in range(1, len(values)):
        smoothed[index] = alpha * values[index] + (1 - alpha) * smoothed[index - 1]

    return smoothed


def T_smooth_and_HDH(temperature: np.ndarray, T_b: float = 18.0) -> tuple[np.ndarray, np.ndarray]:
    smoothed_temperature = ewma(temperature, alpha=0.2)
    heating_degree_hours = np.maximum(0.0, T_b - smoothed_temperature)
    return smoothed_temperature, heating_degree_hours


def estimate_P_temp_huber(temperature: np.ndarray, baseline_adjusted_power: np.ndarray) -> dict[str, Any]:
    smoothed_temperature, heating_degree_hours = T_smooth_and_HDH(temperature)

    valid_mask = np.isfinite(baseline_adjusted_power) & np.isfinite(heating_degree_hours)
    regression_input = heating_degree_hours[valid_mask].reshape(-1, 1)
    regression_target = baseline_adjusted_power[valid_mask]

    model = HuberRegressor().fit(regression_input, regression_target)
    temperature_sensitivity = float(model.coef_[0])

    estimated_temperature_power = np.zeros_like(baseline_adjusted_power, dtype=float)
    estimated_temperature_power[valid_mask] = (
        temperature_sensitivity * heating_degree_hours[valid_mask]
    )
    estimated_temperature_power = np.maximum(0.0, estimated_temperature_power)

    return {
        "P_temp": estimated_temperature_power,
        "model": model,
        "beta": temperature_sensitivity,
        "HDH": heating_degree_hours,
        "mask": valid_mask,
        "T_smooth": smoothed_temperature,
    }


def upper_bound_P_star_hour_of_week(
    power: np.ndarray,
    outdoor_temperature: np.ndarray,
    *,
    T_b: float = 18.0,
    q_low: float = 0.1,
    offset_how: int = 0,
    min_warm_samples_per_slot: int = 10,
) -> dict[str, np.ndarray]:
    """
    Estimate an optimistic upper bound for weather-driven load by first building
    a low baseline for each hour of the week.
    """
    power = np.asarray(power, dtype=float)
    outdoor_temperature = np.asarray(outdoor_temperature, dtype=float)

    if power.ndim != 1 or outdoor_temperature.ndim != 1 or power.shape != outdoor_temperature.shape:
        raise ValueError("P and T_out must be 1D arrays with the same shape")
    if not (0.0 < q_low < 1.0):
        raise ValueError("q_low must be in (0, 1)")

    hour_of_week = (np.arange(power.size) + offset_how) % 168
    valid_mask = np.isfinite(power) & np.isfinite(outdoor_temperature)
    warm_mask = valid_mask & (outdoor_temperature >= T_b)

    slot_baseline = np.full(168, np.nan)
    for slot in range(168):
        slot_baseline[slot] = _estimate_slot_baseline(
            power=power,
            hour_of_week=hour_of_week,
            slot=slot,
            warm_mask=warm_mask,
            valid_mask=valid_mask,
            q_low=q_low,
            min_warm_samples_per_slot=min_warm_samples_per_slot,
        )

    baseline_by_hour = slot_baseline[hour_of_week]
    optimistic_residual = power - baseline_by_hour

    return {
        "P_base_low": baseline_by_hour,
        "P_star_ub": optimistic_residual,
        "slot_baseline": slot_baseline,
    }


def _estimate_slot_baseline(
    *,
    power: np.ndarray,
    hour_of_week: np.ndarray,
    slot: int,
    warm_mask: np.ndarray,
    valid_mask: np.ndarray,
    q_low: float,
    min_warm_samples_per_slot: int,
) -> float:
    warm_samples_in_slot = warm_mask & (hour_of_week == slot)
    if warm_samples_in_slot.sum() >= min_warm_samples_per_slot:
        return float(np.nanquantile(power[warm_samples_in_slot], q_low))

    valid_samples_in_slot = valid_mask & (hour_of_week == slot)
    if valid_samples_in_slot.sum() >= 5:
        return float(np.nanquantile(power[valid_samples_in_slot], q_low))

    return np.nan


def upper_bound_P_temp_from_P_star(
    optimistic_residual: np.ndarray,
    outdoor_temperature: np.ndarray,
    *,
    T_b: float = 18.0,
    warm_q: float = 0.2,
) -> np.ndarray:
    """
    Turn the optimistic residual into an optimistic temperature-dependent load by
    removing a warm-hour baseline and keeping only cold-hour load above that level.
    """
    optimistic_residual = np.asarray(optimistic_residual, dtype=float)
    outdoor_temperature = np.asarray(outdoor_temperature, dtype=float)

    if optimistic_residual.shape != outdoor_temperature.shape:
        raise ValueError("P_star_ub and T_out must have same shape")

    valid_mask = np.isfinite(optimistic_residual) & np.isfinite(outdoor_temperature)
    warm_mask = valid_mask & (outdoor_temperature >= T_b)
    cold_mask = valid_mask & (outdoor_temperature < T_b)

    warm_baseline = 0.0
    if warm_mask.sum() >= 10:
        warm_baseline = float(np.nanquantile(optimistic_residual[warm_mask], warm_q))

    optimistic_temperature_power = np.zeros_like(optimistic_residual, dtype=float)
    optimistic_temperature_power[cold_mask] = np.maximum(
        0.0,
        optimistic_residual[cold_mask] - warm_baseline,
    )
    return optimistic_temperature_power


def peak_share_attribution(
    total: pd.Series,
    sensors: dict[str, pd.Series],
    peak_ref: pd.Series,
    q: float,
    *,
    min_peak_count: int = 24,
) -> dict[str, object]:
    aligned = _build_aligned_meter_frame(total, peak_ref, sensors)

    peak_threshold = aligned["peak_ref"].quantile(q)
    peak_mask = aligned["peak_ref"] >= peak_threshold
    peak_count = int(peak_mask.sum())

    if peak_count < min_peak_count:
        return {
            "error": f"Too few peak hours: {peak_count}",
            "peak_threshold": float(peak_threshold),
            "n_peak": peak_count,
        }

    shares = _calculate_hourly_shares(aligned, list(sensors.keys()))
    peak_shares = shares.loc[peak_mask]
    share_summary = pd.DataFrame(
        {
            "mean_share": peak_shares.mean(),
            "median_share": peak_shares.median(),
            "p95_share": peak_shares.quantile(0.95),
        }
    ).sort_values("mean_share", ascending=False)

    energy_weighted_share = _calculate_energy_weighted_share(
        aligned=aligned,
        peak_mask=peak_mask,
        sensor_names=list(sensors.keys()),
    )

    return {
        "peak_threshold": float(peak_threshold),
        "n_peak": peak_count,
        "summary_peak_shares": share_summary,
        "energy_weighted_peak_share": energy_weighted_share,
        "peak_hours_index": aligned.index[peak_mask],
        "shares_peak_hours": peak_shares,
    }


def peak_times_meter_info(
    P_total_avg: pd.Series,
    P_peak_ref: pd.Series,
    other_avg: dict[str, pd.Series],
    *,
    q: float = 0.99,
    min_peak_count: int = 24,
) -> dict[str, Any]:
    """
    Summarize how each submeter behaves during hours where the reference peak
    signal is above the chosen quantile threshold.
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")

    aligned = _build_aligned_meter_frame(P_total_avg, P_peak_ref, other_avg)
    coverage = float(len(aligned)) / float(len(P_peak_ref.dropna()))

    peak_threshold = float(aligned["peak_ref"].quantile(q))
    peak_mask = aligned["peak_ref"] >= peak_threshold
    peak_count = int(peak_mask.sum())

    if peak_count < min_peak_count:
        return {
            "peak_threshold": peak_threshold,
            "n_peak": peak_count,
            "peak_hours": aligned.index[peak_mask],
            "stats_peak": None,
            "share_peak": None,
            "coverage": coverage,
            "error": f"Too few peak hours for q={q} (n_peak={peak_count})",
        }

    stats_peak = _summarize_peak_meter_stats(aligned, peak_mask, list(other_avg.keys()))
    share_peak = _summarize_peak_meter_shares(aligned, peak_mask, list(other_avg.keys()))

    return {
        "peak_threshold": peak_threshold,
        "n_peak": peak_count,
        "peak_hours": aligned.index[peak_mask],
        "stats_peak": stats_peak,
        "share_peak": share_peak,
        "coverage": coverage,
    }


def _build_aligned_meter_frame(
    total: pd.Series,
    peak_ref: pd.Series,
    sensors: dict[str, pd.Series],
) -> pd.DataFrame:
    frame = pd.DataFrame({"total": total, "peak_ref": peak_ref})
    for sensor_name, sensor_series in sensors.items():
        frame[sensor_name] = sensor_series

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(how="any").sort_index()

    if frame.empty:
        raise ValueError("No overlapping data after alignment.")

    return frame


def _calculate_hourly_shares(aligned: pd.DataFrame, sensor_names: list[str]) -> pd.DataFrame:
    shares = pd.DataFrame(index=aligned.index)

    for sensor_name in sensor_names:
        shares[sensor_name] = (aligned[sensor_name] / aligned["total"]).clip(lower=0.0)

    shares["known_sum"] = shares[sensor_names].sum(axis=1)
    shares["other"] = (1.0 - shares["known_sum"]).clip(lower=0.0)
    return shares


def _calculate_energy_weighted_share(
    *,
    aligned: pd.DataFrame,
    peak_mask: pd.Series,
    sensor_names: list[str],
) -> dict[str, float]:
    denominator = aligned.loc[peak_mask, "total"].sum()
    energy_weighted_share: dict[str, float] = {}

    for sensor_name in sensor_names:
        energy_weighted_share[sensor_name] = float(
            aligned.loc[peak_mask, sensor_name].sum() / (denominator + 1e-12)
        )

    other_power = (aligned["total"] - aligned[sensor_names].sum(axis=1)).clip(lower=0.0)
    energy_weighted_share["other"] = float(
        other_power.loc[peak_mask].sum() / (denominator + 1e-12)
    )
    return energy_weighted_share


def _summarize_peak_meter_stats(
    aligned: pd.DataFrame,
    peak_mask: pd.Series,
    sensor_names: list[str],
) -> pd.DataFrame:
    rows = []

    for sensor_name in sensor_names:
        sensor_values = aligned.loc[peak_mask, sensor_name].to_numpy(dtype=float)
        rows.append(
            {
                "meter": sensor_name,
                "mean_kW": float(np.mean(sensor_values)),
                "median_kW": float(np.median(sensor_values)),
                "p95_kW": float(np.percentile(sensor_values, 95)),
                "max_kW": float(np.max(sensor_values)),
            }
        )

    return pd.DataFrame(rows).set_index("meter").sort_values("mean_kW", ascending=False)


def _summarize_peak_meter_shares(
    aligned: pd.DataFrame,
    peak_mask: pd.Series,
    sensor_names: list[str],
) -> pd.DataFrame:
    denominator = float(aligned.loc[peak_mask, "total"].sum())
    rows = []

    for sensor_name in sensor_names:
        rows.append(
            {
                "meter": sensor_name,
                "energy_weighted_share": float(
                    aligned.loc[peak_mask, sensor_name].sum() / (denominator + 1e-12)
                ),
            }
        )

    known_sum = aligned.loc[peak_mask, sensor_names].sum(axis=1)
    unaccounted_other = (aligned.loc[peak_mask, "total"] - known_sum).clip(lower=0.0)
    rows.append(
        {
            "meter": "UNACCOUNTED_OTHER",
            "energy_weighted_share": float(unaccounted_other.sum() / (denominator + 1e-12)),
        }
    )

    return pd.DataFrame(rows).set_index("meter").sort_values(
        "energy_weighted_share",
        ascending=False,
    )
