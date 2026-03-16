from itertools import product
from typing import Any

import numpy as np


def calc_flex_score(
    P: np.ndarray,
    P_temp: np.ndarray,
    q: float,
    *,
    f_realizable: float = 0.4,
    k_scale: float = 20.0,
) -> dict[str, float]:
    """
    Score how well temperature-dependent power aligns with peak hours.
    """
    power = np.asarray(P, dtype=float)
    temperature_power = np.asarray(P_temp, dtype=float)

    _validate_score_inputs(power, temperature_power, q, f_realizable, k_scale)

    valid_mask = np.isfinite(power) & np.isfinite(temperature_power)
    total_count = float(power.size)
    valid_count = float(valid_mask.sum())
    coverage = valid_count / total_count if total_count > 0 else 0.0

    if valid_count < 10:
        return _empty_score_result(total_count, valid_count, coverage)

    peak_threshold = float(np.nanquantile(power[valid_mask], q))
    peak_mask = valid_mask & (power >= peak_threshold)
    peak_count = float(peak_mask.sum())

    peak_temp_share, temp_in_peak_share = _calculate_peak_shares(
        power=power,
        temperature_power=temperature_power,
        valid_mask=valid_mask,
        peak_mask=peak_mask,
    )
    flex_peak_median, flex_peak_p95 = _calculate_flexible_peak_power(
        temperature_power=temperature_power,
        peak_mask=peak_mask,
        f_realizable=f_realizable,
    )

    score = _combine_score_components(
        peak_temp_share=peak_temp_share,
        temp_in_peak_share=temp_in_peak_share,
        flex_peak_median=flex_peak_median,
        coverage=coverage,
        k_scale=k_scale,
    )

    return {
        "score_0_100": float(score),
        "peak_threshold": peak_threshold,
        "peak_temp_share": peak_temp_share,
        "temp_in_peak_share": temp_in_peak_share,
        "flex_kW_peak_median": flex_peak_median,
        "flex_kW_peak_p95": flex_peak_p95,
        "coverage": coverage,
        "n_total": total_count,
        "n_valid": valid_count,
        "n_peak": peak_count,
    }


def _validate_score_inputs(
    power: np.ndarray,
    temperature_power: np.ndarray,
    q: float,
    f_realizable: float,
    k_scale: float,
) -> None:
    if power.ndim != 1 or temperature_power.ndim != 1:
        raise ValueError("P and P_temp must be 1D arrays")
    if power.shape != temperature_power.shape:
        raise ValueError("P and P_temp must have the same shape")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")
    if not (0.0 <= f_realizable <= 1.0):
        raise ValueError("f_realizable must be in [0, 1]")
    if k_scale <= 0:
        raise ValueError("k_scale must be > 0")


def _empty_score_result(total_count: float, valid_count: float, coverage: float) -> dict[str, float]:
    return {
        "score_0_100": 0.0,
        "peak_threshold": np.nan,
        "peak_temp_share": np.nan,
        "temp_in_peak_share": np.nan,
        "flex_kW_peak_median": np.nan,
        "flex_kW_peak_p95": np.nan,
        "coverage": coverage,
        "n_total": total_count,
        "n_valid": valid_count,
        "n_peak": 0.0,
    }


def _calculate_peak_shares(
    *,
    power: np.ndarray,
    temperature_power: np.ndarray,
    valid_mask: np.ndarray,
    peak_mask: np.ndarray,
) -> tuple[float, float]:
    eps = 1e-12
    temperature_power_during_peaks = temperature_power[peak_mask].sum()
    total_power_during_peaks = power[peak_mask].sum()
    total_temperature_power = temperature_power[valid_mask].sum()

    peak_temp_share = float(temperature_power_during_peaks / (total_power_during_peaks + eps))
    temp_in_peak_share = float(temperature_power_during_peaks / (total_temperature_power + eps))
    return peak_temp_share, temp_in_peak_share


def _calculate_flexible_peak_power(
    *,
    temperature_power: np.ndarray,
    peak_mask: np.ndarray,
    f_realizable: float,
) -> tuple[float, float]:
    flexible_power = f_realizable * np.maximum(0.0, temperature_power)

    if peak_mask.sum() < 3:
        return 0.0, 0.0

    flex_peak_median = float(np.nanmedian(flexible_power[peak_mask]))
    flex_peak_p95 = float(np.nanpercentile(flexible_power[peak_mask], 95))
    return flex_peak_median, flex_peak_p95


def _combine_score_components(
    *,
    peak_temp_share: float,
    temp_in_peak_share: float,
    flex_peak_median: float,
    coverage: float,
    k_scale: float,
) -> float:
    peak_share_component = np.clip(peak_temp_share, 0.0, 1.0)
    peak_capture_component = np.clip(temp_in_peak_share, 0.0, 1.0)
    flexible_power_component = 1.0 - float(np.exp(-flex_peak_median / k_scale))
    coverage_component = np.clip(coverage, 0.0, 1.0)

    return 100.0 * (
        0.35 * peak_share_component
        + 0.25 * peak_capture_component
        + 0.25 * flexible_power_component
        + 0.15 * coverage_component
    )


def grid_search_temperature_flex_score(
    P: np.ndarray,
    P_temp: np.ndarray,
    q_grid: list[float],
    f_grid: list[float],
    k_scale_grid: list[float],
    *,
    min_peak_count: int = 24,
) -> dict[str, Any]:
    """
    For each combination of realizable fraction and k-scale, keep the best score
    found across the candidate peak quantiles.
    """
    power = np.asarray(P, dtype=float)
    temperature_power = np.asarray(P_temp, dtype=float)

    if power.ndim != 1 or temperature_power.ndim != 1:
        raise ValueError("P and P_temp must be 1D arrays")
    if power.shape != temperature_power.shape:
        raise ValueError("P and P_temp must have the same shape")

    minimum_required_peaks = max(min_peak_count, 3)
    valid_mask = np.isfinite(power) & np.isfinite(temperature_power)
    if valid_mask.sum() < 10:
        return {"best_per_param_combo": [], "best_overall": None}

    best_per_param_combo: list[dict[str, Any]] = []

    for f_realizable, k_scale in product(f_grid, k_scale_grid):
        if not _is_valid_parameter_combo(f_realizable, k_scale):
            continue

        best_result_for_combo = _find_best_score_for_parameter_combo(
            power=power,
            temperature_power=temperature_power,
            q_grid=q_grid,
            f_realizable=f_realizable,
            k_scale=k_scale,
            min_peak_count=minimum_required_peaks,
        )

        if best_result_for_combo is not None:
            best_per_param_combo.append(best_result_for_combo)

    best_overall = None
    if best_per_param_combo:
        best_overall = max(best_per_param_combo, key=lambda result: result["score_0_100"])

    return {
        "best_per_param_combo": best_per_param_combo,
        "best_overall": best_overall,
    }


def _is_valid_parameter_combo(f_realizable: float, k_scale: float) -> bool:
    return 0.0 <= f_realizable <= 1.0 and k_scale > 0.0 and np.isfinite(k_scale)


def _find_best_score_for_parameter_combo(
    *,
    power: np.ndarray,
    temperature_power: np.ndarray,
    q_grid: list[float],
    f_realizable: float,
    k_scale: float,
    min_peak_count: int,
) -> dict[str, Any] | None:
    best_result = None

    for q in q_grid:
        if not (0.0 < q < 1.0 and np.isfinite(q)):
            continue

        score = calc_flex_score(
            power,
            temperature_power,
            q,
            f_realizable=f_realizable,
            k_scale=k_scale,
        )

        if not np.isfinite(score["peak_threshold"]) or score["n_peak"] < min_peak_count:
            continue

        candidate = {
            **score,
            "q": float(q),
            "f_realizable": float(f_realizable),
            "k_scale": float(k_scale),
        }

        if best_result is None or candidate["score_0_100"] > best_result["score_0_100"]:
            best_result = candidate

    return best_result
