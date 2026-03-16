import numpy as np
from itertools import product
from typing import Any

def calc_flex_score(
    P: np.ndarray,
    P_temp: np.ndarray,
    q: float,
    *,
    f_realizable: float = 0.4,
    k_scale: float = 20.0,
) -> dict[str, float]:
    """
    Score how well temperature-dependent power aligns with peaks.

    Parameters
    ----------
    P : np.ndarray
        Total power time series (kW) (or consistent units).
    P_temp : np.ndarray
        Estimated temperature-dependent power (same units as P).
    q : float
        Peak quantile (e.g. 0.95 => top 5% hours are "peak").
    f_realizable : float
        Conservative fraction of P_temp assumed flexible in practice (0..1).
    k_scale : float
        Scale for mapping flexible kW into a 0..1 score via 1-exp(-x/k_scale).

    Returns
    -------
    dict[str, float]
        Contains: score_0_100, peak_threshold, peak_temp_share, temp_in_peak_share,
        flex_kW_peak_median, flex_kW_peak_p95, coverage, n_total, n_valid, n_peak
    """
    P = np.asarray(P, dtype=float)
    P_temp = np.asarray(P_temp, dtype=float)

    # Basic input validation
    if P.ndim != 1 or P_temp.ndim != 1:
        raise ValueError("P and P_temp must be 1D arrays")
    if P.shape != P_temp.shape:
        raise ValueError("P and P_temp must have the same shape")
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")
    if not (0.0 <= f_realizable <= 1.0):
        raise ValueError("f_realizable must be in [0, 1]")
    if k_scale <= 0:
        raise ValueError("k_scale must be > 0")

    valid = np.isfinite(P) & np.isfinite(P_temp)
    n_total = float(P.size)
    n_valid = float(valid.sum())
    coverage = n_valid / n_total if n_total > 0 else 0.0

    if n_valid < 10:
        return {
            "score_0_100": 0.0,
            "peak_threshold": np.nan,
            "peak_temp_share": np.nan,
            "temp_in_peak_share": np.nan,
            "flex_kW_peak_median": np.nan,
            "flex_kW_peak_p95": np.nan,
            "coverage": coverage,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_peak": 0.0,
        }

    peak_threshold = float(np.nanquantile(P[valid], q))
    is_peak = valid & (P >= peak_threshold)
    n_peak = float(is_peak.sum())

    eps = 1e-12
    peak_temp_share = float(P_temp[is_peak].sum() / (P[is_peak].sum() + eps))
    temp_in_peak_share = float(P_temp[is_peak].sum() / (P_temp[valid].sum() + eps))

    # Conservative flexible power estimate during peak hours
    P_flex = f_realizable * np.maximum(0.0, P_temp)
    if n_peak >= 3:
        flex_kW_peak_median = float(np.nanmedian(P_flex[is_peak]))
        flex_kW_peak_p95 = float(np.nanpercentile(P_flex[is_peak], 95))
    else:
        flex_kW_peak_median = 0.0
        flex_kW_peak_p95 = 0.0

    # Map metrics to [0, 1]
    s_peak_share = np.clip(peak_temp_share, 0.0, 1.0)
    s_temp_in_peak = np.clip(temp_in_peak_share, 0.0, 1.0)
    s_flex_kw = 1.0 - float(np.exp(-flex_kW_peak_median / k_scale))  # saturating
    s_conf = np.clip(coverage, 0.0, 1.0)

    # Composite score (0..100)
    score_0_100 = 100.0 * (
        0.35 * s_peak_share +
        0.25 * s_temp_in_peak +
        0.25 * s_flex_kw +
        0.15 * s_conf
    )

    return {
        "score_0_100": float(score_0_100),
        "peak_threshold": peak_threshold,
        "peak_temp_share": peak_temp_share,
        "temp_in_peak_share": temp_in_peak_share,
        "flex_kW_peak_median": flex_kW_peak_median,
        "flex_kW_peak_p95": flex_kW_peak_p95,
        "coverage": coverage,
        "n_total": n_total,
        "n_valid": n_valid,
        "n_peak": n_peak,
    }

def grid_search_temperature_flex_score(
    P: np.ndarray,
    P_temp: np.ndarray,
    q_grid: list[float],
    f_grid: list[float],
    k_scale_grid: list[float],
    *,
    min_peak_count: int = 24,   # feasibility: need enough peak hours to be meaningful
) -> dict[str, Any]:
    """
    For each (f_realizable, k_scale) combo, scan q_grid and keep the BEST (max) score.
    Only keeps feasible runs where n_peak >= min_peak_count and basic constraints are met.

    Requires: temperature_flex_score(P, P_temp, q, f_realizable=..., k_scale=...)
    """

    P = np.asarray(P, dtype=float)
    P_temp = np.asarray(P_temp, dtype=float)

    if P.ndim != 1 or P_temp.ndim != 1:
        raise ValueError("P and P_temp must be 1D arrays")
    if P.shape != P_temp.shape:
        raise ValueError("P and P_temp must have the same shape")
    if min_peak_count < 3:
        min_peak_count = 3  # your score uses percentiles/median -> need >= 3

    # Pre-check: any valid data at all?
    valid = np.isfinite(P) & np.isfinite(P_temp)
    if valid.sum() < 10:
        return {"best_per_param_combo": [], "best_overall": None}

    best_per_param_combo: list[dict[str, Any]] = []

    for f_realizable, k_scale in product(f_grid, k_scale_grid):
        # Feasible parameter constraints
        if not (0.0 <= f_realizable <= 1.0):
            continue
        if not (k_scale > 0.0 and np.isfinite(k_scale)):
            continue

        best_for_this_combo = None

        for q in q_grid:
            if not (0.0 < q < 1.0 and np.isfinite(q)):
                continue

            s = calc_flex_score(
                P, P_temp, q,
                f_realizable=f_realizable,
                k_scale=k_scale,
            )

            # Feasibility: enough peak samples
            if (not np.isfinite(s["peak_threshold"])) or (s["n_peak"] < min_peak_count):
                continue

            record = {
                **s,
                "q": float(q),
                "f_realizable": float(f_realizable),
                "k_scale": float(k_scale),
            }

            if (best_for_this_combo is None) or (record["score_0_100"] > best_for_this_combo["score_0_100"]):
                best_for_this_combo = record

        if best_for_this_combo is not None:
            best_per_param_combo.append(best_for_this_combo)

    best_overall = None
    if best_per_param_combo:
        best_overall = max(best_per_param_combo, key=lambda d: d["score_0_100"])

    return {
        "best_per_param_combo": best_per_param_combo,  # one row per (f_realizable, k_scale), best q chosen
        "best_overall": best_overall,                  # best across everything
    }

