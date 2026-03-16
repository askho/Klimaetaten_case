import numpy as np
from statsmodels.tsa.seasonal import MSTL
from sklearn.linear_model import HuberRegressor
import pandas as pd

def mstl_decomposition(
    series: np.ndarray,
    periods: list[int]
) -> dict[int | str, np.ndarray]:

    # Error checks
    if series.ndim != 1:
        raise ValueError("series must be 1D")
    if not periods:
        raise ValueError("periods must contain at least one integer")
    if np.isnan(series).any():
        raise ValueError("series contains NaNs; clean/interpolate missing data before MSTL")

    mstl = MSTL(
        series,
        periods=tuple(periods),
        stl_kwargs={"robust": True},
    )
    res = mstl.fit()


    decomposition_results: dict[int | str, np.ndarray] = {}

    seasonal = np.asarray(res.seasonal)

    if seasonal.ndim == 1:
        decomposition_results[mstl.periods[0]] = seasonal
    else:
        for i, period in enumerate(mstl.periods):
            decomposition_results[period] = seasonal[:, i]

    decomposition_results["res"] = np.asarray(res.resid)
    decomposition_results["trend"] = np.asarray(res.trend)  

    return decomposition_results

def ewma(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """ exponential weighted moving average """
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def T_smooth_and_HDH(T: np.ndarray, T_b: float = 18.0) -> tuple[np.ndarray, np.ndarray]:
    T_smooth = ewma(T, alpha=0.2)  # ~5–10h memory; tune if you want
    HDH = np.maximum(0.0, T_b - T_smooth)  # heating degree "hours" (Δ°C = ΔK)
    return T_smooth, HDH

def estimate_P_temp_huber(T: np.ndarray, P_star: np.ndarray) -> dict[str, Any]:
    T_smooth, HDH = T_smooth_and_HDH(T)

    mask = np.isfinite(P_star) & np.isfinite(HDH)
    X = HDH[mask].reshape(-1, 1)
    y = P_star[mask]

    model = HuberRegressor().fit(X, y)

    beta = float(model.coef_[0])

    P_temp = np.zeros_like(P_star, dtype=float)
    P_temp[mask] = beta * HDH[mask]
    P_temp = np.maximum(0.0, P_temp)

    print(beta)
    return {
        "P_temp": P_temp,
        "model": model,
        "beta": beta,
        "HDH": HDH,
        "mask": mask,
        "T_smooth": T_smooth,
    }

def upper_bound_P_star_hour_of_week(
    P: np.ndarray,
    T_out: np.ndarray,
    *,
    T_b: float = 18.0,
    q_low: float = 0.1,
    offset_how: int = 0,          # shift if your series doesn't start at Mon 00:00
    min_warm_samples_per_slot: int = 10,
) -> dict[str, np.ndarray]:
    """
    Upper bound on P_star by using a low (q_low) baseline per hour-of-week,
    estimated from warm hours (T_out >= T_b) when heating should be minimal.

    Returns dict with:
      - P_base_low: baseline (same length as P)
      - P_star_ub:  optimistic leftover P - P_base_low
      - slot_baseline: length-168 baseline values
    """
    P = np.asarray(P, dtype=float)
    T_out = np.asarray(T_out, dtype=float)

    if P.ndim != 1 or T_out.ndim != 1 or P.shape != T_out.shape:
        raise ValueError("P and T_out must be 1D arrays with the same shape")
    if not (0.0 < q_low < 1.0):
        raise ValueError("q_low must be in (0, 1)")

    n = P.size
    idx = np.arange(n)
    how = (idx + offset_how) % 168

    valid = np.isfinite(P) & np.isfinite(T_out)
    warm = valid & (T_out >= T_b)

    slot_baseline = np.full(168, np.nan)

    for s in range(168):
        m = warm & (how == s)
        if m.sum() >= min_warm_samples_per_slot:
            slot_baseline[s] = float(np.nanquantile(P[m], q_low))
        else:
            # fallback: use all valid samples for that slot if warm hours are scarce
            m2 = valid & (how == s)
            if m2.sum() >= 5:
                slot_baseline[s] = float(np.nanquantile(P[m2], q_low))
            else:
                slot_baseline[s] = np.nan

    P_base_low = slot_baseline[how]
    P_star_ub = P - P_base_low

    return {
        "P_base_low": P_base_low,
        "P_star_ub": P_star_ub,
        "slot_baseline": slot_baseline,
    }


def upper_bound_P_temp_from_P_star(
    P_star_ub: np.ndarray,
    T_out: np.ndarray,
    *,
    T_b: float = 18.0,
    warm_q: float = 0.2,
) -> np.ndarray:
    """
    Optimistic upper bound on temperature-dependent power:
    - subtract a warm-hour baseline from P_star_ub
    - count only cold hours
    """
    P_star_ub = np.asarray(P_star_ub, dtype=float)
    T_out = np.asarray(T_out, dtype=float)
    if P_star_ub.shape != T_out.shape:
        raise ValueError("P_star_ub and T_out must have same shape")

    valid = np.isfinite(P_star_ub) & np.isfinite(T_out)
    warm = valid & (T_out >= T_b)
    cold = valid & (T_out < T_b)

    baseline = 0.0 if warm.sum() < 10 else float(np.nanquantile(P_star_ub[warm], warm_q))

    P_temp_ub = np.zeros_like(P_star_ub, dtype=float)
    P_temp_ub[cold] = np.maximum(0.0, P_star_ub[cold] - baseline)
    return P_temp_ub

def peak_share_attribution(
    total: pd.Series,
    sensors: dict[str, pd.Series],
    peak_ref: pd.Series,
    q: float,
    *,
    min_peak_count: int = 24,
) -> dict[str, object]:
    """
    total: building total power (kW avg, ideally)
    sensors: dict of submeter power series (kW avg)
    peak_ref: series used to define peak hours (can be P_peak or P_avg)
    q: quantile threshold (e.g. 0.99)
    """

    # Align everything to common index
    df = pd.DataFrame({"total": total, "peak_ref": peak_ref})
    for name, s in sensors.items():
        df[name] = s
    df = df.sort_index()

    # Valid rows
    valid = df[["total", "peak_ref"]].notna().all(axis=1)
    for name in sensors:
        valid &= df[name].notna()
    df = df.loc[valid].copy()

    thr = df["peak_ref"].quantile(q)
    peak_mask = df["peak_ref"] >= thr
    n_peak = int(peak_mask.sum())
    if n_peak < min_peak_count:
        return {"error": f"Too few peak hours: {n_peak}", "peak_threshold": float(thr), "n_peak": n_peak}

    # Shares per hour
    shares = pd.DataFrame(index=df.index)
    for name in sensors:
        shares[name] = (df[name] / df["total"]).clip(lower=0.0)

    shares["known_sum"] = shares[list(sensors.keys())].sum(axis=1)
    shares["other"] = (1.0 - shares["known_sum"]).clip(lower=0.0)

    # Summary over peak hours
    peak_shares = shares.loc[peak_mask]
    summary = pd.DataFrame({
        "mean_share": peak_shares.mean(),
        "median_share": peak_shares.median(),
        "p95_share": peak_shares.quantile(0.95),
    }).sort_values("mean_share", ascending=False)

    # Energy-weighted share (often better than plain mean)
    energy_weighted = {}
    denom = df.loc[peak_mask, "total"].sum()
    for name in list(sensors.keys()) + ["other"]:
        if name == "other":
            # other power = total - sum sensors
            other_power = (df["total"] - df[list(sensors.keys())].sum(axis=1)).clip(lower=0.0)
            energy_weighted[name] = float(other_power.loc[peak_mask].sum() / (denom + 1e-12))
        else:
            energy_weighted[name] = float(df.loc[peak_mask, name].sum() / (denom + 1e-12))

    return {
        "peak_threshold": float(thr),
        "n_peak": n_peak,
        "summary_peak_shares": summary,
        "energy_weighted_peak_share": energy_weighted,
        "peak_hours_index": df.index[peak_mask],
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
    Summarize other meters during peak hours defined by P_peak_ref >= quantile(q).

    Parameters
    ----------
    P_total_avg : pd.Series
        Total building hourly average kW (main meter). Used for shares.
    P_peak_ref : pd.Series
        Series used to define peak hours (can be Peak High or Snittlast).
    other_avg : dict[str, pd.Series]
        Dict of other meters' hourly average kW series.
    q : float
        Peak quantile threshold.
    min_peak_count : int
        Require at least this many peak hours.

    Returns
    -------
    dict with:
      - peak_threshold, n_peak, peak_hours
      - stats_peak: DataFrame with mean/median/p95 per meter during peak hours
      - share_peak: DataFrame with energy-weighted shares during peak hours
      - coverage: fraction of timestamps kept after alignment
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must be in (0, 1)")

    # Align everything by timestamp
    df = pd.DataFrame({"total": P_total_avg, "peak_ref": P_peak_ref})
    for name, s in other_avg.items():
        df[name] = s

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="any").sort_index()

    if df.empty:
        raise ValueError("No overlapping data after alignment.")

    thr = float(df["peak_ref"].quantile(q))
    peak_mask = df["peak_ref"] >= thr
    n_peak = int(peak_mask.sum())

    if n_peak < min_peak_count:
        return {
            "peak_threshold": thr,
            "n_peak": n_peak,
            "peak_hours": df.index[peak_mask],
            "stats_peak": None,
            "share_peak": None,
            "coverage": float(len(df)) / float(len(P_peak_ref.dropna())),
            "error": f"Too few peak hours for q={q} (n_peak={n_peak})",
        }

    # Stats per meter during peak hours
    rows = []
    for name in other_avg.keys():
        x = df.loc[peak_mask, name].to_numpy(dtype=float)
        rows.append({
            "meter": name,
            "mean_kW": float(np.mean(x)),
            "median_kW": float(np.median(x)),
            "p95_kW": float(np.percentile(x, 95)),
            "max_kW": float(np.max(x)),
        })

    stats_peak = pd.DataFrame(rows).set_index("meter").sort_values("mean_kW", ascending=False)

    # Energy-weighted share during peak hours
    denom = float(df.loc[peak_mask, "total"].sum())
    share_rows = []
    known_sum = df.loc[peak_mask, list(other_avg.keys())].sum(axis=1)
    other_power = (df.loc[peak_mask, "total"] - known_sum).clip(lower=0.0)

    for name in other_avg.keys():
        share_rows.append({
            "meter": name,
            "energy_weighted_share": float(df.loc[peak_mask, name].sum() / (denom + 1e-12)),
        })

    share_rows.append({
        "meter": "UNACCOUNTED_OTHER",
        "energy_weighted_share": float(other_power.sum() / (denom + 1e-12)),
    })

    share_peak = pd.DataFrame(share_rows).set_index("meter").sort_values(
        "energy_weighted_share", ascending=False
    )

    return {
        "peak_threshold": thr,
        "n_peak": n_peak,
        "peak_hours": df.index[peak_mask],
        "stats_peak": stats_peak,
        "share_peak": share_peak,
        "coverage": float(len(df)) / float(len(P_peak_ref.dropna())),
    }