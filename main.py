import pandas as pd
import numpy as np
from fetch_and_clean_data import fetch_and_clean_df
from utils import mstl_decomposition, estimate_P_temp_huber, upper_bound_P_star_hour_of_week, upper_bound_P_temp_from_P_star, peak_times_meter_info
from scoring import calc_flex_score, grid_search_temperature_flex_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from plotting  import plot_flex_diagnostics

from EV_charging_score import compare_ev_peaks

meters = {"hoved" : "EL_hovedmåler_4089_rest",
          "brytehall" : "EL_brytehall_og_garderober",
          "vent_1" : "EL_ventilasjon_Hall2",
          "vent_2" : "EL_ventilasjon_kontor_garderober",
          "vent_3" : "EL_ventilasjon_treningsrom"
          }
periods_hourly_array = {"daily" : 24, "weekly": 24*7}
periods_daily_array = {"weekly": 7}


def set_time_index_from_tid(df: pd.DataFrame, col: str = "Tid (Time)") -> pd.DataFrame:
    # 1) extract start timestamp ("DD.MM.YYYY HH:MM") from "DD.MM.YYYY HH:MM - HH:MM"
    start = (
        df[col]
        .astype(str)
        .str.strip()
        .str.split(" - ", n=1, expand=True)[0]
        .str.strip()
    )

    # 2) parse using explicit format (fast + reliable)
    dt = pd.to_datetime(start, format="%d.%m.%Y %H:%M", errors="coerce")

    # fallback if any weird rows
    if dt.isna().any():
        dt2 = pd.to_datetime(start, dayfirst=True, errors="coerce")
        dt = dt.fillna(dt2)

    out = df.copy()
    out["timestamp"] = dt
    out = out.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

    # (optional) ensure hourly frequency is consistent
    out = out[~out.index.duplicated(keep="first")]
    return out

def build_other_meters_kW_avg_dict(meters: dict[str, str]) -> dict[str, pd.Series]:
    ids = [meters[key] for key in meters if key != "hoved"]
    out: dict[str, pd.Series] = {}

    for meter_id in ids:
        df = fetch_and_clean_df(meter_id, "time", columns=["Snittlast"])  # <- not [["Snittlast"]]
        out[meter_id] = df["Snittlast"].astype(float)  # <- keep as Series with datetime index

    return out

# ---------------- MAIN ----------------
def main() -> None:
    id = meters["hoved"]
    df_main = fetch_and_clean_df(id, "time")
    
    P_avg_s  = df_main["Snittlast"].astype(float)
    P_peak_s = df_main["Peak High"].astype(float)
    T_out_s  = df_main["Utetemperatur"].astype(float)

    # NP arrays for MSTL
    P_avg = P_avg_s.to_numpy()
    P_peak = P_peak_s.to_numpy()
    T = T_out_s.to_numpy()

    # MSTL decomp to isolate hourly and weekly seasonality
    avg_decomp = mstl_decomposition(P_avg, periods=[24, 24 * 7])
    P_star = P_avg - avg_decomp[24] - avg_decomp[24 * 7]  # = trend + residual (schedule removed)

    # Best-estimate P_temp
    P_temp = estimate_P_temp_huber(T, P_star)["P_temp"]

    # Optimistic upper-bound branch 
    ub_star = upper_bound_P_star_hour_of_week(P_avg, T, T_b=18.0, q_low=0.1)
    P_star_ub = ub_star["P_star_ub"]
    P_temp_ub = upper_bound_P_temp_from_P_star(P_star_ub, T, T_b=18.0, warm_q=0.2)

    # Grid search to find max score combos
    q_grid = [0.90, 0.95, 0.97, 0.99]
    f_grid = [0.2, 0.3, 0.4, 0.5, 0.6]
    k_scale_grid = [5.0, 10.0, 20.0, 40.0]

    out_est = grid_search_temperature_flex_score(P_peak, P_temp, q_grid, f_grid, k_scale_grid, min_peak_count=48)
    out_ub  = grid_search_temperature_flex_score(P_peak, P_temp_ub, q_grid, f_grid, k_scale_grid, min_peak_count=48)

    # Print best scores for Huber and Optimistic
    print("\n=== BEST-ESTIMATE (Huber) ===")
    score_est = out_est["best_overall"]
    for k, v in score_est.items():
        print(f"{k}: {v}")

    print("\n=== UPPER-BOUND (Optimistic) ===")
    score_ub = out_ub["best_overall"]
    for k, v in score_ub.items():
        print(f"{k}: {v}")

    ## main meter series for diagostics on how other submeters behave during main meter peaks
    P_avg_other = build_other_meters_kW_avg_dict(meters)
    P_total_avg = df_main["Snittlast"].astype(float)
    P_peak_ref = df_main["Peak High"].astype(float)  # peak definition

    info = peak_times_meter_info(P_total_avg, P_peak_ref, P_avg_other, q=0.99, min_peak_count=24)

    print("peak_threshold:", info["peak_threshold"])
    print("n_peak:", info["n_peak"])
    print("\n--- Peak-hour meter stats (kW) ---")
    print(info["stats_peak"])
    print("\n--- Peak-hour energy-weighted shares ---")
    print(info["share_peak"])

    ## Plotting diagnostics
    # Convert outputs back to Series (for plotting)

    # Other meters as Series
    P_avg_other = build_other_meters_kW_avg_dict(meters)


    # --- plotting ---
    plot_flex_diagnostics(
        df_main=df_main,
        P_avg=P_avg,
        P_peak=P_peak,
        T_out=T,
        P_star=P_star,
        P_temp=P_temp,
        other_avg=P_avg_other,   # can be dict[str, Series] OR dict[str, np.ndarray]
        q=0.99,
        top_n_peaks=48,
        periods=[("2025-01-10", "2025-01-17"), ("2025-07-10", "2025-07-17")],
        show=True,
    )

if __name__ == '__main__':
    main()