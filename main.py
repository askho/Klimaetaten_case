import numpy as np
import pandas as pd

from fetch_and_clean_data import fetch_and_clean_df
from plotting import plot_flex_diagnostics
from scoring import grid_search_temperature_flex_score
from utils import (
    estimate_P_temp_huber,
    mstl_decomposition,
    peak_times_meter_info,
    upper_bound_P_star_hour_of_week,
    upper_bound_P_temp_from_P_star,
)


METERS = {
    "hoved": "EL_hovedmåler_4089_rest",
    "brytehall": "EL_brytehall_og_garderober",
    "vent_1": "EL_ventilasjon_Hall2",
    "vent_2": "EL_ventilasjon_kontor_garderober",
    "vent_3": "EL_ventilasjon_treningsrom",
}

SEASONAL_PERIODS = [24, 24 * 7]
PEAK_QUANTILES = [0.90, 0.95, 0.97, 0.99]
REALIZABLE_FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6]
FLEX_SCALE_OPTIONS = [5.0, 10.0, 20.0, 40.0]


def load_main_meter_data() -> pd.DataFrame:
    return fetch_and_clean_df(METERS["hoved"], "time")


def get_main_series(df_main: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    average_power = df_main["Snittlast"].astype(float)
    peak_power = df_main["Peak High"].astype(float)
    outdoor_temperature = df_main["Utetemperatur"].astype(float)
    return average_power, peak_power, outdoor_temperature


def build_submeter_average_power(meters: dict[str, str]) -> dict[str, pd.Series]:
    submeter_average_power: dict[str, pd.Series] = {}

    for meter_name, meter_id in meters.items():
        if meter_name == "hoved":
            continue

        meter_data = fetch_and_clean_df(meter_id, "time", columns=["Snittlast"])
        submeter_average_power[meter_id] = meter_data["Snittlast"].astype(float)

    return submeter_average_power


def estimate_schedule_adjusted_power(
    average_power: np.ndarray,
    outdoor_temperature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    decomposition = mstl_decomposition(average_power, periods=SEASONAL_PERIODS)
    seasonal_daily = decomposition[24]
    seasonal_weekly = decomposition[24 * 7]

    # Removing the repeating daily and weekly schedule leaves a load signal that is
    # easier to interpret as weather-driven plus residual variation.
    baseline_adjusted_power = average_power - seasonal_daily - seasonal_weekly

    estimated_temperature_power = estimate_P_temp_huber(
        outdoor_temperature,
        baseline_adjusted_power,
    )["P_temp"]

    upper_bound_result = upper_bound_P_star_hour_of_week(
        average_power,
        outdoor_temperature,
        T_b=18.0,
        q_low=0.1,
    )
    optimistic_temperature_power = upper_bound_P_temp_from_P_star(
        upper_bound_result["P_star_ub"],
        outdoor_temperature,
        T_b=18.0,
        warm_q=0.2,
    )

    return baseline_adjusted_power, estimated_temperature_power, optimistic_temperature_power


def run_score_search(
    peak_power: np.ndarray,
    temperature_power: np.ndarray,
) -> dict[str, object]:
    return grid_search_temperature_flex_score(
        peak_power,
        temperature_power,
        PEAK_QUANTILES,
        REALIZABLE_FRACTIONS,
        FLEX_SCALE_OPTIONS,
        min_peak_count=48,
    )


def print_best_score(label: str, result: dict[str, object]) -> None:
    print(f"\n=== {label} ===")

    best_score = result["best_overall"]
    if best_score is None:
        print("No feasible score found.")
        return

    for key, value in best_score.items():
        print(f"{key}: {value}")


def print_peak_hour_meter_summary(
    total_average_power: pd.Series,
    peak_reference_power: pd.Series,
    submeter_average_power: dict[str, pd.Series],
) -> None:
    peak_info = peak_times_meter_info(
        total_average_power,
        peak_reference_power,
        submeter_average_power,
        q=0.99,
        min_peak_count=24,
    )

    print("peak_threshold:", peak_info["peak_threshold"])
    print("n_peak:", peak_info["n_peak"])
    print("\n--- Peak-hour meter stats (kW) ---")
    print(peak_info["stats_peak"])
    print("\n--- Peak-hour energy-weighted shares ---")
    print(peak_info["share_peak"])


def main() -> None:
    df_main = load_main_meter_data()
    average_power_series, peak_power_series, outdoor_temperature_series = get_main_series(df_main)

    average_power = average_power_series.to_numpy()
    peak_power = peak_power_series.to_numpy()
    outdoor_temperature = outdoor_temperature_series.to_numpy()

    (
        baseline_adjusted_power,
        estimated_temperature_power,
        optimistic_temperature_power,
    ) = estimate_schedule_adjusted_power(average_power, outdoor_temperature)

    estimated_score_result = run_score_search(peak_power, estimated_temperature_power)
    optimistic_score_result = run_score_search(peak_power, optimistic_temperature_power)

    print_best_score("BEST-ESTIMATE (Huber)", estimated_score_result)
    print_best_score("UPPER-BOUND (Optimistic)", optimistic_score_result)

    submeter_average_power = build_submeter_average_power(METERS)
    print_peak_hour_meter_summary(
        average_power_series,
        peak_power_series,
        submeter_average_power,
    )

    plot_flex_diagnostics(
        df_main=df_main,
        P_avg=average_power,
        P_peak=peak_power,
        T_out=outdoor_temperature,
        P_star=baseline_adjusted_power,
        P_temp=estimated_temperature_power,
        other_avg=submeter_average_power,
        q=0.99,
        top_n_peaks=48,
        periods=[("2025-01-10", "2025-01-17"), ("2025-07-10", "2025-07-17")],
        show=True,
    )


if __name__ == "__main__":
    main()
