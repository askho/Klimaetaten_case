import pandas as pd


DEFAULT_COLUMNS = [
    "Tid (Time)",
    "Energi",
    "Peak High",
    "Snittlast",
    "Utetemperatur",
]


def clean_energy_df(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = _normalize_column_names(cleaned.columns)
    cleaned = cleaned.rename(columns={"tid_time": "time", "coâ‚‚": "co2"})
    cleaned = _clean_string_columns(cleaned)
    cleaned = _parse_time_columns(cleaned)
    cleaned = _convert_known_numeric_columns(cleaned)
    return cleaned


def _normalize_column_names(columns: pd.Index) -> pd.Index:
    return (
        columns.str.strip()
        .str.replace('"', "", regex=False)
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )


def _clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    object_columns = df.select_dtypes(include="object").columns

    for column_name in object_columns:
        df[column_name] = (
            df[column_name]
            .astype(str)
            .str.strip()
            .str.replace('"', "", regex=False)
        )
        df[column_name] = df[column_name].replace({"": pd.NA, "nan": pd.NA})

    return df


def _parse_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "time" not in df.columns:
        return df

    time_parts = df["time"].str.split(" - ", expand=True)
    df["start_time"] = pd.to_datetime(
        time_parts[0],
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )
    df["end_time"] = pd.to_datetime(
        df["start_time"].dt.strftime("%d.%m.%Y") + " " + time_parts[1],
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )

    crosses_midnight = df["end_time"] < df["start_time"]
    df.loc[crosses_midnight, "end_time"] = df.loc[crosses_midnight, "end_time"] + pd.Timedelta(days=1)
    return df


def _convert_known_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "energi",
        "co2",
        "kostnad",
        "peak_high",
        "snittlast",
        "utetemperatur",
    ]

    for column_name in numeric_columns:
        if column_name not in df.columns:
            continue

        normalized = (
            df[column_name]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .replace({"<NA>": pd.NA, "nan": pd.NA})
        )
        df[column_name] = pd.to_numeric(normalized, errors="coerce")

    return df


def clean_numeric_columns_auto(
    df: pd.DataFrame,
    allow_negative: list[str] | None = None,
    exclude_cols: list[str] | None = None,
    min_numeric_ratio: float = 0.8,
    interpolate: bool = True,
) -> pd.DataFrame:
    cleaned = df.copy()
    allowed_negative_columns = allow_negative or []
    excluded_columns = exclude_cols or []

    detected_numeric_columns = []
    for column_name in cleaned.columns:
        if column_name in excluded_columns:
            continue

        if pd.api.types.is_numeric_dtype(cleaned[column_name]):
            detected_numeric_columns.append(column_name)
            continue

        converted = _try_convert_series_to_numeric(cleaned[column_name])
        if _is_mostly_numeric(cleaned[column_name], converted, min_numeric_ratio):
            cleaned[column_name] = converted
            detected_numeric_columns.append(column_name)

    _replace_invalid_negative_values(cleaned, detected_numeric_columns, allowed_negative_columns)

    if interpolate and detected_numeric_columns:
        cleaned[detected_numeric_columns] = cleaned[detected_numeric_columns].interpolate(
            method="linear",
            limit_direction="both",
        )

    return cleaned


def _try_convert_series_to_numeric(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(normalized, errors="coerce")


def _is_mostly_numeric(
    original_series: pd.Series,
    converted_series: pd.Series,
    min_numeric_ratio: float,
) -> bool:
    normalized_original = (
        original_series.astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
    )

    original_non_missing = normalized_original.notna().sum()
    converted_non_missing = converted_series.notna().sum()

    if original_non_missing == 0:
        return False

    return (converted_non_missing / original_non_missing) >= min_numeric_ratio


def _replace_invalid_negative_values(
    df: pd.DataFrame,
    numeric_columns: list[str],
    allowed_negative_columns: list[str],
) -> None:
    for column_name in numeric_columns:
        if column_name in allowed_negative_columns:
            continue
        df.loc[df[column_name] < 0, column_name] = pd.NA


def fetch_and_clean_df(
    meter_id: str,
    sampletime: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    selected_columns = columns or DEFAULT_COLUMNS
    filename = f"data/{meter_id}_{sampletime}.csv"

    raw = pd.read_csv(
        filename,
        sep=";",
        decimal=",",
        quotechar='"',
        na_values=["", '""'],
        skiprows=3,
    )

    cleaned = clean_numeric_columns_auto(
        raw[selected_columns],
        allow_negative=["Utetemperatur"],
        min_numeric_ratio=0.8,
        interpolate=True,
    )
    return cleaned[selected_columns].copy()
