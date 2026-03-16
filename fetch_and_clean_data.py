import pandas as pd



def clean_energy_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace('"', '', regex=False)
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )

    # Rename special column names
    rename_map = {
        "tid_time": "time",
        "co₂": "co2",
    }
    df = df.rename(columns=rename_map)

    # Clean all object/string cells
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace('"', '', regex=False)
        )
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA})

    # Parse time interval column
    if "time" in df.columns:
        parts = df["time"].str.split(" - ", expand=True)

        df["start_time"] = pd.to_datetime(
            parts[0],
            format="%d.%m.%Y %H:%M",
            errors="coerce"
        )

        df["end_time"] = pd.to_datetime(
            df["start_time"].dt.strftime("%d.%m.%Y") + " " + parts[1],
            format="%d.%m.%Y %H:%M",
            errors="coerce"
        )

        # Handle intervals crossing midnight
        mask = df["end_time"] < df["start_time"]
        df.loc[mask, "end_time"] = df.loc[mask, "end_time"] + pd.Timedelta(days=1)

    # Convert numeric columns with comma decimals
    numeric_cols = ["energi", "co2", "kostnad", "peak_high", "snittlast", "utetemperatur"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .replace({"<NA>": pd.NA, "nan": pd.NA})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_numeric_columns_auto(
    df: pd.DataFrame,
    allow_negative: list[str] | None = None,
    exclude_cols: list[str] | None = None,
    min_numeric_ratio: float = 0.8,
    interpolate: bool = True
) -> pd.DataFrame:
    df = df.copy()

    if allow_negative is None:
        allow_negative = []

    if exclude_cols is None:
        exclude_cols = []

    detected_numeric_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        # Already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            detected_numeric_cols.append(col)
            continue

        # Try converting string/object columns to numeric
        cleaned = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
            .str.replace(",", ".", regex=False)
        )

        converted = pd.to_numeric(cleaned, errors="coerce")

        # Count as numeric if enough values convert successfully
        non_missing_original = cleaned.notna().sum()
        non_missing_converted = converted.notna().sum()

        ratio = (
            non_missing_converted / non_missing_original
            if non_missing_original > 0 else 0
        )

        if ratio >= min_numeric_ratio:
            df[col] = converted
            detected_numeric_cols.append(col)

    # Remove invalid negative values
    for col in detected_numeric_cols:
        if col not in allow_negative:
            df.loc[df[col] < 0, col] = pd.NA

    # Interpolate missing values
    if interpolate and detected_numeric_cols:
        df[detected_numeric_cols] = df[detected_numeric_cols].interpolate(
            method="linear",
            limit_direction="both"
        )

    return df


def fetch_and_clean_df(meter_id: str, 
             sampletime: str,
             columns = ["Tid (Time)", "Energi", "Peak High","Snittlast", "Utetemperatur"]
             ) -> pd.DataFrame: 
    # File path
    filename = f"data/{meter_id}_{sampletime}.csv"
    num_lines_metadata = 3

    df = pd.read_csv(
        filename,
        sep=";",
        decimal=",",
        quotechar='"',
        na_values=["", '""'],
        skiprows=num_lines_metadata)
    
    df_clean = clean_numeric_columns_auto(
        df[columns],
        allow_negative=["Utetemperatur"],
        min_numeric_ratio=0.8,
        interpolate=True
    )
    
    return df_clean[columns].copy()