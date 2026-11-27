"""Step 3: clean, merge and add calendar features.

Owner: Person 2
Input:  data/raw/consumption_no1.csv, data/raw/temperature_blindern.csv
Output: data/processed/dataset.csv, figures/data_quality.csv
"""
import pandas as pd

import config
from features import calendar_features


def load_raw():
    load = pd.read_csv(config.RAW / "consumption_no1.csv", index_col="time_utc", parse_dates=True)
    temp = pd.read_csv(config.RAW / "temperature_blindern.csv", index_col="time_utc", parse_dates=True)
    return load, temp


def quality_report(load, temp):
    """Check the raw files against things we already know the answer to.

    An hourly series has exactly one row per hour, so the number of rows, the
    number of duplicates and the hours per year can all be checked without
    looking at a single value.
    """
    rows = []
    for name, df, column, low, high in [("consumption_no1.csv", load, "load_mw", 0, 15000),
                                        ("temperature_blindern.csv", temp, "temp_c", -40, 40)]:
        expected = pd.date_range(df.index.min(), df.index.max(), freq="1h")
        values = df[column]
        gaps = values.reindex(expected).isna()
        longest_gap = gaps.groupby((~gaps).cumsum()).sum().max()
        rows.append({
            "file": name,
            "first_hour_utc": f"{df.index.min():%Y-%m-%d %H:%M}",
            "last_hour_utc": f"{df.index.max():%Y-%m-%d %H:%M}",
            "rows": len(df),
            "expected_hours": len(expected),
            "duplicate_timestamps": int(df.index.duplicated().sum()),
            "missing_hours": int(gaps.sum()),
            "longest_gap_hours": int(longest_gap),
            "values_outside_range": int(((values <= low) | (values > high)).sum()),
            "min": round(float(values.min()), 1),
            "max": round(float(values.max()), 1),
        })
    report = pd.DataFrame(rows).set_index("file").T
    print("Data quality check on the raw files:\n", report.to_string())

    local_year = load.index.tz_convert(config.TZ).year
    per_year = load["load_mw"].groupby(local_year).count()
    print("Hours with load per year (a full year has 8760, a leap year 8784):\n", per_year.to_string())

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.FIGURES / "data_quality.csv")
    return report


def clean(df):
    """Report and fill small gaps. Large gaps are left as NaN and dropped later."""
    full = pd.date_range(df.index.min(), df.index.max(), freq="1h", tz="UTC")
    df = df.reindex(full)
    df.index.name = "time_utc"
    print("Missing values before filling:\n", df.isna().sum())

    # Fill gaps of up to 3 hours by linear interpolation.
    df = df.interpolate(limit=3, limit_area="inside")
    print("Missing values after filling:\n", df.isna().sum())
    return df


def add_features(df):
    df = df.join(calendar_features(df.index))

    # Lagged consumption: what we would know when forecasting the next day.
    df["load_lag_48h"] = df["load_mw"].shift(48)
    df["load_lag_168h"] = df["load_mw"].shift(168)   # same hour last week
    return add_daytype_features(df)


def add_daytype_features(df):
    """Fix for the weekend error: the 48 h lag on a Monday is Saturday's load,
    and on a Sunday it is Friday's. When the lag comes from a different kind of
    day (workday vs weekend/holiday), use the same hour last week instead.

    day_type:            0 = workday, 1 = weekend or holiday
    lag48_same_daytype:  1 if the 48 h lag comes from the same day type
    load_lag_daytype:    48 h lag if same day type, else the 168 h lag
    """
    df["day_type"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)
    same = df["day_type"].shift(48) == df["day_type"]
    df["lag48_same_daytype"] = same.astype(int)
    df["load_lag_daytype"] = df["load_lag_48h"].where(same, df["load_lag_168h"])
    return df


def main():
    load, temp = load_raw()
    quality_report(load, temp)
    df = load.join(temp, how="outer")
    df = clean(df)
    df = add_features(df)
    df = df.dropna()

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED / "dataset.csv"
    df.to_csv(out)
    print(f"Saved {len(df)} rows and {df.shape[1]} columns to {out}")


if __name__ == "__main__":
    main()