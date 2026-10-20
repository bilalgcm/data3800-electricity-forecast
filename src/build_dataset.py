"""Step 3: clean, merge and add calendar features.

Owner: Person 2
Input:  data/raw/consumption_no1.csv, data/raw/temperature_blindern.csv
Output: data/processed/dataset.csv
"""
import holidays
import pandas as pd

import config


def load_raw():
    load = pd.read_csv(config.RAW / "consumption_no1.csv", index_col="time_utc", parse_dates=True)
    temp = pd.read_csv(config.RAW / "temperature_blindern.csv", index_col="time_utc", parse_dates=True)
    return load, temp


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
    local = df.index.tz_convert(config.TZ)
    no_holidays = holidays.Norway(years=range(local.year.min(), local.year.max() + 1))

    df["hour"] = local.hour
    df["weekday"] = local.dayofweek          # 0 = Monday
    df["month"] = local.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_holiday"] = [int(d in no_holidays) for d in local.date]

    # Lagged consumption: what we would know when forecasting the next day.
    df["load_lag_48h"] = df["load_mw"].shift(48)
    df["load_lag_168h"] = df["load_mw"].shift(168)   # same hour last week
    return df


def main():
    load, temp = load_raw()
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
