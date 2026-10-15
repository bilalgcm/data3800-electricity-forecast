"""Step 1: fetch actual total load (consumption) for NO1 from ENTSO-E.

Owner: Person 1
Output: data/raw/consumption_no1.csv  (columns: time_utc, load_mw)
"""
import pandas as pd
from entsoe import EntsoePandasClient

import config


def fetch_year(client, start, end):
    """ENTSO-E is easiest to query in chunks of at most one year."""
    df = client.query_load(config.BIDDING_ZONE, start=start, end=end)
    # query_load returns a DataFrame with one column, usually 'Actual Load'
    return df.iloc[:, 0].rename("load_mw")


def main():
    if not config.ENTSOE_API_KEY:
        raise SystemExit("Missing ENTSOE_API_KEY in .env")

    client = EntsoePandasClient(api_key=config.ENTSOE_API_KEY)
    start = pd.Timestamp(config.START, tz=config.TZ)
    end = pd.Timestamp(config.END, tz=config.TZ)

    parts = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + pd.DateOffset(years=1), end)
        print(f"Fetching {chunk_start.date()} to {chunk_end.date()}")
        parts.append(fetch_year(client, chunk_start, chunk_end))
        chunk_start = chunk_end

    load = pd.concat(parts)
    load = load[~load.index.duplicated(keep="first")]
    load.index = load.index.tz_convert("UTC")
    load.index.name = "time_utc"

    # Data may come at 15-minute resolution; we work with hourly means.
    load = load.resample("1h").mean()

    config.RAW.mkdir(parents=True, exist_ok=True)
    out = config.RAW / "consumption_no1.csv"
    load.to_csv(out)
    print(f"Saved {len(load)} rows to {out}")


if __name__ == "__main__":
    main()
