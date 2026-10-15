"""Step 2: fetch hourly air temperature for Oslo Blindern from the Frost API.

Owner: Person 2
Output: data/raw/temperature_blindern.csv  (columns: time_utc, temp_c)
"""
import pandas as pd
import requests

import config

URL = "https://frost.met.no/observations/v0.jsonld"


def fetch_period(start, end):
    params = {
        "sources": config.FROST_STATION,
        "elements": "air_temperature",
        "referencetime": f"{start}/{end}",
        "timeresolutions": "PT1H",
    }
    r = requests.get(URL, params=params, auth=(config.FROST_CLIENT_ID, ""), timeout=60)
    r.raise_for_status()

    rows = []
    for item in r.json()["data"]:
        for obs in item["observations"]:
            rows.append({"time_utc": item["referenceTime"], "temp_c": obs["value"]})
    return pd.DataFrame(rows)


def main():
    if not config.FROST_CLIENT_ID:
        raise SystemExit("Missing FROST_CLIENT_ID in .env")

    # Frost limits how much data one request returns, so we ask for one year at a time.
    bounds = pd.date_range(config.START, config.END, freq="YS").strftime("%Y-%m-%d").tolist()
    if bounds[-1] != config.END:
        bounds.append(config.END)

    parts = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        print(f"Fetching {start} to {end}")
        parts.append(fetch_period(start, end))

    df = pd.concat(parts, ignore_index=True)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    # A station can report several sensors for the same hour; keep one value per hour.
    df = df.groupby("time_utc")["temp_c"].mean()

    config.RAW.mkdir(parents=True, exist_ok=True)
    out = config.RAW / "temperature_blindern.csv"
    df.to_csv(out)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
