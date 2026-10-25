"""Tools the agent can call. Each tool is a plain Python function that returns
a dict; chat.py sends the dict back to the LLM as JSON.

get_forecast(date)     -> hourly load forecast for one day, from our random forest
get_actual_load(date)  -> measured hourly load for one day, from ENTSO-E

The forecast works like a real day-ahead forecast: it uses load that is already
measured (48 h and 168 h earlier) and met.no's TEMPERATURE FORECAST, not the
measured temperature we used in the report.
"""
import sys
from datetime import date as Date
from pathlib import Path

import holidays
import joblib
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

MODEL_PATH = config.ROOT / "models" / "rf_final.joblib"
TEST_MAE_MW = 130.4  # random forest MAE on the 2026 test period (figures/results.csv)

# Oslo Blindern, the same place as our training temperature (Frost SN18700).
BLINDERN = {"lat": 59.9423, "lon": 10.7200, "altitude": 94}
MET_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
# met.no requires a User-Agent that identifies the app and a contact.
MET_HEADERS = {"User-Agent": "data3800-electricity-forecast github.com/bilalgcm"}

_model = None


# ---------- data fetchers (kept separate so they are easy to test) ----------

def _entsoe_client():
    from entsoe import EntsoePandasClient
    if not config.ENTSOE_API_KEY:
        raise RuntimeError("Missing ENTSOE_API_KEY in .env")
    return EntsoePandasClient(api_key=config.ENTSOE_API_KEY)


def fetch_actual_load(start, end):
    """Hourly actual load in MW, UTC index. start/end: tz-aware timestamps."""
    df = _entsoe_client().query_load(config.BIDDING_ZONE, start=start, end=end)
    s = df["Actual Load"] if isinstance(df, pd.DataFrame) else df
    s.index = s.index.tz_convert("UTC")
    return s.resample("1h").mean().rename("load_mw")


def fetch_entsoe_forecast(start, end):
    """ENTSO-E's own day-ahead load forecast, hourly, UTC index."""
    df = _entsoe_client().query_load_forecast(config.BIDDING_ZONE, start=start, end=end)
    s = df["Forecasted Load"] if isinstance(df, pd.DataFrame) else df
    s.index = s.index.tz_convert("UTC")
    return s.resample("1h").mean().rename("entsoe_mw")


def fetch_temperature_forecast():
    """met.no hourly air temperature forecast for Blindern, UTC index."""
    r = requests.get(MET_URL, params=BLINDERN, headers=MET_HEADERS, timeout=20)
    r.raise_for_status()
    rows = [(item["time"], item["data"]["instant"]["details"]["air_temperature"])
            for item in r.json()["properties"]["timeseries"]]
    s = pd.Series(dict(rows), name="temp_c")
    s.index = pd.to_datetime(s.index, utc=True)
    return s


# ---------- helpers ----------

def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("No saved model. Run: python src/train_final_model.py")
        _model = joblib.load(MODEL_PATH)
    return _model


def _day_hours_utc(day):
    """All hours of one local (Oslo) day, as UTC timestamps. 23 or 25 on DST days."""
    start = pd.Timestamp(day, tz=config.TZ)
    end = start + pd.Timedelta(days=1)
    return pd.date_range(start, end, freq="1h", inclusive="left").tz_convert("UTC")


def _parse_date(text):
    try:
        return Date.fromisoformat(text)
    except (TypeError, ValueError):
        raise ValueError(f"Date must be YYYY-MM-DD, got {text!r}")


# ---------- tools ----------

def get_forecast(date):
    """Forecast hourly load in NO1 for one local day (today or tomorrow)."""
    day = _parse_date(date)
    today = pd.Timestamp.now(tz=config.TZ).date()
    if day < today:
        return {"error": f"{day} is in the past. Use get_actual_load for past days."}
    if day > today + pd.Timedelta(days=1):
        return {"error": "Can only forecast today or tomorrow: the model needs load from "
                         "48 hours earlier, which is not measured yet for later days."}

    hours = _day_hours_utc(day)
    # Load from 7 days before the target day up to now covers both lags.
    load = fetch_actual_load(hours[0] - pd.Timedelta(hours=168), pd.Timestamp.now(tz="UTC"))
    temp = fetch_temperature_forecast()

    bundle = _load_model()
    local = hours.tz_convert(config.TZ)
    no_holidays = holidays.Norway(years=day.year)
    X = pd.DataFrame({
        "temp_c": temp.reindex(hours).to_numpy(),
        "hour": local.hour,
        "weekday": local.dayofweek,
        "month": local.month,
        "is_weekend": (local.dayofweek >= 5).astype(int),
        "is_holiday": int(day in no_holidays),
        "load_lag_48h": load.reindex(hours - pd.Timedelta(hours=48)).to_numpy(),
        "load_lag_168h": load.reindex(hours - pd.Timedelta(hours=168)).to_numpy(),
    }, index=hours)[bundle["features"]]

    ok = X.notna().all(axis=1)
    preds = pd.Series(float("nan"), index=hours)
    if ok.any():
        preds[ok] = bundle["model"].predict(X[ok])

    try:
        entsoe = fetch_entsoe_forecast(hours[0], hours[-1] + pd.Timedelta(hours=1)).reindex(hours)
    except Exception:
        entsoe = pd.Series(float("nan"), index=hours)

    rows = []
    for t in hours:
        rows.append({
            "time_local": t.tz_convert(config.TZ).strftime("%Y-%m-%d %H:%M"),
            "forecast_mw": None if pd.isna(preds[t]) else round(float(preds[t])),
            "temperature_forecast_c": None if pd.isna(X.at[t, "temp_c"]) else round(float(X.at[t, "temp_c"]), 1),
            "entsoe_forecast_mw": None if pd.isna(entsoe[t]) else round(float(entsoe[t])),
        })
    missing = int((~ok).sum())
    valid = preds.dropna()
    return {
        "date": str(day),
        "zone": "NO1",
        "hours": rows,
        "daily_mean_forecast_mw": round(float(valid.mean())) if len(valid) else None,
        "peak": None if valid.empty else {
            "time_local": valid.idxmax().tz_convert(config.TZ).strftime("%H:%M"),
            "forecast_mw": round(float(valid.max()))},
        "hours_without_forecast": missing,
        "note": (f"Random forest trained {bundle['trained_from'][:10]} to {bundle['trained_to'][:10]}. "
                 f"Typical error (MAE on the 2026 test period): about {TEST_MAE_MW:.0f} MW. "
                 "Temperature is met.no's forecast for Oslo Blindern."
                 + (f" {missing} hours have no forecast because input data was missing." if missing else "")),
    }


def get_actual_load(date):
    """Measured hourly load in NO1 for one local day, from ENTSO-E."""
    day = _parse_date(date)
    today = pd.Timestamp.now(tz=config.TZ).date()
    if day > today:
        return {"error": f"{day} is in the future, so there is no measured load yet."}
    hours = _day_hours_utc(day)
    load = fetch_actual_load(hours[0], hours[-1] + pd.Timedelta(hours=1)).reindex(hours)
    rows = [{"time_local": t.tz_convert(config.TZ).strftime("%Y-%m-%d %H:%M"),
             "load_mw": None if pd.isna(v) else round(float(v))} for t, v in load.items()]
    valid = load.dropna()
    return {
        "date": str(day),
        "zone": "NO1",
        "hours": rows,
        "daily_mean_mw": round(float(valid.mean())) if len(valid) else None,
        "hours_missing": int(load.isna().sum()),
        "note": "Source: ENTSO-E Transparency Platform, actual total load.",
    }


# Tool descriptions the LLM sees. Clear descriptions matter more than clever code.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "get_forecast",
        "description": ("Forecast hourly electricity consumption (MW) in NO1 for today or "
                        "tomorrow, using the project's random forest with met.no's temperature "
                        "forecast. Also returns ENTSO-E's official forecast for comparison."),
        "parameters": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": "Local Oslo date, YYYY-MM-DD"}},
            "required": ["date"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_actual_load",
        "description": ("Measured hourly electricity consumption (MW) in NO1 for one past day "
                        "or today, from ENTSO-E. Use it to check how a forecast turned out."),
        "parameters": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": "Local Oslo date, YYYY-MM-DD"}},
            "required": ["date"],
            "additionalProperties": False,
        },
    },
]

TOOLS = {"get_forecast": get_forecast, "get_actual_load": get_actual_load}
