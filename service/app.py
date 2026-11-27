"""Prediction service: the saved random forest behind a web address (week 45).

Start it from the repo root:
    python src/train_final_model.py                       # writes models/rf_final.joblib
    uvicorn service.app:app --host 127.0.0.1 --port 8000
Then open http://127.0.0.1:8000/docs, or call it:
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
      -d '{"time_local": "2026-12-01T18:00", "temp_c": -3.5, "load_lag_48h": 5600, "load_lag_168h": 5450}'

The caller sends what a caller knows: the hour, a temperature and two earlier
load values. Hour of day, weekday, month, weekend and holiday are worked out
here, with the same function that built the training data (src/features.py).
"""
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402
from features import calendar_features  # noqa: E402

MODEL_PATH = config.ROOT / "models" / "rf_final.joblib"

app = FastAPI(title="NO1 load forecast service")
# Loaded once at start-up, not on every request.
bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class Hour(BaseModel):
    """One hour to forecast. Values outside the bounds are rejected with a 422."""
    time_local: datetime = Field(description="Start of the hour in Oslo time, e.g. 2026-12-01T18:00")
    temp_c: float = Field(ge=-40, le=40, description="Air temperature at Oslo Blindern, °C")
    load_lag_48h: float = Field(gt=0, le=15000, description="NO1 load 48 hours earlier, MW")
    load_lag_168h: float = Field(gt=0, le=15000, description="NO1 load one week earlier, MW")


@app.get("/health")
def health():
    if bundle is None:
        return {"status": "no model", "hint": "run python src/train_final_model.py"}
    return {"status": "ok", "model": "random forest", "features": bundle["features"],
            "trained_from": bundle["trained_from"][:10], "trained_to": bundle["trained_to"][:10]}


@app.post("/predict")
def predict(hour: Hour):
    if bundle is None:
        raise HTTPException(503, "No saved model. Run: python src/train_final_model.py")
    try:
        index = pd.DatetimeIndex([hour.time_local.replace(tzinfo=None)]).tz_localize(config.TZ)
    except Exception:
        # An hour that does not exist or exists twice (the night the clocks change).
        raise HTTPException(422, "time_local is not a unique hour in Oslo time (clock change)")
    row = calendar_features(index)
    row["temp_c"] = hour.temp_c
    row["load_lag_48h"] = hour.load_lag_48h
    row["load_lag_168h"] = hour.load_lag_168h
    forecast = float(bundle["model"].predict(row[bundle["features"]])[0])
    return {"time_local": index[0].strftime("%Y-%m-%d %H:%M"), "zone": "NO1",
            "forecast_mw": round(forecast, 1)}
