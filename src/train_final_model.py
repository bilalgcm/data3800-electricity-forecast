"""Train the final random forest on ALL data and save it for the agent.

models.py trains on 2023 to 2025 and tests on 2026, which is how we measure
accuracy. For real forecasts we want the model to learn from every hour we
have, so this script retrains the same model on the full period.

Input:  data/processed/dataset.csv
Output: models/rf_final.joblib (not committed, rerun this script to recreate)
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config
from models import FEATURES, TARGET

MODEL_PATH = config.ROOT / "models" / "rf_final.joblib"


def main():
    df = pd.read_csv(config.PROCESSED / "dataset.csv", index_col="time_utc", parse_dates=True)
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(df[FEATURES], df[TARGET])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "features": FEATURES,
                 "trained_from": str(df.index.min()), "trained_to": str(df.index.max())},
                MODEL_PATH, compress=3)
    print(f"Trained on {len(df)} hours ({df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d})")
    print(f"Saved {MODEL_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
