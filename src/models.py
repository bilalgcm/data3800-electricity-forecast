"""Step 5: baseline, linear regression and random forest, evaluated on a held-out period.

Owner: Person 4
Input:  data/processed/dataset.csv
Output: results table printed and saved to figures/results.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config

FEATURES = ["temp_c", "hour", "weekday", "month", "is_weekend", "is_holiday",
            "load_lag_24h", "load_lag_168h"]
TARGET = "load_mw"


def split(df):
    """Time-based split: train on the past, test on the future."""
    train = df[df.index < config.TEST_START]
    test = df[df.index >= config.TEST_START]
    return train, test


def score(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def main():
    df = pd.read_csv(config.PROCESSED / "dataset.csv", index_col="time_utc", parse_dates=True)
    train, test = split(df)
    print(f"Train: {len(train)} rows, test: {len(test)} rows")

    results = []

    # Baseline: same hour last week.
    results.append(score("Baseline (same hour last week)", test[TARGET], test["load_lag_168h"]))

    # Benchmark: ENTSO-E's official day-ahead forecast. Not used as a feature,
    # only as a reference for how good a professional forecast is.
    if "forecast_mw" in test:
        results.append(score("ENTSO-E day-ahead forecast", test[TARGET], test["forecast_mw"]))

    # Linear regression. Hour, weekday and month are one-hot encoded so the model
    # does not treat them as ordered numbers.
    X_train = pd.get_dummies(train[FEATURES], columns=["hour", "weekday", "month"])
    X_test = pd.get_dummies(test[FEATURES], columns=["hour", "weekday", "month"])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    lr = LinearRegression().fit(X_train, train[TARGET])
    results.append(score("Linear regression", test[TARGET], lr.predict(X_test)))

    # Random forest can use the raw features directly.
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(train[FEATURES], train[TARGET])
    results.append(score("Random forest", test[TARGET], rf.predict(test[FEATURES])))

    table = pd.DataFrame(results).round(1)
    print(table.to_string(index=False))
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.FIGURES / "results.csv", index=False)


if __name__ == "__main__":
    main()
