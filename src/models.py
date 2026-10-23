"""Step 5: baseline, linear regression and random forest, evaluated on a held-out period.

Owner: Person 4
Input:  data/processed/dataset.csv
Output: figures/results.csv, figures/feature_importance.csv,
        figures/model_forecast_vs_actual_week.png, figures/model_feature_importance.png,
        data/processed/predictions_test.csv
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config
import plot_style as ps

FEATURES = ["temp_c", "hour", "weekday", "month", "is_weekend", "is_holiday",
            "load_lag_48h", "load_lag_168h"]
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

    # Save test-period predictions so figures (and later tools) can reuse them.
    preds = pd.DataFrame({
        "actual_mw": test[TARGET],
        "random_forest_mw": rf.predict(test[FEATURES]),
        "linear_regression_mw": lr.predict(X_test),
        "baseline_mw": test["load_lag_168h"],
    }, index=test.index)
    if "forecast_mw" in test:
        preds["entsoe_mw"] = test["forecast_mw"]
    preds.to_csv(config.PROCESSED / "predictions_test.csv")

    plot_test_week(preds)
    plot_feature_importance(rf)


def plot_test_week(preds):
    """Forecast vs actual for one week in the test period.

    We pick the first full Monday-to-Sunday week in February: a cold, high-load
    month where errors matter most. Change WEEK_START to show another week.
    """
    WEEK_START = "2026-02-02"  # a Monday
    local = preds.tz_convert(config.TZ)
    start = pd.Timestamp(WEEK_START, tz=config.TZ)
    week = local.loc[start:start + pd.Timedelta(days=6, hours=23)]
    if week.empty:
        print(f"No test data for week starting {WEEK_START}, skipping plot")
        return

    fig, ax = plt.subplots()
    ax.plot(week.index, week["actual_mw"], color=ps.ACTUAL, linewidth=2.5, label="Actual load")
    ax.plot(week.index, week["random_forest_mw"], color=ps.BLUE, label="Random forest (ours)")
    if "entsoe_mw" in week:
        ax.plot(week.index, week["entsoe_mw"], color=ps.ORANGE, linestyle="--",
                label="ENTSO-E day-ahead forecast")
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=config.TZ))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=config.TZ))
    ax.set_title(f"Forecast vs actual, week of {pd.Timestamp(WEEK_START):%d %B %Y}")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=3)
    mae_rf = (week["actual_mw"] - week["random_forest_mw"]).abs().mean()
    ax.text(0.01, 0.02, f"Random forest MAE this week: {mae_rf:.0f} MW",
            transform=ax.transAxes, color=ps.TEXT)
    ps.save(fig, "model_forecast_vs_actual_week")


def plot_feature_importance(rf):
    """Impurity-based importance from the random forest (sums to 1)."""
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    labels = {
        "temp_c": "Temperature", "hour": "Hour of day", "weekday": "Day of week",
        "month": "Month", "is_weekend": "Weekend", "is_holiday": "Holiday",
        "load_lag_48h": "Load 48 h earlier", "load_lag_168h": "Load 1 week earlier",
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh([labels[f] for f in imp.index], imp.values, color=ps.BLUE, height=0.65)
    for y, v in enumerate(imp.values):
        ax.text(v + 0.005, y, f"{v:.1%}", va="center", color=ps.TEXT, fontsize=9)
    ax.set_title("What the random forest relies on")
    ax.set_xlabel("Feature importance (share of total)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, min(1, imp.max() * 1.15))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ps.save(fig, "model_feature_importance")
    imp.sort_values(ascending=False).round(4).to_csv(
        config.FIGURES / "feature_importance.csv", header=["importance"])


if __name__ == "__main__":
    main()
