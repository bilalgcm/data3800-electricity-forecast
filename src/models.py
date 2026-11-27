"""Step 5: train the models, choose settings on a validation year, and score on the test period.

Models (all from the course): a naive baseline, linear regression, a decision
tree, k nearest neighbours and a random forest. ENTSO-E's official day-ahead
forecast is included as a professional benchmark (it is never used as a feature).

How the data is used (never shuffled, because this is a time series):
    train       2023 to 2024   fit the models
    validation  2025           choose settings (tree depth, k, leaf size)
    test        2026           touched once, for the final table
After the settings are chosen, each model is refitted on train + validation
(2023 to 2025) and scored on the test period.

Owner: Person 4
Input:  data/processed/dataset.csv
Output: figures/results.csv, figures/tuning.csv, figures/model_tuning.png,
        figures/feature_importance.csv, figures/model_forecast_vs_actual_week.png,
        figures/model_feature_importance.png, data/processed/predictions_test.csv
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

import config
import plot_style as ps

NUMERIC = ["temp_c", "load_lag_48h", "load_lag_168h"]
CATEGORICAL = ["hour", "weekday", "month"]
BINARY = ["is_weekend", "is_holiday"]
FEATURES = ["temp_c", "hour", "weekday", "month", "is_weekend", "is_holiday",
            "load_lag_48h", "load_lag_168h"]
TARGET = "load_mw"

# Settings tried on the validation year, ordered from the most flexible model
# to the simplest one. None = no depth limit.
TREE_DEPTHS = [None, 20, 16, 12, 10, 8, 6, 4, 2]
KNN_NEIGHBOURS = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100]
RF_MIN_LEAF = [1, 2, 5, 10, 20, 50]

# We pick the simplest setting whose validation MAE is within 2 % of the best.
# Differences smaller than that are noise, and a simpler model overfits less
# and is smaller on disk.
TOLERANCE = 0.02


# ---------- data splits ----------

def load_dataset():
    return pd.read_csv(config.PROCESSED / "dataset.csv", index_col="time_utc", parse_dates=True)


def split(df):
    """Train on the past (2023 to 2025), test on the future (2026)."""
    train = df[df.index < config.TEST_START]
    test = df[df.index >= config.TEST_START]
    return train, test


def split_validation(train):
    """Split the training period again: fit on 2023 to 2024, validate on 2025."""
    fit = train[train.index < config.VAL_START]
    val = train[train.index >= config.VAL_START]
    return fit, val


# ---------- models ----------

def scale_and_encode():
    """Preparation for models that do arithmetic on the features (week 45).

    Numbers are put on a common scale, so that load in MW (thousands) does not
    drown out temperature (tens). Hour, weekday and month are one hot encoded,
    so the model does not read month 12 as "twelve times January".
    """
    return ColumnTransformer([
        ("numbers", StandardScaler(), NUMERIC),
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("flags", "passthrough", BINARY),
    ])


def make_linear():
    return Pipeline([("prepare", scale_and_encode()), ("model", LinearRegression())])


def make_knn(k):
    # Distance based, so scaling is not optional here.
    return Pipeline([("prepare", scale_and_encode()),
                     ("model", KNeighborsRegressor(n_neighbors=k))])


def make_tree(max_depth):
    # Trees split on thresholds, so they need neither scaling nor encoding.
    return DecisionTreeRegressor(max_depth=max_depth, random_state=42)


def make_forest(min_samples_leaf):
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=min_samples_leaf,
                                 n_jobs=-1, random_state=42)


# ---------- scoring ----------

def score(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def tune(fit, val):
    """Try each setting: fit on 2023 to 2024, measure MAE on 2025.

    The training error is saved too. A model whose training error keeps falling
    while the validation error rises is overfitting.
    """
    families = [
        ("Decision tree", "max_depth", TREE_DEPTHS, make_tree),
        ("k nearest neighbours", "n_neighbors", KNN_NEIGHBOURS, make_knn),
        ("Random forest", "min_samples_leaf", RF_MIN_LEAF, make_forest),
    ]
    rows = []
    for family, setting, values, make in families:
        for value in values:
            model = make(value).fit(fit[FEATURES], fit[TARGET])
            rows.append({
                "model": family, "setting": setting,
                "value": "no limit" if value is None else value,
                "MAE_train": mean_absolute_error(fit[TARGET], model.predict(fit[FEATURES])),
                "MAE_validation": mean_absolute_error(val[TARGET], model.predict(val[FEATURES])),
            })
            print(f"  {family:<22} {setting}={rows[-1]['value']!s:<9} "
                  f"train {rows[-1]['MAE_train']:6.1f}  validation {rows[-1]['MAE_validation']:6.1f}")
    table = pd.DataFrame(rows)
    table["chosen"] = False
    for family in table.model.unique():
        t = table[table.model == family]
        good_enough = t[t.MAE_validation <= t.MAE_validation.min() * (1 + TOLERANCE)]
        table.loc[good_enough.index[-1], "chosen"] = True   # last row = simplest
    return table


def chosen(tuning, family, values):
    """The chosen setting for one model family, as the real Python value."""
    best = str(tuning[(tuning.model == family) & tuning.chosen].iloc[0]["value"])
    return next(v for v in values if str("no limit" if v is None else v) == best)


def saved_forest_setting(default=5):
    """min_samples_leaf chosen by the last run of this script (figures/tuning.csv).

    experiments.py and train_final_model.py use it, so every random forest in
    the project has the same settings.
    """
    path = config.FIGURES / "tuning.csv"
    if not path.exists():
        return default
    return chosen(pd.read_csv(path), "Random forest", RF_MIN_LEAF)


def evaluate(train, test, fit, val, tuning):
    """Refit each model on train + validation with the chosen settings and
    score it once on the test period. Returns (table, predictions, fitted models)."""
    depth = chosen(tuning, "Decision tree", TREE_DEPTHS)
    k = chosen(tuning, "k nearest neighbours", KNN_NEIGHBOURS)
    leaf = chosen(tuning, "Random forest", RF_MIN_LEAF)
    print(f"Chosen settings: tree max_depth={depth}, kNN k={k}, forest min_samples_leaf={leaf}")

    models = {
        "Linear regression": make_linear(),
        f"Decision tree (max depth {depth})": make_tree(depth),
        f"k nearest neighbours (k = {k})": make_knn(k),
        "Random forest": make_forest(leaf),
    }
    columns = {"Linear regression": "linear_regression_mw", "Decision tree": "decision_tree_mw",
               "k nearest neighbours": "knn_mw", "Random forest": "random_forest_mw"}
    preds = pd.DataFrame({"actual_mw": test[TARGET]}, index=test.index)
    results = []

    # Baseline: same hour last week. Needs no training.
    r = score("Baseline (same hour last week)", test[TARGET], test["load_lag_168h"])
    r["MAE_validation"] = mean_absolute_error(val[TARGET], val["load_lag_168h"])
    results.append(r)
    preds["baseline_mw"] = test["load_lag_168h"]

    # Benchmark: ENTSO-E's official day-ahead forecast. Not used as a feature,
    # only as a reference for how good a professional forecast is.
    if "forecast_mw" in test:
        r = score("ENTSO-E day-ahead forecast", test[TARGET], test["forecast_mw"])
        r["MAE_validation"] = mean_absolute_error(val[TARGET], val["forecast_mw"])
        results.append(r)
        preds["entsoe_mw"] = test["forecast_mw"]

    fitted = {}
    for name, model in models.items():
        # Validation MAE: fitted on 2023 to 2024, measured on 2025.
        val_mae = mean_absolute_error(
            val[TARGET], model.fit(fit[FEATURES], fit[TARGET]).predict(val[FEATURES]))
        # Test scores: refitted on 2023 to 2025, measured on 2026.
        model.fit(train[FEATURES], train[TARGET])
        pred = model.predict(test[FEATURES])
        r = score(name, test[TARGET], pred)
        r["MAE_validation"] = val_mae
        results.append(r)
        fitted[name] = model
        preds[next(c for key, c in columns.items() if name.startswith(key))] = pred

    table = pd.DataFrame(results)[["model", "MAE_validation", "MAE", "RMSE", "R2"]]
    table[["MAE_validation", "MAE", "RMSE"]] = table[["MAE_validation", "MAE", "RMSE"]].round(1)
    table["R2"] = table["R2"].round(3)
    return table, preds, fitted


def main():
    df = load_dataset()
    train, test = split(df)
    fit, val = split_validation(train)
    print(f"Train: {len(fit)} rows, validation: {len(val)} rows, test: {len(test)} rows")

    # 1. Choose settings on the validation year.
    print("\nTuning on the validation year (2025):")
    tuning = tune(fit, val)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    tuning.round(1).to_csv(config.FIGURES / "tuning.csv", index=False)
    plot_tuning(tuning)

    # 2. Refit on train + validation and score once on the test period.
    table, preds, fitted = evaluate(train, test, fit, val, tuning)
    print("\nFinal scores (MAE, RMSE and R2 are on the 2026 test period):")
    print(table.to_string(index=False))
    table.to_csv(config.FIGURES / "results.csv", index=False)

    # Save test-period predictions so the evaluation step and the agent can reuse them.
    preds.to_csv(config.PROCESSED / "predictions_test.csv")

    plot_test_week(preds)
    plot_feature_importance(fitted["Random forest"])
    return table


# ---------- figures ----------

def plot_tuning(tuning):
    """Training against validation error for each setting (overfitting check)."""
    families = list(tuning.model.unique())
    fig, axes = plt.subplots(1, len(families), figsize=(12, 3.8), sharey=True)
    for ax, family in zip(axes, families):
        t = tuning[tuning.model == family]
        if family == "Decision tree":
            t = t.iloc[::-1]          # shallow to deep, so flexibility grows to the right
        t = t.reset_index(drop=True)
        x = range(len(t))
        ax.plot(x, t.MAE_train, color="#b9b8b2", marker="o", markersize=4, label="Training (2023 to 2024)")
        ax.plot(x, t.MAE_validation, color=ps.BLUE, marker="o", markersize=4, label="Validation (2025)")
        best = t.index[t.chosen][0]
        ax.scatter([best], [t.MAE_validation[best]], s=110, facecolors="none",
                   edgecolors=ps.ORANGE, linewidths=2, zorder=5)
        ax.set_xticks(list(x), [str(v) for v in t.value], fontsize=8)
        ax.set_xlabel(t.setting.iloc[0])
        ax.set_title(family)
    axes[0].set_ylabel("MAE (MW)")
    axes[0].set_ylim(0)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    ps.save(fig, "model_tuning")


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
