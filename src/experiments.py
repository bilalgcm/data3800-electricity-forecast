"""Step 7: experiments for the final report.

1. Ablation test: how much do temperature, calendar effects and lagged load
   each contribute? Retrain the random forest on different feature groups
   and compare the error on the test period.
2. Weekend fix: does replacing the 48 h lag with a day-type-aware lag reduce
   the errors on Mondays, Tuesdays and weekends?
3. Temperature forecast error: our models are tested with MEASURED temperature,
   which a real day-ahead forecast does not have. How much worse does the
   random forest get when the temperature it is given is wrong by 1, 2 or 3
   degrees?

Input:  data/processed/dataset.csv (run build_dataset.py first)
Output: figures/ablation.csv, figures/ablation.png,
        figures/weekend_fix.csv, figures/weekend_fix.png,
        figures/temperature_error.csv, figures/temperature_error.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import config
import plot_style as ps
from build_dataset import add_daytype_features
from models import TARGET, make_forest, saved_forest_setting, split

TEMPERATURE = ["temp_c"]
CALENDAR = ["hour", "weekday", "month", "is_weekend", "is_holiday"]
LAGS = ["load_lag_48h", "load_lag_168h"]
ALL = TEMPERATURE + CALENDAR + LAGS

# Name -> feature list. Order = order in the table and figure.
ABLATION = {
    "All features": ALL,
    "Without temperature": CALENDAR + LAGS,
    "Without calendar": TEMPERATURE + LAGS,
    "Without lagged load": TEMPERATURE + CALENDAR,
    "Lagged load only": LAGS,
    "Temperature only": TEMPERATURE,
    "Calendar only": CALENDAR,
}

# Weekend fix: swap the 48 h lag for the day-type-aware lag.
FIXED = TEMPERATURE + CALENDAR + ["load_lag_daytype", "load_lag_168h", "lag48_same_daytype"]


def fit_predict(train, test, features):
    """Same random forest settings as models.py, so results are comparable."""
    rf = make_forest(saved_forest_setting())
    rf.fit(train[features], train[TARGET])
    return rf.predict(test[features])


def run_ablation(train, test):
    rows = []
    for name, features in ABLATION.items():
        pred = fit_predict(train, test, features)
        mae = mean_absolute_error(test[TARGET], pred)
        rows.append({"experiment": name, "features": ", ".join(features), "MAE": mae})
        print(f"  {name:<22} MAE {mae:6.1f}")
    table = pd.DataFrame(rows)
    full = table.loc[table.experiment == "All features", "MAE"].iloc[0]
    table["MAE_change_vs_all"] = table["MAE"] - full
    table = table.round(1)
    table.to_csv(config.FIGURES / "ablation.csv", index=False)
    return table


def plot_ablation(table, baseline_mae, entsoe_mae):
    t = table.iloc[::-1]  # best at the top after barh
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = [ps.BLUE if e == "All features" else "#b9b8b2" for e in t.experiment]
    ax.barh(t.experiment, t.MAE, color=colors, height=0.65)
    for y, v in enumerate(t.MAE):
        ax.text(v + 8, y, f"{v:.0f}", va="center", color=ps.TEXT, fontsize=9)
    for x, label in [(baseline_mae, "Baseline"), (entsoe_mae, "ENTSO-E")]:
        if x is not None:
            ax.axvline(x, color=ps.TEXT, linestyle=":", linewidth=1)
            ax.text(x, -0.75, f" {label} {x:.0f}", color=ps.TEXT, fontsize=8, va="center")
    ax.set_title("Ablation test: random forest error by feature set")
    ax.set_xlabel("MAE on test period (MW), lower is better")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-1.2, len(t) - 0.5)
    ax.set_xlim(0, max(t.MAE.max(), baseline_mae or 0) * 1.12)
    ps.save(fig, "ablation")


def run_weekend_fix(train, test):
    before = fit_predict(train, test, ALL)
    after = fit_predict(train, test, FIXED)

    local = test.index.tz_convert(config.TZ)
    groups = pd.Series("Tue to Fri", index=test.index)
    groups[local.dayofweek == 0] = "Monday"
    groups[local.dayofweek >= 5] = "Weekend"
    groups[test["is_holiday"] == 1] = "Holiday"

    masks = {"All hours": pd.Series(True, index=test.index)}
    for g in ["Monday", "Tue to Fri", "Weekend", "Holiday"]:
        masks[g] = groups == g
    masks["48 h lag from other day type"] = test["lag48_same_daytype"] == 0

    rows = []
    for name, mask in masks.items():
        m = mask.to_numpy()
        if m.sum() == 0:
            continue
        y = test[TARGET].to_numpy()[m]
        rows.append({
            "group": name,
            "hours": int(m.sum()),
            "MAE_before": mean_absolute_error(y, before[m]),
            "MAE_after": mean_absolute_error(y, after[m]),
        })
    table = pd.DataFrame(rows)
    table["change_pct"] = (table.MAE_after / table.MAE_before - 1) * 100
    table = table.round(1)
    table.to_csv(config.FIGURES / "weekend_fix.csv", index=False)
    return table


def plot_weekend_fix(table):
    t = table[table.group != "48 h lag from other day type"]
    x = range(len(t))
    w = 0.38
    fig, ax = plt.subplots()
    ax.bar([i - w / 2 for i in x], t.MAE_before, width=w, color="#b9b8b2", label="48 h lag (current)")
    ax.bar([i + w / 2 for i in x], t.MAE_after, width=w, color=ps.BLUE, label="Day-type-aware lag (fix)")
    for i, (b, a) in enumerate(zip(t.MAE_before, t.MAE_after)):
        ax.text(i - w / 2, b + 3, f"{b:.0f}", ha="center", color=ps.TEXT, fontsize=8)
        ax.text(i + w / 2, a + 3, f"{a:.0f}", ha="center", color=ps.TEXT, fontsize=8)
    ax.set_xticks(list(x), [f"{g}\n({h} h)" for g, h in zip(t.group, t.hours)])
    ax.set_title("Weekend fix: error by type of day")
    ax.set_ylabel("MAE on test period (MW)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(t.MAE_before.max(), t.MAE_after.max()) * 1.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=2)
    ps.save(fig, "weekend_fix")


TEMP_ERROR_SD = [0, 1, 2, 3]   # degrees C
TEMP_ERROR_REPEATS = 10


def run_temperature_error(train, test):
    """Give the trained forest a wrong temperature and measure the damage.

    We have no archive of old weather forecasts, so we simulate one. A weather
    forecast is usually wrong in the same direction for a whole day, so each
    day gets one random offset (normal, standard deviation = sd) and each hour
    a smaller one on top (sd / 2). The model itself is not retrained.
    """
    rf = make_forest(saved_forest_setting()).fit(train[ALL], train[TARGET])
    days = pd.Series(test.index.tz_convert(config.TZ).date, index=test.index)
    unique_days = days.unique()

    rows = []
    for sd in TEMP_ERROR_SD:
        maes = []
        for seed in range(1 if sd == 0 else TEMP_ERROR_REPEATS):
            rng = np.random.default_rng(seed)
            day_offset = pd.Series(rng.normal(0, sd, len(unique_days)), index=unique_days)
            noisy = test[ALL].copy()
            noisy["temp_c"] = (test["temp_c"] + days.map(day_offset).to_numpy()
                               + rng.normal(0, sd / 2, len(test)))
            maes.append(mean_absolute_error(test[TARGET], rf.predict(noisy)))
        rows.append({"temperature_error_sd_c": sd, "MAE": np.mean(maes),
                     "MAE_min": np.min(maes), "MAE_max": np.max(maes)})
    table = pd.DataFrame(rows)
    table["MAE_change_pct"] = (table.MAE / table.MAE.iloc[0] - 1) * 100
    table = table.round(1)
    table.to_csv(config.FIGURES / "temperature_error.csv", index=False)
    return table


def plot_temperature_error(table, entsoe_mae):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    x = range(len(table))
    ax.bar(x, table.MAE, color=ps.BLUE, width=0.6)
    for i, v in enumerate(table.MAE):
        ax.text(i, v + 3, f"{v:.0f}", ha="center", color=ps.TEXT, fontsize=9)
    if entsoe_mae is not None:
        ax.axhline(entsoe_mae, color=ps.ORANGE, linestyle="--", linewidth=1.5)
        ax.text(-0.45, entsoe_mae + 4, f"ENTSO-E day-ahead forecast: {entsoe_mae:.0f}",
                color=ps.ORANGE, ha="left", va="bottom", fontsize=9)
    labels = ["Measured\ntemperature"] + [f"Error ±{sd} °C" for sd in table.temperature_error_sd_c[1:]]
    ax.set_xticks(list(x), labels)
    ax.set_title("Random forest error when the temperature is a forecast")
    ax.set_ylabel("MAE on test period (MW)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(table.MAE.max(), entsoe_mae or 0) * 1.15)
    ps.save(fig, "temperature_error")


def main():
    df = pd.read_csv(config.PROCESSED / "dataset.csv", index_col="time_utc", parse_dates=True)
    if "load_lag_daytype" not in df:
        print("Day-type features missing, adding them (rerun build_dataset.py to save them)")
        df = add_daytype_features(df)
    train, test = split(df)

    baseline_mae = mean_absolute_error(test[TARGET], test["load_lag_168h"])
    entsoe_mae = mean_absolute_error(test[TARGET], test["forecast_mw"]) if "forecast_mw" in test else None

    print("Ablation test (random forest, test period):")
    ablation = run_ablation(train, test)
    plot_ablation(ablation, baseline_mae, entsoe_mae)
    print(ablation[["experiment", "MAE", "MAE_change_vs_all"]].to_string(index=False))

    print("\nWeekend fix:")
    fix = run_weekend_fix(train, test)
    plot_weekend_fix(fix)
    print(fix.to_string(index=False))

    print("\nTemperature forecast error:")
    temp = run_temperature_error(train, test)
    plot_temperature_error(temp, entsoe_mae)
    print(temp.to_string(index=False))


if __name__ == "__main__":
    main()