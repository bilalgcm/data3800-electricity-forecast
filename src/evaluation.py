"""Step 6: look at where the errors fall, not only how large they are on average.

1. Residuals: predicted against actual, and the distribution of the errors.
2. Error by month and by hour, random forest against ENTSO-E.
3. High-load hours as a classification problem: confusion matrix, accuracy,
   precision, recall and F1. For the grid the expensive mistake is a high-load
   hour that nobody warned about, so recall is the metric that matters most.

Input:  data/processed/predictions_test.csv (from models.py), data/processed/dataset.csv
Output: figures/eval_*.png and figures/eval_*.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import config
import plot_style as ps
from models import TARGET, load_dataset, split

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MODELS = {  # column in predictions_test.csv -> name in tables
    "baseline_mw": "Baseline (same hour last week)",
    "entsoe_mw": "ENTSO-E day-ahead forecast",
    "linear_regression_mw": "Linear regression",
    "decision_tree_mw": "Decision tree",
    "knn_mw": "k nearest neighbours",
    "random_forest_mw": "Random forest",
}
HIGH_LOAD_QUANTILE = 0.90   # a "high-load hour" is in the top 10 % of the training period


def load_predictions():
    preds = pd.read_csv(config.PROCESSED / "predictions_test.csv",
                        index_col="time_utc", parse_dates=True)
    return preds.tz_convert(config.TZ)


# ---------- 1. residuals ----------

def plot_residuals(preds):
    """Predicted against actual, and the histogram of the errors, for the random forest."""
    actual, pred = preds["actual_mw"], preds["random_forest_mw"]
    residual = pred - actual          # positive = we forecast too high

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    left.scatter(actual, pred, s=6, color=ps.BLUE, alpha=0.25, linewidths=0)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    left.plot(lims, lims, color=ps.ACTUAL, linewidth=1, linestyle="--", label="Perfect forecast")
    left.set_title("Predicted against actual (random forest)")
    left.set_xlabel("Actual load (MW)")
    left.set_ylabel("Predicted load (MW)")
    left.legend(loc="upper left")

    right.hist(residual, bins=60, color=ps.BLUE)
    right.axvline(0, color=ps.ACTUAL, linewidth=1, linestyle="--")
    right.axvline(residual.mean(), color=ps.ORANGE, linewidth=1.5,
                  label=f"Mean error {residual.mean():+.0f} MW")
    right.set_title("Distribution of the errors")
    right.set_xlabel("Predicted minus actual (MW)")
    right.set_ylabel("Hours")
    right.legend(loc="upper right")
    ps.save(fig, "eval_residuals")

    summary = pd.Series({
        "mean_error_mw": residual.mean(),
        "median_abs_error_mw": residual.abs().median(),
        "p95_abs_error_mw": residual.abs().quantile(0.95),
        "max_abs_error_mw": residual.abs().max(),
        "share_within_100_mw_pct": (residual.abs() <= 100).mean() * 100,
        "share_over_300_mw_pct": (residual.abs() > 300).mean() * 100,
    }).round(1)
    summary.to_csv(config.FIGURES / "eval_residual_summary.csv", header=["value"])
    return summary


# ---------- 2. where the errors fall ----------

def error_by_month(preds):
    """MAE per calendar month in the test period, for every model."""
    abs_err = pd.DataFrame({name: (preds[col] - preds["actual_mw"]).abs()
                            for col, name in MODELS.items() if col in preds})
    table = abs_err.groupby(preds.index.month).mean()
    table.insert(0, "mean_load_mw", preds["actual_mw"].groupby(preds.index.month).mean())
    table.insert(0, "hours", preds["actual_mw"].groupby(preds.index.month).size())
    table.index = [MONTHS[m - 1] for m in table.index]
    table.index.name = "month"
    table.round(1).to_csv(config.FIGURES / "eval_error_by_month.csv")
    return table


def plot_error_by_month_and_hour(preds, by_month):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    x = np.arange(len(by_month))
    w = 0.38
    left.bar(x - w / 2, by_month["Random forest"], width=w, color=ps.BLUE, label="Random forest (ours)")
    left.bar(x + w / 2, by_month["ENTSO-E day-ahead forecast"], width=w, color=ps.ORANGE,
             label="ENTSO-E day-ahead forecast")
    left.set_xticks(x, by_month.index)
    left.set_title("Error by month, test period 2026")
    left.set_ylabel("MAE (MW)")
    left.grid(axis="x", visible=False)
    left.legend(loc="upper right")

    by_hour = pd.DataFrame({
        "rf": (preds["random_forest_mw"] - preds["actual_mw"]).abs(),
        "entsoe": (preds["entsoe_mw"] - preds["actual_mw"]).abs(),
    }).groupby(preds.index.hour).mean()
    right.plot(by_hour.index, by_hour["rf"], color=ps.BLUE, marker="o", markersize=3)
    right.plot(by_hour.index, by_hour["entsoe"], color=ps.ORANGE, marker="o", markersize=3)
    right.set_xticks(range(0, 24, 3))
    right.set_title("Error by hour of day")
    right.set_xlabel("Hour (local time)")
    ps.save(fig, "eval_error_by_month_and_hour")


# ---------- 3. high-load hours as classification ----------

def classify_high_load(preds, threshold):
    """Turn each forecast into a yes/no: does it call this a high-load hour?"""
    actual_high = preds["actual_mw"] > threshold
    rows = [{  # the lazy model from the lecture: never raise the alarm
        "model": "Never warns (lazy model)",
        "accuracy": accuracy_score(actual_high, np.zeros(len(preds), dtype=bool)),
        "precision": np.nan, "recall": 0.0, "F1": 0.0,
        "TP": 0, "FP": 0, "FN": int(actual_high.sum()), "TN": int((~actual_high).sum()),
    }]
    matrices = {}
    for col, name in MODELS.items():
        if col not in preds:
            continue
        predicted_high = preds[col] > threshold
        tn, fp, fn, tp = confusion_matrix(actual_high, predicted_high).ravel()
        matrices[name] = np.array([[tp, fn], [fp, tn]])
        rows.append({
            "model": name,
            "accuracy": accuracy_score(actual_high, predicted_high),
            "precision": precision_score(actual_high, predicted_high, zero_division=0),
            "recall": recall_score(actual_high, predicted_high),
            "F1": f1_score(actual_high, predicted_high),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    table = pd.DataFrame(rows)
    table.round(3).to_csv(config.FIGURES / "eval_classification.csv", index=False)
    return table, matrices


def plot_confusion_matrices(matrices, threshold, share):
    names = ["Random forest", "ENTSO-E day-ahead forecast"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, name in zip(axes, names):
        m = matrices[name]
        ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max() * 1.4)
        for (i, j), v in np.ndenumerate(m):
            label = [["TP", "FN"], ["FP", "TN"]][i][j]
            ax.text(j, i, f"{label}\n{v:,}", ha="center", va="center", fontsize=11,
                    color="white" if v > m.max() * 0.7 else ps.ACTUAL)
        ax.set_xticks([0, 1], ["Forecast: high", "Forecast: not high"])
        ax.set_yticks([0, 1], ["Actual: high", "Actual: not high"])
        ax.set_title(name)
        ax.grid(False)
    fig.suptitle(f"High-load hours (above {threshold:,.0f} MW, {share:.0%} of the test hours)",
                 x=0.02, ha="left", fontsize=11, color=ps.TEXT)
    fig.tight_layout()
    ps.save(fig, "eval_confusion_matrix")


def alert_margin(preds, threshold):
    """Precision against recall: warn already when the forecast is close to the limit.

    A lower alarm limit catches more of the real high-load hours (recall up)
    but raises more false alarms (precision down).
    """
    actual_high = preds["actual_mw"] > threshold
    rows = []
    for margin in [0, 50, 100, 150, 200, 300]:
        warned = preds["random_forest_mw"] > threshold - margin
        rows.append({
            "warn_when_forecast_above_mw": round(threshold - margin),
            "margin_mw": margin,
            "precision": precision_score(actual_high, warned),
            "recall": recall_score(actual_high, warned),
            "F1": f1_score(actual_high, warned),
            "missed_high_load_hours": int((actual_high & ~warned).sum()),
            "false_alarms": int((~actual_high & warned).sum()),
        })
    table = pd.DataFrame(rows)
    table.round(3).to_csv(config.FIGURES / "eval_alert_margin.csv", index=False)
    return table


def main():
    preds = load_predictions()
    train, _ = split(load_dataset())
    threshold = train[TARGET].quantile(HIGH_LOAD_QUANTILE)

    print("Residuals, random forest (MW):")
    print(plot_residuals(preds).to_string())

    by_month = error_by_month(preds)
    plot_error_by_month_and_hour(preds, by_month)
    print("\nMAE by month in the test period (MW):")
    print(by_month.round(0).to_string())

    share = (preds["actual_mw"] > threshold).mean()
    print(f"\nHigh-load hours: above {threshold:.0f} MW (top 10 % of 2023 to 2025). "
          f"{share:.1%} of the test hours are high-load.")
    table, matrices = classify_high_load(preds, threshold)
    plot_confusion_matrices(matrices, threshold, share)
    print(table.round(3).to_string(index=False))

    print("\nWarning earlier: precision against recall for the random forest:")
    print(alert_margin(preds, threshold).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
