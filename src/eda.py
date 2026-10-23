"""Step 4: exploratory data analysis.

Input:  data/processed/dataset.csv
Output: figures/eda_*.png and figures/eda_summary.csv

Hour, weekday and month in the dataset are already in local time (Europe/Oslo),
so the patterns below are what people in Oslo actually experience.
"""
import matplotlib.pyplot as plt
import pandas as pd

import config
import plot_style as ps

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_dataset():
    df = pd.read_csv(config.PROCESSED / "dataset.csv", index_col="time_utc", parse_dates=True)
    df.index = df.index.tz_convert(config.TZ)
    return df


def plot_over_time(df):
    """Daily average load for the whole period, with the test period marked."""
    daily = df["load_mw"].resample("D").mean()
    weekly = daily.rolling(7, center=True).mean()

    fig, ax = plt.subplots()
    ax.plot(daily.index, daily, color=ps.BLUE, alpha=0.35, linewidth=1, label="Daily average")
    ax.plot(weekly.index, weekly, color=ps.BLUE, label="7-day rolling average")
    test_start = pd.Timestamp(config.TEST_START, tz=config.TZ)
    ax.axvspan(test_start, daily.index.max(), color=ps.GRID, alpha=0.6, zorder=0)
    ax.text(test_start, ax.get_ylim()[1], "  Test period", va="top", color=ps.TEXT)
    ax.set_title("Electricity consumption in NO1, 2023 to 2026")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="lower left")
    ps.save(fig, "eda_load_over_time")


def plot_by_hour(df):
    """Average daily profile, weekdays vs weekends."""
    weekday = df[df["is_weekend"] == 0].groupby("hour")["load_mw"].mean()
    weekend = df[df["is_weekend"] == 1].groupby("hour")["load_mw"].mean()

    fig, ax = plt.subplots()
    ax.plot(weekday.index, weekday, color=ps.BLUE, marker="o", markersize=4, label="Weekday")
    ax.plot(weekend.index, weekend, color=ps.ORANGE, marker="o", markersize=4, label="Weekend")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Average load by hour of day")
    ax.set_xlabel("Hour (local time)")
    ax.set_ylabel("Load (MW)")
    ax.legend()
    ps.save(fig, "eda_load_by_hour")


def plot_bars(series, labels, title, xlabel, name):
    fig, ax = plt.subplots()
    ax.bar(labels, series.values, color=ps.BLUE, width=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average load (MW)")
    ax.grid(axis="x", visible=False)
    # Start the axis a bit below the smallest bar so differences are visible,
    # and say so in the figure, since a cut axis exaggerates differences.
    low = series.min() * 0.9
    ax.set_ylim(low, series.max() * 1.03)
    ax.text(0.99, 0.98, f"y-axis starts at {low:,.0f} MW", transform=ax.transAxes,
            ha="right", va="top", color=ps.TEXT, fontsize=8)
    ps.save(fig, name)


def plot_vs_temperature(df):
    """Daily average load against daily average temperature."""
    daily = df[["load_mw", "temp_c"]].resample("D").mean().dropna()
    corr = daily["load_mw"].corr(daily["temp_c"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(daily["temp_c"], daily["load_mw"], s=10, color=ps.BLUE, alpha=0.4, linewidths=0)
    ax.set_title("Colder days mean higher consumption")
    ax.set_xlabel("Daily average temperature at Blindern (°C)")
    ax.set_ylabel("Daily average load (MW)")
    ax.text(0.98, 0.95, f"Pearson r = {corr:.2f}\n(daily averages, n = {len(daily)})",
            transform=ax.transAxes, ha="right", va="top", color=ps.TEXT)
    ps.save(fig, "eda_load_vs_temperature")
    return corr


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} hourly rows from {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")

    plot_over_time(df)
    plot_by_hour(df)

    by_weekday = df.groupby("weekday")["load_mw"].mean()
    plot_bars(by_weekday, WEEKDAYS, "Average load by day of week", "Day of week", "eda_load_by_weekday")

    by_month = df.groupby("month")["load_mw"].mean()
    plot_bars(by_month, [MONTHS[m - 1] for m in by_month.index],
              "Average load by month", "Month", "eda_load_by_month")

    corr = plot_vs_temperature(df)

    # Numbers we will quote in the report.
    summary = pd.Series({
        "rows": len(df),
        "mean_load_mw": df["load_mw"].mean(),
        "min_load_mw": df["load_mw"].min(),
        "max_load_mw": df["load_mw"].max(),
        "winter_to_summer_ratio": by_month.max() / by_month.min(),
        "weekday_vs_weekend_pct": (df.loc[df.is_weekend == 0, "load_mw"].mean()
                                   / df.loc[df.is_weekend == 1, "load_mw"].mean() - 1) * 100,
        "corr_daily_load_temp": corr,
    }).round(2)
    print(summary.to_string())
    summary.to_csv(config.FIGURES / "eda_summary.csv", header=["value"])


if __name__ == "__main__":
    main()
