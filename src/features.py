"""Calendar features, built in ONE place.

Training (build_dataset.py), the agent's forecast tool (agent/tools.py) and the
prediction service (service/app.py) all call this function. That way live data
is prepared exactly as the training data was (week 45: consistency).
"""
import holidays
import pandas as pd

import config


def calendar_features(index):
    """Hour, weekday, month, weekend and holiday for a tz-aware hourly index.

    The index can be in any time zone. Features are always in local Oslo time,
    because that is the clock people live by.
    """
    local = index.tz_convert(config.TZ)
    no_holidays = holidays.Norway(years=range(local.year.min(), local.year.max() + 1))
    out = pd.DataFrame(index=index)
    out["hour"] = local.hour
    out["weekday"] = local.dayofweek          # 0 = Monday
    out["month"] = local.month
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    out["is_holiday"] = [int(d in no_holidays) for d in local.date]
    return out
