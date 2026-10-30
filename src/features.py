"""Feature engineering shared by training, evaluation and the Streamlit app.

Keeping this in one module guarantees the app computes features exactly the
way the model saw them during training (no training/serving skew).
"""
import numpy as np
import pandas as pd

from src.config import THRESHOLD_UNITS, BILLING_BASE_DAYS

# Inputs a household can provide (from its meter, its past bills and a short
# questionnaire). The model only ever sees these + derived features below.
BASE_INPUTS = {
    "day": "Days into the billing month (1-15)",
    "days_in_month": "Length of the billing month in days",
    "so_far": "Units used so far this month",
    "last7": "Units used in the last 7 days",
    "last3": "Units used in the last 3 days",
    "weekend_days_left": "Weekend days left this month",
    "holidays_left": "Public / Poya holidays left this month",
    "prev1": "Units on last month's bill",
    "prev2": "Units on the bill 2 months ago",
    "prev3": "Units on the bill 3 months ago",
    "month": "Calendar month (1-12)",
    "members": "People living in the house",
    "n_ac": "Air conditioners",
    "n_fridge": "Refrigerators / freezers",
    "electric_water_heater": "Electric water heater for bathing (0/1)",
    "washing_machine": "Washing machine (0/1)",
    "electric_cooking": "Cooks with electricity (0/1)",
    "home_business": "Runs a business from home (0/1)",
}

DERIVED = [
    "limit",            # pro-rated 60-unit limit for this month length
    "pace_projection",  # naive projection: so_far / day * days_in_month
    "pace_ratio",       # pace projection divided by the limit
    "units_left",       # units remaining before the limit is crossed
    "allowed_per_day",  # average daily units allowed for the rest of the month
    "recent_rate",      # recent daily usage (last 7 days)
    "allowed_vs_recent",  # allowed per day / recent per day  (<1 means trouble)
    "history_avg",      # mean of the last three bills
    "history_ratio",    # history_avg divided by the limit
    "history_max",      # highest of the last three bills
    "history_trend",    # last bill minus the one 3 months ago
    "remaining_by_history",  # so_far + rest of month at the historical daily rate
    "recent3_rate",     # daily usage over the last 3 days
    "trend_ratio",      # last-3-day rate vs month-to-date rate (>1 = speeding up)
]

FEATURES = list(BASE_INPUTS) + DERIVED


def month_limit(days_in_month):
    """60 units per 30-day period, pro-rated as in the PUCSL tariff."""
    return THRESHOLD_UNITS * np.asarray(days_in_month, dtype=float) / BILLING_BASE_DAYS


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered domain features added."""
    d = df.copy()
    day = d["day"].astype(float)
    dim = d["days_in_month"].astype(float)
    so_far = d["so_far"].astype(float)
    rest = (dim - day).clip(lower=1)

    d["limit"] = month_limit(dim)
    d["pace_projection"] = so_far / day * dim
    d["pace_ratio"] = d["pace_projection"] / d["limit"]
    d["units_left"] = d["limit"] - so_far
    d["allowed_per_day"] = d["units_left"] / rest
    d["recent_rate"] = d["last7"] / 7.0
    d["allowed_vs_recent"] = d["allowed_per_day"] / d["recent_rate"].clip(lower=0.1)
    hist = d[["prev1", "prev2", "prev3"]]
    d["history_avg"] = hist.mean(axis=1)
    d["history_ratio"] = d["history_avg"] / d["limit"]
    d["history_max"] = hist.max(axis=1)
    d["history_trend"] = d["prev1"] - d["prev3"]
    d["remaining_by_history"] = so_far + rest * d["history_avg"] / BILLING_BASE_DAYS
    d["recent3_rate"] = d["last3"] / 3.0
    d["trend_ratio"] = d["recent3_rate"] / (so_far / day).clip(lower=0.1)
    return d


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    """Model matrix in a fixed column order."""
    return add_derived(df)[FEATURES].astype(float)


# Plain-English names used in the report figures and in the app.
FRIENDLY = {
    "day": "Days into the month",
    "days_in_month": "Days in the month",
    "so_far": "Units used so far",
    "last7": "Units in the last 7 days",
    "last3": "Units in the last 3 days",
    "weekend_days_left": "Weekend days left",
    "holidays_left": "Holidays left (incl. Poya)",
    "prev1": "Last month's bill (units)",
    "prev2": "Bill 2 months ago (units)",
    "prev3": "Bill 3 months ago (units)",
    "month": "Month of the year",
    "members": "People in the house",
    "n_ac": "Air conditioners",
    "n_fridge": "Fridges / freezers",
    "electric_water_heater": "Electric water heater",
    "washing_machine": "Washing machine",
    "electric_cooking": "Cooks with electricity",
    "home_business": "Business run from home",
    "limit": "60-unit limit for this month",
    "pace_projection": "Projected month total at current pace",
    "pace_ratio": "Current pace vs the limit",
    "units_left": "Units left before the limit",
    "allowed_per_day": "Units allowed per day from now",
    "recent_rate": "Recent daily usage (7 days)",
    "allowed_vs_recent": "Allowed per day vs recent daily usage",
    "history_avg": "Average of last 3 bills",
    "history_ratio": "Usual bill vs the limit",
    "history_max": "Highest of last 3 bills",
    "history_trend": "Change in bills over 3 months",
    "remaining_by_history": "Projected total using usual habits",
    "recent3_rate": "Recent daily usage (3 days)",
    "trend_ratio": "Speeding up or slowing down",
}


def friendly(term: str) -> str:
    """'a & b' interaction terms -> 'A x B'."""
    return " x ".join(FRIENDLY.get(t.strip(), t.strip()) for t in term.split("&"))


def calendar_left(today, period_end):
    """Weekend days and Sri Lankan public/Poya holidays after `today` up to `period_end`."""
    import holidays
    hol = holidays.country_holidays("LK", years=range(today.year, period_end.year + 1))
    rest = pd.date_range(pd.Timestamp(today) + pd.Timedelta(days=1), pd.Timestamp(period_end), freq="D")
    return int((rest.dayofweek >= 5).sum()), int(sum(d.date() in hol for d in rest))
