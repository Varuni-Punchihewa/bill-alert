"""Step 1 - clean the smart-meter register, join the survey, build the modelling table.

Output: data/processed/checkpoints.parquet  (one row per household x month x checkpoint day)
        data/processed/monthly.parquet      (one row per household x month)
        data/processed/households.parquet   (survey features per household)
        reports/data_quality.json           (what was removed and why)

    python -m src.data_prep
"""
import json

import numpy as np
import pandas as pd

from src.config import (DAILY_REGISTER, SURVEY, DATA_PROC, REPORTS, MAX_DAILY_KWH, MIN_PREV_UNITS,
                        CHECKPOINT_DAYS, THRESHOLD_UNITS, BILLING_BASE_DAYS)

REG = ["imp", "exp", "tr1", "tr2", "tr3"]


# ------------------------------------------------------------------ smart meter
def load_daily_usage(path=DAILY_REGISTER):
    """Turn cumulative daily register readings into daily usage (kWh/day)."""
    reg = pd.read_csv(path, parse_dates=["date"]).sort_values(["hid", "date"])
    q = {"raw_household_days": int(len(reg)), "raw_households": int(reg.hid.nunique())}

    frames = []
    for hid, g in reg.groupby("hid", sort=False):
        # Re-index to a full calendar so a missing day is an explicit gap.
        # No interpolation: filling a gap would use the *next* reading, which
        # is future information at a mid-month checkpoint.
        g = g.set_index("date").reindex(pd.date_range(g.date.min(), g.date.max(), freq="D"))
        g["hid"] = hid
        frames.append(g)
    x = pd.concat(frames).rename_axis("date").reset_index()
    for c in REG:
        x["d_" + c] = x.groupby("hid")[c].diff()

    # A negative step = meter reset / replacement; a huge step = glitch.
    bad = (x.d_imp < 0) | (x.d_imp > MAX_DAILY_KWH)
    q["glitch_days_removed"] = int(bad.sum())
    for c in REG:
        x.loc[bad, "d_" + c] = np.nan

    # Rooftop-solar homes are net-metered; their import register is not their
    # billed consumption, so the 60-unit rule does not apply the same way.
    solar = x.groupby("hid").d_exp.max().fillna(0) > 0
    q["solar_households_excluded"] = int(solar.sum())
    x = x[~x.hid.isin(solar[solar].index)].copy()

    x["ym"] = x.date.dt.to_period("M")
    x["dom"] = x.date.dt.day
    x["dim"] = x.date.dt.days_in_month
    x["glitch"] = bad.loc[x.index].astype(int)
    q["households_after_cleaning"] = int(x.hid.nunique())
    q["missing_reading_share"] = round(float(x.imp.isna().mean()), 4)
    return x[["hid", "date", "ym", "dom", "dim", "imp", "tr2", "tr3", "d_imp", "glitch"]], q


def monthly_totals(x):
    """Month usage = closing register - previous month's closing register.

    The register is cumulative, so a missing day *inside* the month does not
    matter; only the two boundary readings are needed. A month is invalid if
    either boundary reading is missing or the meter glitched/reset inside it.
    """
    x = x.sort_values(["hid", "date"])
    last = x[x.dom == x.dim]                       # closing reading of each month
    close = last.set_index(["hid", "ym"])[["imp", "tr2", "tr3"]]
    opening = close.copy()
    opening.index = pd.MultiIndex.from_arrays([close.index.get_level_values(0),
                                               close.index.get_level_values(1) + 1])
    m = x.groupby(["hid", "ym"]).agg(dim=("dim", "first"), glitches=("glitch", "sum"),
                                     n_readings=("imp", "count")).reset_index()
    idx = pd.MultiIndex.from_frame(m[["hid", "ym"]])
    o = opening.reindex(idx).to_numpy()
    c = close.reindex(idx).to_numpy()
    m["open_reg"] = o[:, 0]
    m["total"] = c[:, 0] - o[:, 0]
    m["peak"] = c[:, 1] - o[:, 1]
    m["offpeak"] = c[:, 2] - o[:, 2]
    m["complete"] = (m.total.notna() & (m.glitches == 0) & (m.total >= 0)
                     & (m.total <= MAX_DAILY_KWH * m.dim))
    m["limit"] = THRESHOLD_UNITS * m.dim / BILLING_BASE_DAYS
    m["over_limit"] = (m.total > m.limit).astype(int)
    m = m.sort_values(["hid", "ym"])
    # bill history = previous *complete* months only
    # Put every household on a gap-free monthly calendar so that "previous
    # month" really is the previous calendar month.
    m["mi"] = m.ym.dt.year * 12 + m.ym.dt.month
    tot = m.set_index(["hid", "mi"]).total.where(m.set_index(["hid", "mi"]).complete)
    for k in (1, 2, 3):
        shifted = tot.copy()
        shifted.index = pd.MultiIndex.from_arrays([tot.index.get_level_values(0),
                                                   tot.index.get_level_values(1) + k])
        m[f"prev{k}"] = shifted.reindex(pd.MultiIndex.from_frame(m[["hid", "mi"]])).to_numpy()
    return m.drop(columns="mi")


def build_checkpoints(x, m):
    """One row per (household, complete month, checkpoint day)."""
    x = x.sort_values(["hid", "date"]).copy()
    g = x.groupby("hid")
    # readings 3 and 7 days earlier (calendar is gap-free per household after reindex)
    x["imp_3ago"] = g.imp.shift(3)
    x["imp_7ago"] = g.imp.shift(7)
    x["glitch3"] = g.glitch.transform(lambda s: s.rolling(3, min_periods=1).sum())
    x["glitch7"] = g.glitch.transform(lambda s: s.rolling(7, min_periods=1).sum())

    # Household rhythm: share of LAST month's total that had been used by the
    # same day of last month (known before this month starts -> no leakage).
    full = x.merge(m.loc[m.complete, ["hid", "ym", "open_reg", "total"]], on=["hid", "ym"], how="inner")
    full["frac"] = ((full.imp - full.open_reg) / full.total.where(full.total > 0)).clip(0, 1)
    lag = full[["hid", "ym", "dom", "frac"]].copy()
    lag["ym"] = lag.ym + 1
    lag = lag.rename(columns={"frac": "prev_month_share_by_day"})

    cp = x[x.dom.isin(list(CHECKPOINT_DAYS)) & x.imp.notna()].copy()
    keep = m[m.complete & m.prev1.notna()]
    cp = cp.merge(keep[["hid", "ym", "open_reg", "total", "limit", "over_limit",
                        "prev1", "prev2", "prev3"]], on=["hid", "ym"], how="inner")
    cp = cp.merge(lag, on=["hid", "ym", "dom"], how="left")
    # usage so far uses only the reading available on the checkpoint day
    cp["so_far"] = cp.imp - cp.open_reg
    cp["last7"] = (cp.imp - cp.imp_7ago).where(cp.glitch7 == 0)
    cp["last3"] = (cp.imp - cp.imp_3ago).where(cp.glitch3 == 0)
    cp = cp[(cp.so_far >= 0) & (cp.so_far <= cp.total)]
    cp = cp[["hid", "ym", "date", "dom", "so_far", "last7", "last3", "prev_month_share_by_day", "dim",
             "total", "limit", "over_limit", "prev1", "prev2", "prev3"]].rename(columns={"dom": "day"})
    cp = cp.rename(columns={"dim": "days_in_month"})
    cal = calendar_features(cp.date)
    cp = pd.concat([cp.reset_index(drop=True), cal.reset_index(drop=True)], axis=1).drop(columns="date")
    cp["month"] = cp.ym.dt.month
    cp["period"] = cp.ym.dt.to_timestamp()
    return cp


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Weekend days and Sri Lankan public/Poya holidays left in the month."""
    import holidays
    years = range(dates.dt.year.min() - 1, dates.dt.year.max() + 2)
    hol = {pd.Timestamp(d) for d in holidays.country_holidays("LK", years=years)}
    uniq = pd.Series(dates.unique())
    rows = []
    for d in uniq:
        rest = pd.date_range(d + pd.Timedelta(days=1), d + pd.offsets.MonthEnd(0), freq="D")
        rows.append({"date": d, "weekend_days_left": int((rest.dayofweek >= 5).sum()),
                     "holidays_left": int(sum(r in hol for r in rest))})
    table = pd.DataFrame(rows).set_index("date")
    return table.reindex(dates.to_numpy()).reset_index(drop=True)


# ----------------------------------------------------------------------- survey
def household_features():
    w1 = SURVEY / "wave_1"
    h = pd.read_csv(w1 / "w1_household_information_and_history.csv")
    a = pd.read_csv(w1 / "w1_appliances.csv")
    r = pd.read_csv(w1 / "w1_room_roster.csv", low_memory=False)
    e = pd.read_csv(w1 / "w1_electricity_generation_water_heating_cooking.csv")

    hh = h.set_index("household_ID")
    f = pd.DataFrame(index=hh.index)
    f["members"] = hh.no_of_household_members
    f["n_ac"] = r.groupby("household_ID").no_of_ACs_in_the_room.apply(
        lambda s: pd.to_numeric(s, errors="coerce").sum())

    def count(pattern):
        hit = a.appliance_type.str.contains(pattern, case=False, regex=True, na=False)
        return a[hit].groupby("household_ID").size()

    f["n_fridge"] = count(r"Refrigerator|Separate Freezer")
    f["washing_machine"] = (count(r"^Washing Machine") > 0).astype(float)
    ee = e.set_index("household_ID")
    f["electric_water_heater"] = ee.water_heating_method_for_bathing.str.contains(
        "grid supply|electric kettle/heater", case=False, na=False).astype(float)
    f["electric_cooking"] = (ee.electricity_from_national_grid_used_for_cooking == "Yes").astype(float)
    f["home_business"] = (hh.is_there_business_carried_out_in_the_household == "Yes").astype(float)
    f = f.fillna({"n_ac": 0, "n_fridge": 0, "washing_machine": 0})

    # Kept ONLY for fairness / error analysis - never used as model inputs.
    f["socio_economic_class"] = hh.socio_economic_class
    f["csc_area"] = hh.electricity_provider_csc_area
    f["type_of_house"] = hh.type_of_house
    return f.rename_axis("hid").reset_index()


def main():
    x, q = load_daily_usage()
    m = monthly_totals(x)
    q["household_months"] = int(len(m))
    q["complete_household_months"] = int(m.complete.sum())
    cp = build_checkpoints(x, m)
    q["rows_dropped_house_empty_last_month"] = int((cp.prev1 < MIN_PREV_UNITS).sum())
    cp = cp[cp.prev1 >= MIN_PREV_UNITS]
    hh = household_features()
    cp = cp.merge(hh, on="hid", how="left")
    q["checkpoint_rows"] = int(len(cp))
    q["checkpoint_households"] = int(cp.hid.nunique())
    q["households_without_survey"] = int(cp[cp.members.isna()].hid.nunique())
    q["share_over_limit"] = round(float(cp.drop_duplicates(["hid", "ym"]).over_limit.mean()), 4)
    q["period"] = f"{cp.ym.min()} .. {cp.ym.max()}"

    m.to_parquet(DATA_PROC / "monthly.parquet")
    hh.to_parquet(DATA_PROC / "households.parquet")
    cp.to_parquet(DATA_PROC / "checkpoints.parquet")
    (REPORTS / "data_quality.json").write_text(json.dumps(q, indent=2))
    print(json.dumps(q, indent=2))


if __name__ == "__main__":
    main()
