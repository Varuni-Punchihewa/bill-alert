"""Step 1b - exploratory data analysis figures + summary table for the report.

    python -m src.eda
"""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import plotstyle as ps
from src.config import DATA_PROC, FIGURES, REPORTS, MONTHLY_BILLS
from src.features import add_derived
from src.tariff import estimate_bill

ps.apply()


def fig_bill_cliff():
    u = np.arange(0, 201)
    b = [estimate_bill(x) for x in u]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(u, b, color=ps.BLUE)
    ax.axvline(60, color=ps.INK2, lw=1, ls="--")
    ax.annotate(f"60 units -> Rs {estimate_bill(60):,.0f}\n61 units -> Rs {estimate_bill(61):,.0f}",
                xy=(61, estimate_bill(61)), xytext=(75, 4200), color=ps.INK,
                arrowprops=dict(arrowstyle="->", color=ps.INK2, lw=1), fontsize=9)
    ax.set(xlabel="Units used in a 30-day month", ylabel="Estimated bill (LKR)",
           title="One extra unit past 60 doubles the electricity bill")
    ps.save(fig, FIGURES / "eda_bill_cliff.png")


def fig_distribution(m):
    mc = m[m.complete]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.arange(0, 400, 5)
    ax.hist(mc.total.clip(upper=399), bins=bins, color=ps.BLUE, edgecolor=ps.SURFACE, linewidth=0.5)
    ax.axvline(60, color=ps.ORANGE, lw=2)
    share = (mc.total > mc.limit).mean()
    ax.text(64, ax.get_ylim()[1] * 0.9, f"{share:.0%} of household-months\nare above the limit",
            color=ps.INK, fontsize=9)
    ax.set(xlabel="Monthly units (smart-meter households, clipped at 400)", ylabel="Household-months",
           title=f"Monthly consumption: median {mc.total.median():.0f} units; "
                 f"{((mc.total - mc.limit).abs() <= 15).mean():.0%} of months end within 15 of the limit")
    ps.save(fig, FIGURES / "eda_monthly_distribution.png")


def fig_seasonality(m, bills):
    mc = m[m.complete].copy()
    s = mc.groupby(mc.ym.dt.to_timestamp()).agg(median=("total", "median"),
                                                 over=("over_limit", "mean"), n=("hid", "size"))
    s = s[s.n >= 300]
    bills = bills.copy()
    bills["ym"] = pd.to_datetime(bills.month).dt.to_period("M").dt.to_timestamp()
    nb = bills.groupby("ym").consumption.median()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(s.index, s["median"], marker="o", ms=4, label="Smart-meter households (this study)")
    ax.plot(nb.index, nb.values, marker="o", ms=4, color=ps.ORANGE, label="All 4,063 surveyed households (LECO bills)")
    ax.axhline(60, color=ps.INK2, lw=1, ls="--")
    ax.set(ylabel="Median monthly units", title="Median household stays above 60 units all year; Apr-May 2024 was the peak")
    ax.set_ylim(50, 112)
    ax.legend(loc="lower right")
    ps.save(fig, FIGURES / "eda_seasonality.png")


def fig_trajectories(cp):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sample = cp[cp.total.between(40, 90)][["hid", "ym"]].drop_duplicates().sample(40, random_state=1)
    for _, r in sample.iterrows():
        g = cp[(cp.hid == r.hid) & (cp.ym == r.ym)].sort_values("day")
        c = ps.ORANGE if g.over_limit.iloc[0] else ps.BLUE
        ax.plot(g.day, g.so_far, color=c, alpha=0.55, lw=1.2)
    d = np.arange(1, 16)
    ax.plot(d, 60 / 30 * d, color=ps.INK, ls="--", lw=1.5, label="2 units/day = exactly 60 a month")
    ax.plot([], [], color=ps.ORANGE, label="Month ended above limit")
    ax.plot([], [], color=ps.BLUE, label="Month ended at/below limit")
    ax.set(xlabel="Day of month", ylabel="Units used so far",
           title="Near the line, the first half of the month is ambiguous")
    ax.legend(loc="upper left")
    ps.save(fig, FIGURES / "eda_trajectories.png")


def fig_appliances(cp):
    one = cp.drop_duplicates(["hid", "ym"]).dropna(subset=["members"])
    rows = [("No AC", one.n_ac == 0), ("1+ AC", one.n_ac >= 1),
            ("No elec. water heater", one.electric_water_heater == 0), ("Elec. water heater", one.electric_water_heater == 1),
            ("1-2 people", one.members <= 2), ("5+ people", one.members >= 5)]
    labels = [r[0] for r in rows]
    vals = [one.loc[mask, "over_limit"].mean() for _, mask in rows]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = [ps.BLUE, ps.ORANGE] * 3
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9, color=ps.INK)
    ax.set(xlim=(0, 1.08), xlabel="Share of months above the limit",
           title="Household make-up shifts the odds of crossing 60 units")
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIGURES / "eda_household_factors.png")


def fig_pace(cp):
    d = add_derived(cp[cp.day.isin([3, 7, 15])])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, day in zip(axes, [3, 7, 15]):
        g = d[d.day == day].sample(4000, random_state=0)
        ax.scatter(g.pace_projection.clip(0, 300), g.total.clip(0, 300), s=3, alpha=0.25, color=ps.BLUE)
        ax.plot([0, 300], [0, 300], color=ps.INK2, lw=1)
        ax.axhline(60, color=ps.ORANGE, lw=1); ax.axvline(60, color=ps.ORANGE, lw=1)
        r = np.corrcoef(g.pace_projection, g.total)[0, 1]
        ax.set(title=f"Day {day}  (r = {r:.2f})", xlabel="Simple pace projection")
    axes[0].set_ylabel("Actual month total")
    fig.suptitle("Extrapolating the month gets better as the month goes on", x=0.01, y=1.04, ha="left",
                 fontsize=12, fontweight="bold")
    ps.save(fig, FIGURES / "eda_pace_vs_actual.png")


def main():
    m = pd.read_parquet(DATA_PROC / "monthly.parquet")
    cp = pd.read_parquet(DATA_PROC / "checkpoints.parquet")
    bills = pd.read_csv(MONTHLY_BILLS)
    bills["consumption"] = pd.to_numeric(bills.consumption.astype(str).str.replace(",", ""), errors="coerce")
    fig_bill_cliff(); fig_distribution(m); fig_seasonality(m, bills)
    fig_trajectories(cp); fig_appliances(cp); fig_pace(cp)
    one = cp.drop_duplicates(["hid", "ym"])
    summary = {
        "bill_households": int(bills.household_ID.nunique()),
        "bill_months": f"{bills.month.min()} .. {bills.month.max()}",
        "bill_median_units": float(bills.consumption.median()),
        "bill_share_over_60": round(float((bills.consumption > 60).mean()), 4),
        "smart_household_months_modelled": int(len(one)),
        "smart_median_units": round(float(one.total.median()), 1),
        "share_within_15_units_of_limit": round(float(((one.total - one.limit).abs() <= 15).mean()), 4),
    }
    (REPORTS / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
