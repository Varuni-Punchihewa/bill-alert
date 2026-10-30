"""Plain-language explanations for people with no ML background.

RuleFit's prediction is exactly:
    log-odds = baseline (an average home) + sum of centred rule contributions
so every sentence shown in the app is built from the model's own numbers -
no approximation. This module

  * turns a rule such as "pace_ratio > 1.02 and prev1 <= 61.5" into words, and
  * groups contributions into five everyday concepts (pace, recent days,
    past bills, home, calendar) for the headline reasons.
"""
import math

from src.features import FRIENDLY

GROUPS = {
    "so_far": "pace", "pace_projection": "pace", "pace_ratio": "pace", "units_left": "pace",
    "allowed_per_day": "pace", "limit": "pace", "day": "pace", "days_in_month": "pace",
    "last7": "recent", "last3": "recent", "recent_rate": "recent", "recent3_rate": "recent",
    "allowed_vs_recent": "recent", "trend_ratio": "recent",
    "prev1": "history", "prev2": "history", "prev3": "history", "history_avg": "history",
    "history_ratio": "history", "history_max": "history", "history_trend": "history",
    "remaining_by_history": "history",
    "members": "home", "n_ac": "home", "n_fridge": "home", "electric_water_heater": "home",
    "washing_machine": "home", "electric_cooking": "home", "home_business": "home",
    "month": "calendar", "weekend_days_left": "calendar", "holidays_left": "calendar",
}

# Contributions are centred on an average home, so the wording is comparative.
SENTENCES = {
    "pace": ("You have used more so far this month than a typical home at this point.",
             "You have used less so far this month than a typical home at this point."),
    "recent": ("Your last few days were heavier than a typical home's, for what you have left.",
               "Your last few days were lighter than a typical home's, for what you have left."),
    "history": ("Your past bills were higher than a typical home's.",
                "Your past bills were lower than a typical home's."),
    "home": ("Your household tends to use more electricity than a typical home.",
             "Your household tends to use less electricity than a typical home."),
    "calendar": ("Homes tend to use more at this time of year.",
                 "Homes tend to use less at this time of year, and with the holidays ahead."),
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
BINARY = {"electric_water_heater": ("an electric water heater", "electric water heater"),
          "washing_machine": ("a washing machine", "washing machine"),
          "electric_cooking": ("electric cooking", "electric cooking"),
          "home_business": ("a business at home", "business at home")}
COUNTS = {"n_ac": "air conditioner(s)", "members": "people at home", "n_fridge": "fridge(s)/freezer(s)"}
UNITS = {"so_far": "units used so far", "pace_projection": "units projected at the current pace",
         "remaining_by_history": "units projected from usual habits", "units_left": "units left before the limit",
         "prev1": "units on last month's bill", "prev2": "units on the bill 2 months ago",
         "prev3": "units on the bill 3 months ago", "history_avg": "units on an average recent bill",
         "history_max": "units on the highest recent bill", "last7": "units in the last 7 days",
         "last3": "units in the last 3 days", "limit": "units allowed this month",
         "weekend_days_left": "weekend days left", "holidays_left": "holidays left"}
RATES = {"allowed_per_day": "units a day allowed from now on", "recent_rate": "units a day over the last week",
         "recent3_rate": "units a day over the last 3 days"}


def parse(rule):
    """'a > 1 and b <= 2' -> [('a', '>', 1.0), ('b', '<=', 2.0)]"""
    out = []
    for c in rule.split(" and "):
        f, op, v = c.strip().split(" ")
        out.append((f, op, float(v)))
    return out


def condition_short(f, op, v):
    """For the report: friendly name + symbol."""
    sym = ">" if op == ">" else "≤"
    if f in BINARY:
        return ("Has " if op == ">" else "No ") + BINARY[f][1]
    if f in COUNTS:
        return f"{FRIENDLY[f]} ≥ {math.floor(v) + 1}" if op == ">" else f"{FRIENDLY[f]} ≤ {math.floor(v)}"
    if f in ("pace_ratio", "history_ratio"):
        return f"{FRIENDLY[f]} {sym} {v:.0%}"
    if abs(v) < 0.5 and f in UNITS:
        v = 0.0
    return f"{FRIENDLY.get(f, f)} {sym} {v:.3g}"


def condition_plain(f, op, v):
    """For the app: a short everyday phrase."""
    more = op == ">"
    if f in BINARY:
        return ("the home has " if more else "the home does not have ") + BINARY[f][0]
    if f in COUNTS:
        return (f"at least {math.floor(v) + 1} " if more else f"at most {math.floor(v)} ") + COUNTS[f]
    if f == "pace_ratio":
        return f"at the current pace the month would end {'above' if more else 'at or below'} {v:.0%} of the limit"
    if f == "history_ratio":
        return f"the usual bill is {'above' if more else 'at or below'} {v:.0%} of the limit"
    if f == "allowed_vs_recent":
        return (f"the units you may still use per day are {'more than' if more else 'at most'} "
                f"{v:.2f} times your recent daily use")
    if f == "trend_ratio":
        return (f"the last 3 days ran {'faster than' if more else 'at most'} {v:.2f} times "
                f"the month's average pace")
    if f == "history_trend":
        return f"bills changed by {'more than' if more else 'at most'} {v:+.0f} units over 3 months"
    if f == "day":
        return f"it is {'after' if more else 'on or before'} day {math.floor(v)} of the month"
    if f == "month":
        m = min(max(math.floor(v), 1), 12)
        return f"the bill is for a month {'after' if more else 'up to'} {MONTHS[m - 1]}"
    if f == "days_in_month":
        return f"the billing month is {'longer than' if more else 'at most'} {v:.0f} days"
    if f in RATES:
        return f"{'more than' if more else 'at most'} {v:.1f} {RATES[f]}"
    if f in UNITS:
        num = f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}"
        return f"{'more than' if more else 'at most'} {num} {UNITS[f]}"
    return f"{FRIENDLY.get(f, f)} {'>' if more else '≤'} {v:.3g}"


def rule_short(rule):
    return "  AND  ".join(condition_short(*c) for c in parse(rule))


def rule_plain(rule):
    s = " and ".join(condition_plain(*c) for c in parse(rule))
    return s[0].upper() + s[1:]


def rule_group(rule):
    gs = [GROUPS.get(f, "pace") for f, _, _ in parse(rule)]
    return max(set(gs), key=gs.count)


# ------------------------------------------------------------ headline reasons
def grouped_reasons(names, scores):
    """[(group, total contribution)] sorted by absolute size. `names` are feature names."""
    tot = {}
    for n, s in zip(names, scores):
        g = GROUPS.get(n, "pace")
        tot[g] = tot.get(g, 0.0) + float(s)
    return sorted(tot.items(), key=lambda kv: -abs(kv[1]))


def strength(score):
    a = abs(score)
    word = "raises" if score > 0 else "lowers"
    if a >= 2.0:
        return f"{word} the chance a lot"
    if a >= 0.6:
        return f"{word} the chance"
    return f"{word} the chance a little"


def reason_lines(names, scores, k=4):
    out = []
    for g, s in grouped_reasons(names, scores)[:k]:
        if abs(s) < 0.15:
            continue
        up, down = SENTENCES[g]
        out.append({"group": g, "score": s, "up": s > 0, "text": up if s > 0 else down,
                    "strength": strength(s)})
    return out
