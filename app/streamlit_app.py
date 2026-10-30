"""Bill Alert - will this month's electricity bill cross 60 units?

A front end for households and LECO customer-service staff with no ML
background. Run from the project root:

    streamlit run app/streamlit_app.py
"""
import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import TARIFF_EFFECTIVE, THRESHOLD_UNITS  # noqa: E402
from src.features import calendar_left, make_X, month_limit  # noqa: E402
from src.tariff import estimate_bill  # noqa: E402
from src.explain_text import reason_lines, rule_plain  # noqa: E402

st.set_page_config(page_title="Bill Alert", page_icon="⚡", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Share+Tech+Mono&display=swap');
:root { --paper:#f5f7f6; --ink:#1c2b31; --glass:#dfe7e4; --under:#2e7a58; --over:#c0531d; --muted:#5c6a6f; }
html, body, [class*="css"], .stMarkdown, .stText, label, input, button, textarea { font-family:'Atkinson Hyperlegible', sans-serif !important; }
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stNumberInputField"], [data-testid="stMetricValue"], [data-baseweb="select"] * { font-family:'Atkinson Hyperlegible', sans-serif !important; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; }
h1 { font-size: 2.1rem !important; margin-bottom: 0 !important; }
.lede { color: var(--muted); font-size: 1.05rem; max-width: 62ch; margin-top: .2rem; }
.verdict { font-size: 1.75rem; font-weight: 700; line-height: 1.25; margin: .2rem 0 .3rem 0; max-width: 30ch; }
.verdict.over { color: var(--over); } .verdict.under { color: var(--under); }
.chance { color: var(--muted); font-size: 1.05rem; margin-bottom: 1.1rem; }
.meter { background: var(--glass); border-radius: 14px; padding: 18px 20px 14px 20px; margin: 4px 0 18px 0; }
.readout { font-family: 'Share Tech Mono', monospace; font-size: 2.6rem; line-height: 1; color: var(--ink); }
.readout small { font-family: 'Atkinson Hyperlegible', sans-serif; font-size: .95rem; color: var(--muted); margin-left: .4rem; }
.bills { display:flex; gap: 28px; flex-wrap: wrap; margin-top: 12px; }
.bill b { display:block; font-size: 1.35rem; }
.bill span { color: var(--muted); font-size: .92rem; }
.budget { border-left: 4px solid var(--ink); padding: 4px 0 4px 14px; font-size: 1.08rem; max-width: 60ch; margin-bottom: 18px; }
.reason { display:flex; gap:10px; align-items:baseline; margin: 6px 0; max-width: 62ch; }
.reason .dot { width:10px; height:10px; border-radius:50%; flex: 0 0 10px; transform: translateY(1px); }
.reason .s { color: var(--muted); font-size: .9rem; }
.dots { display:grid; grid-template-columns: repeat(20, 14px); gap: 5px; margin: 10px 0 6px 0; }
.dots i { width:14px; height:14px; border-radius: 50%; display:block; }
.note { color: var(--muted); font-size: .9rem; max-width: 70ch; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ resources
@st.cache_resource
def load_model():
    b = joblib.load(ROOT / "models" / "rulefit_bill_alert.joblib")
    return b["model"]


@st.cache_data
def load_card():
    card = {}
    for f in ["metrics.json", "explainability.json"]:
        p = ROOT / "reports" / f
        if p.exists():
            card.update(json.loads(p.read_text()))
    return card


EXAMPLES = {
    "Your own numbers": None,
    "Example: family of four, close to the line": dict(day=9, dim=30, prev_reading=12450, today_reading=12471,
                                                      week=15.5, p1=63, p2=58, p3=55, members=4, ac=0, fridge=1,
                                                      heater=False, washer=True, cooking=True, business=False),
    "Example: home with an air conditioner": dict(day=6, dim=31, prev_reading=30210, today_reading=30248,
                                                  week=44.0, p1=142, p2=131, p3=150, members=3, ac=1, fridge=1,
                                                  heater=True, washer=True, cooking=True, business=False),
    "Example: careful two-person home": dict(day=11, dim=30, prev_reading=8820, today_reading=8836,
                                             week=10.0, p1=47, p2=52, p3=44, members=2, ac=0, fridge=1,
                                             heater=False, washer=False, cooking=False, business=False),
}

# ------------------------------------------------------------------- header
st.markdown("<h1>Will my electricity bill jump this month?</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p class='lede'>In Sri Lanka, going past {THRESHOLD_UNITS} units in a month moves the whole bill to a "
    f"higher rate: 60 units costs about Rs {estimate_bill(60):,.0f}, 61 units about Rs {estimate_bill(61):,.0f}. "
    "Enter what your meter shows today and this tool tells you, in the first half of the month, "
    "whether you are heading over the line and how much you can use each day to stay under.</p>",
    unsafe_allow_html=True)

tab_check, tab_trust, tab_about = st.tabs(["Check my month", "How reliable is this?", "About this tool"])

with tab_check:
    left, right = st.columns([1, 1.35], gap="large")
    with left:
        ex_name = st.selectbox("Start from", list(EXAMPLES), help="Pick an example to see how it works.")
        ex = EXAMPLES[ex_name] or dict(day=8, dim=30, prev_reading=0, today_reading=0, week=None, p1=None, p2=None,
                                       p3=None, members=4, ac=0, fridge=1, heater=False, washer=True,
                                       cooking=False, business=False)
        with st.form("inputs"):
            st.subheader("Your meter")
            c1, c2 = st.columns(2)
            prev_reading = c1.number_input("Reading on your last bill", min_value=0.0, value=float(ex["prev_reading"]),
                                           step=1.0, format="%.0f", help="Printed on the bill as 'present reading'.")
            today_reading = c2.number_input("Reading on your meter today", min_value=0.0,
                                            value=float(ex["today_reading"]), step=1.0, format="%.0f")
            c3, c4 = st.columns(2)
            day = c3.slider("Days since that bill reading", 1, 15, int(ex["day"]),
                            help="The alert is designed for the first half of the month, while there is time to act.")
            dim = c4.selectbox("Days in this billing period", [28, 29, 30, 31], index=[28, 29, 30, 31].index(ex["dim"]))
            knows_week = st.checkbox("I know how many units I used in the last 7 days", value=ex["week"] is not None)
            week = st.number_input("Units used in the last 7 days", min_value=0.0,
                                   value=float(ex["week"] or 0.0), step=0.5, format="%.1f",
                                   disabled=not knows_week)

            st.subheader("Your last three bills")
            b1, b2, b3 = st.columns(3)
            p1 = b1.number_input("Last month", min_value=0, value=int(ex["p1"] or 0), step=1)
            p2 = b2.number_input("2 months ago", min_value=0, value=int(ex["p2"] or 0), step=1,
                                 help="Leave as 0 if you don't have it.")
            p3 = b3.number_input("3 months ago", min_value=0, value=int(ex["p3"] or 0), step=1)

            st.subheader("Your home")
            h1, h2, h3 = st.columns(3)
            members = h1.number_input("People living here", 1, 15, int(ex["members"]))
            ac = h2.number_input("Air conditioners", 0, 8, int(ex["ac"]))
            fridge = h3.number_input("Fridges / freezers", 0, 4, int(ex["fridge"]))
            heater = st.checkbox("Electric water heater for bathing", value=ex["heater"])
            washer = st.checkbox("Washing machine", value=ex["washer"])
            cooking = st.checkbox("We cook with electricity (rice cooker, induction, oven)", value=ex["cooking"])
            business = st.checkbox("We run a business from home", value=ex["business"])
            submitted = st.form_submit_button("Check my month", type="primary", use_container_width=True)

    with right:
        so_far = today_reading - prev_reading
        if today_reading == 0 or so_far < 0 or p1 == 0:
            st.markdown("### Enter your two meter readings and last month's bill")
            st.markdown("<p class='note'>Use the readings printed on your LECO/CEB bill and the number on your meter "
                        "today. Or choose an example from the list to see what the result looks like.</p>",
                        unsafe_allow_html=True)
        else:
            today = dt.date.today()
            period_end = today + dt.timedelta(days=int(dim - day))
            wk_left, hol_left = calendar_left(today, period_end)
            if knows_week:
                last7 = week
            else:   # estimate the last week from the month-to-date average
                last7 = so_far / day * 7 if day >= 7 else np.nan
            last3 = (last7 / 7 * 3) if not np.isnan(last7) else so_far / day * min(day, 3)
            row = pd.DataFrame([{
                "day": day, "days_in_month": dim, "so_far": so_far, "last7": last7, "last3": last3,
                "weekend_days_left": wk_left, "holidays_left": hol_left,
                "prev1": p1, "prev2": p2 or np.nan, "prev3": p3 or np.nan, "month": today.month,
                "members": members, "n_ac": ac, "n_fridge": fridge, "electric_water_heater": int(heater),
                "washing_machine": int(washer), "electric_cooking": int(cooking), "home_business": int(business)}])
            X = make_X(row)
            model = load_model()
            prob = float(model.predict_proba(X)[0, 1])
            limit = float(month_limit(dim))
            pace = so_far / day * dim
            left_units = limit - so_far
            days_left = dim - day
            per_day_now = so_far / day

            if prob >= 0.5:
                verdict = f"You are likely to go over {THRESHOLD_UNITS} units this month."
                cls = "over"
            else:
                verdict = f"You are likely to stay within {THRESHOLD_UNITS} units this month."
                cls = "under"
            sure = max(prob, 1 - prob)
            if sure >= 0.95:
                odds = "more than 9 times out of 10"
            else:
                odds = f"about {int(round(sure * 10))} times out of 10"
            shown = min(max(prob, 0.01), 0.99)
            st.markdown(f"<div class='verdict {cls}'>{verdict}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chance'>Homes in a situation like yours ended up this way "
                        f"<b>{odds}</b> (chance of going over: {'<1%' if prob < 0.01 else '>99%' if prob > 0.99 else f'{shown:.0%}'})."
                        f"</div>", unsafe_allow_html=True)

            # ---- the meter strip
            scale = max(limit * 1.6, pace * 1.1, so_far * 1.2)
            used_w = min(so_far / scale, 1) * 100
            proj_w = min(max(pace - so_far, 0) / scale, 1 - used_w / 100) * 100
            line_x = limit / scale * 100
            ink_over = "var(--over)" if pace > limit else "var(--under)"
            if pace > limit:
                compare_bill = (f"<div class='bill'><b>Rs {estimate_bill(limit, dim):,.0f}</b>"
                                f"<span>if you keep it to {limit:.0f} units</span></div>")
            else:
                compare_bill = (f"<div class='bill'><b style='color:var(--over)'>Rs {estimate_bill(limit + 1, dim):,.0f}"
                                f"</b><span>if you slip to {limit + 1:.0f} units</span></div>")
            st.markdown(f"""
<div class="meter">
  <div class="readout">{so_far:,.0f}<small>units used in {day} day{'s' if day > 1 else ''}</small></div>
  <svg viewBox="0 0 100 14" preserveAspectRatio="none" style="width:100%;height:46px;margin-top:10px" role="img"
       aria-label="{so_far:.0f} units used, about {pace:.0f} expected by month end, limit {limit:.0f}">
    <rect x="0" y="3" width="100" height="8" rx="2" fill="#c9d4d0"/>
    <rect x="0" y="3" width="{used_w:.2f}" height="8" rx="2" fill="var(--ink)"/>
    <rect x="{used_w:.2f}" y="3" width="{proj_w:.2f}" height="8" fill="{ink_over}" opacity="0.45"/>
    <line x1="{line_x:.2f}" x2="{line_x:.2f}" y1="0" y2="14" stroke="var(--over)" stroke-width="0.6"/>
  </svg>
  <div style="display:flex;justify-content:space-between;color:var(--muted);font-size:.9rem">
    <span>Dark: used so far. Light: the rest of the month at your current pace (about {pace:,.0f} units).</span>
    <span style="color:var(--over);white-space:nowrap">| {limit:.0f}-unit line</span>
  </div>
  <div class="bills">
    <div class="bill"><b style="color:{ink_over}">Rs {estimate_bill(pace, dim):,.0f}</b><span>at your current pace (~{pace:,.0f} units)</span></div>
    {compare_bill}
  </div>
</div>""", unsafe_allow_html=True)

            # ---- what to do
            if left_units <= 0:
                st.markdown(f"<div class='budget'>You have already passed the {limit:.0f}-unit line this month, "
                            f"so this month's bill will use the higher rate. Your next chance to stay under starts "
                            f"with the next bill.</div>", unsafe_allow_html=True)
            else:
                allowed = left_units / max(days_left, 1)
                compare = (f"Lately you have used about <b>{(last7 / 7 if not np.isnan(last7) else per_day_now):.1f} "
                           f"units a day</b>.")
                recent = last7 / 7 if not np.isnan(last7) else per_day_now
                hard = (" That means cutting your daily use by more than half, so this month may be hard to save."
                        if recent > 0 and allowed < 0.5 * recent else "")
                st.markdown(f"<div class='budget'>To stay under, keep to about <b>{allowed:.1f} units a day</b> "
                            f"for the remaining {days_left} days. {compare}{hard}</div>", unsafe_allow_html=True)

            # ---- why
            st.markdown("#### Why the tool thinks this")
            fc = model.feature_contributions(X).iloc[0]
            reasons = reason_lines(list(fc.index), fc.to_numpy())
            st.markdown("<p class='note'>Compared with an average home in our data:</p>", unsafe_allow_html=True)
            for r in reasons:
                col = "var(--over)" if r["up"] else "var(--under)"
                st.markdown(f"<div class='reason'><span class='dot' style='background:{col}'></span>"
                            f"<span>{r['text']} <span class='s'>({r['strength']})</span></span></div>",
                            unsafe_allow_html=True)
            if reasons and all(r["up"] != (prob >= 0.5) for r in reasons[:3]):
                st.markdown("<p class='note'>These pull the other way, but not by enough: "
                            + ("most homes in our data go well over 60 units, so yours is still likely to go over."
                               if prob >= 0.5 else
                               "your numbers so far still leave you comfortably on the safe side.")
                            + "</p>", unsafe_allow_html=True)

            contrib = model.contributions(X)[0]
            is_true = model._rules_true(model._impute(X))[0]
            active = [k for k in np.argsort(-np.abs(contrib)) if is_true[k] and abs(contrib[k]) >= 0.05]
            over = prob >= 0.5
            support = [k for k in active if (model.coef_[k] > 0) == over]
            against = [k for k in active if (model.coef_[k] > 0) != over]

            def rule_line(k, muted=False):
                up = model.coef_[k] > 0
                col = "var(--over)" if up else "var(--under)"
                style = " style='opacity:.62'" if muted else ""
                return (f"<div class='reason'{style}><span class='dot' style='background:{col}'></span>"
                        f"<span>{rule_plain(model.rules_[k])} <span class='s'>"
                        f"({'points to going over' if up else 'points to staying under'})</span></span></div>")

            with st.expander(f"See the rules that matched your home ({len(active)} of {len(model.rules_)})"):
                st.markdown("<p class='note'>The tool is a short rulebook learned from thousands of real months. "
                            "Each rule that is true for your home nudges the chance up or down; the nudges simply "
                            "add up to the answer above.</p>", unsafe_allow_html=True)
                if support:
                    st.markdown(f"**Rules behind the answer** (strongest first)", unsafe_allow_html=True)
                    for k in support[:6]:
                        st.markdown(rule_line(k), unsafe_allow_html=True)
                if against:
                    st.markdown("<p class='note' style='margin-top:.8rem'><b>Small corrections that pull the other way.</b> "
                                "Some rules fine-tune the stronger rules above, so on their own they can look surprising. "
                                "Read them together with the rules above.</p>", unsafe_allow_html=True)
                    for k in against[:4]:
                        st.markdown(rule_line(k, muted=True), unsafe_allow_html=True)

            tips = []
            if ac:
                tips.append("An air conditioner uses roughly 1 unit for every hour it runs. One hour less a day "
                            f"saves about {days_left} units by the end of the month.")
            if heater:
                tips.append("An electric water heater uses about 3 units an hour. Switching it on only just "
                            "before bathing makes a big difference.")
            tips.append("Ironing clothes in one weekly session and keeping the fridge door closed are easy savings.")
            with st.expander("Ways to cut down (rough guide)"):
                for t in tips:
                    st.markdown(f"- {t}")
            st.markdown(f"<p class='note'>Bill amounts are estimates using the PUCSL domestic tariff effective "
                        f"{TARIFF_EFFECTIVE}, before taxes and levies.</p>", unsafe_allow_html=True)

card = load_card()
with tab_trust:
    tm = card.get("test_metrics", {})
    acc = tm.get("accuracy")
    st.markdown("### How often is the tool right?")
    if acc:
        right_n = int(round(acc * 100))
        dots = "".join(f"<i style='background:{'var(--under)' if i < right_n else 'var(--over)'}'></i>"
                       for i in range(100))
        st.markdown(
            f"<p class='lede'>We tested the tool on <b>5 months of real meter data (Jul-Dec 2024)</b> from Sri Lankan "
            f"homes, which it had never seen while it was being built. For every 100 alerts it gave in the first half "
            f"of the month, <b>about {right_n} were right</b>.</p><div class='dots'>{dots}</div>"
            f"<p class='note'>Green: the alert was right. Orange: it was wrong.</p>", unsafe_allow_html=True)
        by_day = card.get("accuracy_by_day", {}).get("RuleFit (final)", {})
        if by_day:
            d = {int(k): v for k, v in by_day.items()}
            early = np.mean([d[k] for k in d if k <= 3])
            mid = np.mean([d[k] for k in d if 4 <= k <= 9])
            late = np.mean([d[k] for k in d if k >= 10])
            st.markdown("#### It gets surer as the month goes on")
            c = st.columns(3)
            c[0].metric("Days 1-3", f"{early:.0f} in 100 right")
            c[1].metric("Days 4-9", f"{mid:.0f} in 100 right")
            c[2].metric("Days 10-15", f"{late:.0f} in 100 right")
        st.markdown("#### When to be careful")
        st.markdown("- Most mistakes happen when a home ends the month **within about 5 units of the line**. "
                    "If the tool says you are close, treat it as a warning either way.\n"
                    "- It has learned from homes in the LECO areas along the western and southern coast. "
                    "Homes elsewhere may behave differently.\n"
                    "- It does not know about one-off events like guests staying, a new appliance or a power cut.")
    else:
        st.info("Run the training script to see test results here.")

with tab_about:
    st.markdown("### What this tool is")
    st.markdown(
        "This tool learns from more than 37,000 months of smart-meter readings from Sri Lankan homes "
        "(the RECON-SL study by LIRNEasia with Lanka Electricity Company), together with a household survey. "
        "It looks at how much you have used so far, how quickly, what your recent bills were and what is in your home, "
        "and compares that with how thousands of similar months actually ended.")
    st.markdown("### How it decides")
    st.markdown(
        "Instead of a hidden formula, the tool is a **rulebook** of short IF-THEN rules found in the data, such as "
        "*\"IF at the current pace the month would end above the limit AND last month's bill was over 60 units, "
        "THEN the chance of going over is higher\"*. Every rule carries a weight. For your home, the tool adds up "
        "the weights of the rules that are true, and that sum is the answer. Nothing else is involved, so every "
        "answer can be traced back to the rules that produced it.")
    with st.expander("See the whole rulebook"):
        m = load_model()
        tb = m.rule_table()
        tb = tb.assign(**{
            "Rule": tb.rule.map(rule_plain),
            "Points to": np.where(tb.weight > 0, "going over", "staying under"),
            "Strength": pd.cut(tb.weight.abs(), [0, 0.3, 1, np.inf], labels=["weak", "medium", "strong"]).astype(str),
            "Applies to": (tb.support * 100).round(0).astype(int).astype(str) + "% of months"})
        st.dataframe(tb[["Rule", "Points to", "Strength", "Applies to"]], hide_index=True, use_container_width=True)
    st.markdown("### Your information")
    st.markdown("Nothing you type is stored or sent anywhere. The calculation happens on the server running this page "
                "and is forgotten when you close it. The training data contains no names or addresses.")
    st.markdown("### Limits")
    st.markdown("This is a guide, not your bill. Solar (net-metered) homes are not covered. "
                "Bill amounts are estimates using the PUCSL domestic tariff and do not include taxes or levies.")
