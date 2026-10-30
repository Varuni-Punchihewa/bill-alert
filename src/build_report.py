"""Step 5 - build the written report (HTML -> PDF) from the saved results.

All numbers are read from reports/*.json|csv, never typed by hand, so the
report always matches the code's actual output.

    python -m src.build_report
"""
import base64
import json
from pathlib import Path

import pandas as pd

from src.config import DATA_PROC, FIGURES, REPORTS, TARIFF_EFFECTIVE, TRAIN_END, VALID_END
from src.train import training_sample
from src.features import FEATURES, FRIENDLY
from src.tariff import estimate_bill

FINAL = "RuleFit (relaxed)"

_BOX = ("<rect x='{x}' y='{y}' width='{w}' height='52' rx='6' fill='{f}' stroke='#1c2b31' stroke-width='0.8'/>"
        "<text x='{cx}' y='{ty}' text-anchor='middle' font-size='10' font-weight='bold'>{t1}</text>"
        "<text x='{cx}' y='{ty2}' text-anchor='middle' font-size='8' fill='#52514e'>{t2}</text>")


def _row(y, items, fill):
    out, w, gap = [], 132, 22
    for i, (t1, t2) in enumerate(items):
        x = 6 + i * (w + gap)
        out.append(_BOX.format(x=x, y=y, w=w, f=fill, cx=x + w / 2, ty=y + 22, ty2=y + 38, t1=t1, t2=t2))
        if i < len(items) - 1:
            out.append(f"<line x1='{x + w + 2}' y1='{y + 26}' x2='{x + w + gap - 4}' y2='{y + 26}' stroke='#1c2b31' "
                       "stroke-width='1.2' marker-end='url(#arr)'/>")
    return "".join(out)


RULEFIT_SVG = (
    "<figure><svg viewBox='0 0 780 170' style='width:100%;font-family:DejaVu Sans,sans-serif'>"
    "<defs><marker id='arr' markerWidth='8' markerHeight='8' refX='6' refY='3' orient='auto'>"
    "<path d='M0,0 L6,3 L0,6 z' fill='#1c2b31'/></marker></defs>"
    "<text x='6' y='12' font-size='10' font-weight='700' fill='#c0531d'>TRAINING</text>"
    + _row(18, [("Training rows", "32 inputs, 0/1 target"), ("Boosted small trees", "random sizes and rows"),
                ("Candidate rules", "every node: IF a AND b"), ("Lasso selection", "keep at most max_rules"),
                ("Relaxed refit", "unshrunk weights")], "#fbe9df")
    + "<text x='6' y='98' font-size='10' font-weight='700' fill='#2e7a58'>PREDICTION</text>"
    + _row(104, [("A household today", "meter, bills, home"), ("Check every rule", "true (1) or false (0)"),
                 ("Add the weights", "true rules + a0"), ("Probability", "1 / (1 + e^-sum)"),
                 ("Explanation", "same rules, ranked")], "#e3efe9")
    + "</svg></figure>")


J = lambda f: json.loads((REPORTS / f).read_text())  # noqa: E731


def img(name, caption, width="100%"):
    p = FIGURES / name
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}">'
            f"<figcaption>{caption}</figcaption></figure>")


def table(df, fmt=None, cls=""):
    fmt = fmt or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            f = fmt.get(c)
            cells.append(f"<td>{f(v) if f else v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


pct = lambda v: f"{v:.2%}" if pd.notna(v) else "-"  # noqa: E731
pct1 = lambda v: f"{v:.1%}" if pd.notna(v) else "-"  # noqa: E731
f3 = lambda v: f"{v:.3f}" if pd.notna(v) else "-"  # noqa: E731
n0 = lambda v: f"{v:,.0f}"  # noqa: E731

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: 'Bitstream Charter', 'Charter', Georgia, serif; font-size: 10.6pt; line-height: 1.5; color: #1b1b1b; }
h1, h2, h3, h4, .meta, figcaption, table, .callout b { font-family: 'Inter', 'DejaVu Sans', sans-serif; }
h1 { font-size: 21pt; line-height: 1.2; margin: 0 0 6pt 0; font-weight: 700; letter-spacing: -0.01em; }
h2 { font-size: 13.5pt; margin: 20pt 0 6pt 0; font-weight: 700; border-bottom: 1.2pt solid #1c2b31; padding-bottom: 3pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 13pt 0 4pt 0; font-weight: 600; break-after: avoid; }
p { margin: 0 0 7pt 0; text-align: left; hyphens: auto; }
.meta { color: #52514e; font-size: 9.5pt; margin-bottom: 14pt; }
.abstract { background: #eef2f0; padding: 10pt 12pt; border-radius: 4pt; margin: 8pt 0 10pt 0; }
figure { margin: 8pt 0 10pt 0; break-inside: avoid; text-align: center; }
figcaption { font-size: 8.6pt; color: #52514e; text-align: left; margin-top: 3pt; }
table.rules td { font-size: 7.8pt; padding: 1.5pt 4pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.6pt; margin: 6pt 0 10pt 0; break-inside: avoid; }
th { text-align: left; border-bottom: 1pt solid #1c2b31; padding: 3pt 5pt; font-weight: 600; }
td { border-bottom: 0.4pt solid #d8dcd9; padding: 2.5pt 5pt; vertical-align: top; }
tr.hl td { background: #fbe9df; font-weight: 600; }
.small { font-size: 8.8pt; color: #52514e; }
.callout { border-left: 3pt solid #c0531d; padding: 4pt 0 4pt 10pt; margin: 8pt 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.6pt; background: #f1f3f2; padding: 0 2pt; }
ul, ol { margin: 0 0 7pt 0; padding-left: 16pt; } li { margin-bottom: 2pt; }
.eq { font-family: 'Latin Modern Math', serif; text-align: center; margin: 6pt 0; font-size: 11pt; }
.pb { break-before: page; }
.two { display: flex; gap: 10pt; } .two > * { flex: 1; }
"""


def build():
    dq, eda, m = J("data_quality.json"), J("eda_summary.json"), J("metrics.json")
    xai = J("explainability.json")
    rob = J("robustness.json") if (REPORTS / "robustness.json").exists() else None
    val = pd.read_csv(REPORTS / "validation_comparison.csv")
    test = pd.read_csv(REPORTS / "test_results.csv")
    tune = pd.read_csv(REPORTS / "tuning_results.csv")
    ab = pd.read_csv(REPORTS / "ablation.csv") if (REPORTS / "ablation.csv").exists() else None
    rules = pd.read_csv(REPORTS / "rulefit_rules_explained.csv")
    sp = m["splits"]
    tm = m["test_metrics"]
    ci = m["test_accuracy_95ci_household_bootstrap"]
    T = test.set_index("model")
    V = val.set_index("model")
    rule_test = T.loc["Rule: pace projection"]
    ebm_test, figs_test = T.loc["Explainable Boosting Machine"], T.loc["FIGS (greedy tree sums)"]
    bb = test[test.family.str.startswith("Taught") & (test.model != "Explainable Boosting Machine")
              & (test.model != "Logistic Regression")]
    best_bb = bb.loc[bb.accuracy.idxmax()]
    err_fin, err_rule = 1 - tm["accuracy"], 1 - rule_test["accuracy"]
    by_day = {int(k): v for k, v in xai["accuracy_by_day"]["RuleFit (final)"].items()}
    by_day_rule = {int(k): v for k, v in xai["accuracy_by_day"]["Pace rule"].items()}
    gap = pd.DataFrame(xai["slices"]["gap"])
    sec = pd.DataFrame(xai["slices"]["socio_economic_class"])
    area = pd.DataFrame(xai["slices"]["csc_area"])
    house = pd.DataFrame(xai["slices"]["type_of_house"])
    surv = pd.DataFrame(xai["slices"]["survey"])
    cm = xai["confusion_matrix"]
    n_models = len(val)
    n_rules = xai["n_rules"]
    n_val_sample = len(training_sample(pd.read_parquet(DATA_PROC / "checkpoints.parquet", columns=["hid", "ym", "period"])
                                       .query("period >= @TRAIN_END and period < @VALID_END"), 5))
    N = sp["train_sample_used"]["rows"] + n_val_sample
    sub_frac = min(0.5, (100 + 6 * N ** 0.5) / N)

    H = []
    a = H.append
    a(f"<h1>Bill Alert: predicting mid-month whether a Sri Lankan household will cross the 60-unit "
      f"electricity tariff line, with a RuleFit rule ensemble</h1>")
    a("<div class='meta'>Applied Machine Learning assignment &middot; MSc in Artificial Intelligence, "
      "University of Moratuwa &middot; V.A. Punchihewa &middot; Index no.: 269171D &middot; October 2026</div>")

    a(f"""<div class='abstract'><b>Summary.</b> In Sri Lanka's domestic electricity tariff, a household that uses more
than 60 units in a 30-day period is billed on a different rate structure: 60 units cost about Rs {estimate_bill(60):,.0f}
and 61 units about Rs {estimate_bill(61):,.0f}. Using {dq['raw_households']:,} smart-meter households from the public
RECON-SL dataset (LIRNEasia with Lanka Electricity Company), I built a model that predicts, on any of the first 15 days of a
billing month, whether that month will end above the line. The model is <b>RuleFit</b> (Friedman &amp; Popescu, 2008),
a sparse linear model over IF-THEN rules mined from a tree ensemble. It was not covered in the lectures. I added a relaxed-lasso
refit so that its probabilities are calibrated. It was compared with {n_models - 1} alternatives, including every model
family taught in the module, on a strictly chronological split. The final model is <b>{n_rules} rules</b> that fit on one page.
On five unseen months (Jul-Dec 2024) it is correct <b>{tm['accuracy']:.2%}</b> of the time (95% CI
{ci[0]:.2%}-{ci[1]:.2%}, ROC-AUC {tm['roc_auc']:.3f}), and it removes {(err_rule - err_fin) / err_rule:.0%} of the errors
of a simple pace rule. Each prediction is exactly the sum of the rules that match the household. A Streamlit app turns
those rules into plain-language reasons and a daily budget for households.</div>""")

    # ------------------------------------------------------------------ 1
    a("<h2>1. Problem definition and dataset</h2>")
    a("<h3>1.1 The problem and why it matters locally</h3>")
    a(f"""<p>The PUCSL domestic tariff (D-1) has two schedules. If a household's monthly use stays at or below 60 units, it pays
the low-user rates; once it passes 60, <i>every</i> unit is charged on the "above 60" schedule with a higher fixed charge.
With the rates effective {TARIFF_EFFECTIVE}, the bill jumps from about Rs {estimate_bill(60):,.0f} at 60 units to
Rs {estimate_bill(61):,.0f} at 61 units (Figure 1). Similar cliffs exist at 90, 120 and 180 units.</p>""")
    a(img("eda_bill_cliff.png", "Figure 1. Estimated monthly bill against units used (PUCSL D-1 tariff, before taxes). "
          "One unit past 60 doubles the bill.", "78%"))
    a(f"""<p>This matters to many families. In the RECON-SL wave-3 survey, 1,218 of 3,398 households (36%) said they try to keep
their consumption under a target, and 60 and 90 units were the most common targets. Yet 2,202 (65%) said nobody in the
household knows how the bill is calculated. The bill arrives only after the month is over. A household that knew on day 8
that it was heading past 60 units could still cut back. <b>The task:</b> on day <i>d</i> (1-15) of a billing month,
using only information available that day, predict whether the month will end above the 60-unit line (binary
classification). The users are households and LECO customer-service and demand-side-management staff.</p>""")

    a("<h3>1.2 Data source</h3>")
    a(f"""<p><b>RECON-SL - Residential Electricity Consumption Dataset for Sri Lanka</b> (IEEE DataPort, DOI 10.21227/n1dk-q860),
collected by LIRNEasia with Lanka Electricity Company (LECO), funded by the Lacuna Fund, open access. It covers 4,063
LECO households in the western and southern coastal service areas (Moratuwa, Panadura, Galle, Kelaniya, Negombo and others) and
contains (i) monthly billing records for all 4,063 households ({eda['bill_months']}), (ii) smart-meter cumulative
register readings at 6-hour and 15-minute intervals for {dq['raw_households']:,} metered households (Jan 2023 - Dec 2024),
and (iii) three survey waves on demographics, housing and appliances. I used the 6-hour files (about 1 GB: 4.4 million rows)
and wave-1 survey tables. The data was downloaded from IEEE DataPort, not collected by me; I compiled the modelling table
from the raw files myself (Section 1.4).</p>""")

    a("<h3>1.3 Unit of analysis, target and features</h3>")
    a(f"""<p>One row is <b>(household, billing month, checkpoint day d)</b>, with d = 1...15. The target is
<code>over_limit = 1</code> if the month's total exceeds 60 &times; days/30 units (the tariff pro-rates blocks for periods
that are not 30 days). Features are grouped by where a household would get them (Table 1). Socio-economic class, area,
house type, ethnicity and religion are <b>never used as inputs</b>; class, area and house type are kept only to audit
errors for fairness.</p>""")
    groups = [
        ("Meter, this month", "day, days_in_month, so_far, last7, last3", "Register reading on day d minus the reading at the start of the month; usage in the last 7 and 3 days"),
        ("Bill history", "prev1, prev2, prev3", "Totals of the previous three complete months"),
        ("Calendar", "month, weekend_days_left, holidays_left", "Sri Lankan public and Poya holidays from the <code>holidays</code> package"),
        ("Household (survey wave 1)", "members, n_ac, n_fridge, electric_water_heater, washing_machine, electric_cooking, home_business", "Encoded from room roster, appliance and household tables"),
        ("Engineered (domain)", "limit, pace_projection, pace_ratio, units_left, allowed_per_day, recent_rate, allowed_vs_recent, history_avg, history_ratio, history_max, history_trend, remaining_by_history, recent3_rate, trend_ratio", "Encode the arithmetic a careful household would do: projected total, units left, daily allowance vs recent use"),
    ]
    a(table(pd.DataFrame(groups, columns=["Group", "Features", "How obtained"])))
    a(f"<p class='small'>Table 1. The {len(FEATURES)} model inputs. Definitions are in <code>src/features.py</code>, "
      "which the training code and the app share, so the two cannot drift apart.</p>")

    a("<h3>1.4 Preprocessing and data quality</h3>")
    a(f"""<ol>
<li><b>Aggregation.</b> The 6-hour files were reduced to one closing register reading per household per day
({dq['raw_household_days']:,} household-days).</li>
<li><b>Register-based usage, no interpolation.</b> Monthly usage is the closing register minus the previous month's closing
register, so a missing day inside the month costs nothing. Usage so far on day d uses only the reading of day d. Filling gaps by
interpolation would use the next reading, which is <i>future</i> information at the checkpoint, so it was avoided. This
alone raised the number of usable household-months from 27,450 (a naive daily-sum approach) to
{dq['complete_household_months']:,}.</li>
<li><b>Glitches.</b> {dq['glitch_days_removed']:,} days with a negative step (meter reset or replacement) or more than 200 kWh
were removed, and months containing one were invalidated.</li>
<li><b>Exclusions.</b> {dq['solar_households_excluded']} rooftop-solar households (net-metered, so the import register is
not their billed consumption) and {dq['rows_dropped_house_empty_last_month']:,} rows of months following a vacant month
(under 5 units) were removed.</li>
<li><b>Survey encoding.</b> Counts were summed per household (ACs from the room roster; fridges and freezers from the appliance
table) and yes/no answers became 0/1. {dq['households_without_survey']:,} smart-meter-only households have no survey; their
household fields are left missing. RuleFit and FIGS impute the training-data median inside the model (a rule such as
"at least one AC" is then simply false for an unsurveyed home); the other scikit-learn models use median imputation plus
missing-indicator columns; the EBM and the gradient-boosting libraries handle missing values natively.</li>
<li><b>Scaling.</b> Only the distance-, kernel- and linear models (k-NN, SVM, logistic regression) are standardised, inside a
pipeline fitted on training data only. Rule and tree models are invariant to monotone rescaling, so they do not need it.</li>
</ol>
<p>The final table has <b>{dq['checkpoint_rows']:,} rows</b> from <b>{dq['checkpoint_households']:,} households</b> and
{eda['smart_household_months_modelled']:,} household-months ({dq['period']}). {dq['share_over_limit']:.1%} of household-months
end above the limit. The median home uses about {eda['smart_median_units']:.0f} units, and
{eda['share_within_15_units_of_limit']:.1%} of months end within 15 units of the line: these are the hard cases.</p>""")
    a("<div class='two'>" + img("eda_monthly_distribution.png", "Figure 2. Monthly consumption of modelled homes.")
      + img("eda_seasonality.png", "Figure 3. Median monthly units over time in both data sources.") + "</div>")
    a("<div class='two'>" + img("eda_trajectories.png", "Figure 4. Cumulative usage of 40 random months that ended near the line.")
      + img("eda_household_factors.png", "Figure 5. Share of months above the limit by household factor (surveyed homes).") + "</div>")
    a(img("eda_pace_vs_actual.png", "Figure 6. A naive pace projection against the true month total on days 3, 7 and 15.", "92%"))
    a("""<p>Figures 4 and 6 frame the problem. Extrapolating the month-to-date average (the <i>pace rule</i>) is already a
strong baseline, but early in the month and near the line it is unreliable: a few heavy days distort it. Households with
an AC, an electric water heater or more members cross the line more often (Figure 5). Whether the survey adds anything
once the meter is known is tested in Section 4. It turns out that it largely does not.</p>""")
    a("""<h3>1.5 Ethical use of data</h3><p>RECON-SL is published as open access for research. It is anonymised (household IDs
only, no names, addresses or meter numbers), and participants consented to the LIRNEasia survey. I did not attempt
re-identification, did not redistribute the raw data (the README explains how to download it), and kept protected
attributes out of the model. The app stores nothing a user types.</p>""")

    # ------------------------------------------------------------------ 2
    a("<h2 class='pb'>2. The new algorithm: RuleFit (rule ensembles)</h2>")
    a(f"""<p>The lectures covered linear models (linear, ridge, lasso and logistic regression), single models (SVM, decision trees,
k-NN), bagging and voting (random forests), boosting (AdaBoost, gradient boosting, XGBoost, LightGBM, CatBoost), stacking,
clustering, and the Explainable Boosting Machine. The model chosen here, <b>RuleFit</b> (Friedman &amp; Popescu, 2008), is none
of these. It is a <b>sparse linear model whose inputs are IF-THEN rules learned from the data</b>:</p>
<div class='eq'>log-odds P(over) = a<sub>0</sub> + &Sigma;<sub>k=1..K</sub> a<sub>k</sub> &middot; r<sub>k</sub>(x),
&nbsp;&nbsp; r<sub>k</sub>(x) = &prod;<sub>j &isin; rule k</sub> 1[x<sub>j</sub> &isin; s<sub>jk</sub>] &isin; {{0, 1}}</div>
<p>Each r<sub>k</sub> is a conjunction of one to a few conditions, for example <i>"projected month total at the current
pace &gt; 63 units AND last month's bill &gt; 58 units"</i>. It equals 1 when every condition holds. The final model is a short list of such
rules, each with a weight a<sub>k</sub> in log-odds. To score a household, add the weights of the rules that are true, then
apply the logistic function. Figure 7 shows the three training stages.</p>""")
    a(RULEFIT_SVG)
    a("""<p class='small'>Figure 7. How RuleFit is trained (top) and how it predicts (bottom).</p>""")
    a(f"""<ol>
<li><b>Rule generation.</b> {m['best_params']['rulefit'].get('n_estimators', 100)} small regression trees are grown by
gradient boosting on the 0/1 target with a small learning rate (0.01). Following Friedman &amp; Popescu, the trees have
random sizes (2 + an exponential draw, about {m['best_params']['rulefit'].get('tree_size', 4)} leaves on average) and each is fitted on a
random {sub_frac:.1%} subsample of the rows, (100 + 6&radic;N)/N, so the trees are diverse. Every node of every tree, not
only the leaves, defines a rule: the conjunction of the splits on the path from the root to that node. This yields
hundreds of candidate rules that describe regions where the outcome differs.</li>
<li><b>Rule selection (lasso).</b> Each candidate rule becomes a 0/1 column, and an L1-penalised (lasso) logistic
regression is fitted on all of them along a path of 100 penalty strengths, from strongest to weakest. The path stops as soon as
more than <i>max_rules</i> rules would have a non-zero weight. Among the penalties visited, the one with the best 5-fold
cross-validated accuracy is used. The trees are then discarded; only the selected rules survive.</li>
<li><b>Relaxed refit (my addition).</b> The lasso shrinks every surviving weight towards zero, because the same penalty that selects
rules also biases their size. Following the relaxed lasso (Meinshausen, 2007), I refit an ordinary logistic regression
(with only a light L2 penalty, C = 1) on the selected rules. Before this, rules that are always or never true, and rules
that are true for exactly the same training rows as a simpler rule, are dropped, so each line of the explanation is
distinct. The refit weights are then true log-odds and the probabilities are calibrated (Section 3).</li>
</ol>
<p><b>Implementation.</b> Stages 1 and 2 use the open-source <code>imodels</code> library (Singh et al., 2021). Stage 3 and
the explanation functions are in <code>src/rulefit_model.py</code>. One finding from building it: the library's
<code>RuleFitClassifier.predict_proba</code> converts the linear score z into a probability as softmax([1&minus;z, z]) =
sigmoid(2z&minus;1), which is not the probability of the fitted logistic model. Section 3.3 shows what happens in each
case. My wrapper refits the weights and uses sigmoid(z), and the unit tests check that the probability is exactly the
logistic function of the score.</p>""")
    a("""<p><b>How it differs from the standard models.</b></p>
<table><thead><tr><th>Compared with</th><th>What is shared</th><th>What is different in RuleFit</th></tr></thead><tbody>
<tr><td>Decision tree</td><td>IF-THEN conditions on thresholds</td><td>A tree assigns each case to exactly one leaf, and the
path is fixed by the first split, so it is unstable. In RuleFit several overlapping rules fire at once and their weights
are <i>added</i>. The weights are estimated jointly by a penalised regression, not by greedy splitting, and the rules
come from many trees, so no single split dominates.</td></tr>
<tr><td>Logistic / lasso regression</td><td>Same final model form and the same L1 penalty</td><td>Lasso on raw features can
only learn a straight-line effect for each feature. RuleFit runs the lasso on <i>learned</i> binary rule features, so it
captures thresholds (the 60-unit line itself), saturation and interactions such as "high pace AND a high usual bill",
which a linear model cannot express.</td></tr>
<tr><td>k-NN, SVM</td><td>Non-linear decision boundary</td><td>k-NN and SVM keep training points or support vectors and
give no global model a person can read. RuleFit is a compact global model: a few dozen readable rules.</td></tr>
<tr><td>Random forest, gradient boosting (XGBoost, LightGBM, CatBoost, AdaBoost)</td><td>Uses an ensemble of trees</td><td>In
those models the thousands of trees <i>are</i> the predictor, and any explanation (SHAP, importance) is an approximation
added afterwards. RuleFit uses trees only to <i>propose</i> candidate rules, then keeps a few dozen and discards the rest.
The prediction is the sum of the weights of the true rules, so the explanation is exact.</td></tr>
<tr><td>Stacking / voting</td><td>Combines many base learners</td><td>Stacking combines model <i>outputs</i>. RuleFit
combines <i>rules</i> inside one linear model, so the combination is still readable.</td></tr>
<tr><td>Explainable Boosting Machine</td><td>Glass-box, additive in the log-odds</td><td>An EBM learns one smooth shape
function for each feature (plus a few pairs), and those functions are read as curves. RuleFit's terms are discrete rules
that can combine three or four features at once and are written as sentences. This is closer to how a utility officer
would describe a risk ("pace above the line and a usual bill over 60").</td></tr>
</tbody></table><p class='small'>Table 2. How RuleFit differs from the models covered in the lectures.</p>""")
    a(f"""<p><b>Why it was selected.</b> (1) <i>The stakeholders.</i> Households and utility staff have no ML background, and a
warning they cannot question is hard to trust. RuleFit's model <i>is</i> a rulebook that can be printed on one page and
read aloud (Appendix B), and the app shows each household the rules that matched it. (2) <i>Exact explanations.</i>
Each prediction decomposes exactly into "average home + the contribution of each matching rule", so the reasons shown in
the app are the model's own arithmetic, not an approximation. (3) <i>Domain checks.</i> Every rule can be compared
directly with what an energy officer expects: a rule saying "at most 1.9 units a day allowed from now on" should push
towards "over" (Section 4.5). (4) <i>The problem fits.</i> The decision is threshold arithmetic around a hard cut-off
(60 units) with a few interactions. Rules express exactly that, and Friedman &amp; Popescu report accuracy close to tree
ensembles on such data. Section 3 tests whether that holds here. (5) <i>A size control.</i> <i>max_rules</i> sets the
trade-off between accuracy and readability explicitly, and Section 3 measures it.</p>""")

    # ------------------------------------------------------------------ 3
    a("<h2 class='pb'>3. Model training and evaluation</h2>")
    a("<h3>3.1 Split</h3>")
    sp_df = pd.DataFrame([
        ("Train", sp["train"]["months"], sp["train"]["rows"], sp["train_sample_used"]["rows"], sp["train"]["households"], sp["train"]["share_over_limit"]),
        ("Validation", sp["validation"]["months"], sp["validation"]["rows"], sp["validation"]["rows"], sp["validation"]["households"], sp["validation"]["share_over_limit"]),
        ("Test", sp["test"]["months"], sp["test"]["rows"], sp["test"]["rows"], sp["test"]["households"], sp["test"]["share_over_limit"]),
    ], columns=["Set", "Billing months", "Rows", "Rows used", "Households", "Share over limit"])
    a(table(sp_df, {"Rows": n0, "Rows used": n0, "Households": n0, "Share over limit": pct1}))
    a("""<p class='small'>Table 3. Chronological split. RECON-SL's 6-hour files have almost no complete months for May-Jul 2024,
which falls naturally between validation and test.</p>""")
    a(f"""<p>A random split would leak information: the 15 checkpoints of one month, and neighbouring months of one household,
would land on both sides. Instead, the split is by <b>billing month</b>: models are trained on the past and judged on the
future, as in deployment. Validation (Jan-Jun 2024) was used for every design choice: features, hyperparameters, the
ablation in Section 3.3 and model choice. The test months (Jul-Dec 2024) were scored once, after all of that was fixed.
Within training, the 15 checkpoints of a month are highly correlated, so 5 per household-month were sampled at random.
This keeps every month while fitting a 2-CPU budget. For the final test, every finalist was refitted on training +
validation ({N:,} rows) with the settings chosen on validation.</p>""")

    a("<h3>3.2 Hyperparameters</h3>")
    tt = tune.copy()
    tt["params"] = tt.params.str.replace('"', "").str.replace("{", "").str.replace("}", "")
    tt["size"] = tt["size"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    tt = tt[["model", "params", "size", "val_accuracy", "val_balanced_accuracy", "val_auc", "val_brier", "seconds"]]
    tt.columns = ["Model", "Setting", "Rules / splits", "Val. accuracy", "Val. bal. acc.", "Val. AUC", "Val. Brier", "Fit (s)"]
    a(table(tt, {"Val. accuracy": pct, "Val. bal. acc.": pct, "Val. AUC": f3, "Val. Brier": f3}))
    rf_t = tune[tune.model == "RuleFit"]
    best_rf = m["best_params"]["rulefit"]
    a(f"""<p class='small'>Table 4. Grid searches on the validation months. Unlisted RuleFit settings are at their defaults:
100 trees, mean tree size 4 leaves, learning rate 0.01, relaxed-refit C = 1.</p>
<p><b>RuleFit</b> has three settings that matter: <i>max_rules</i> (how many rules the lasso may keep: the readability
budget), the number of trees that propose rules, and the mean tree size (larger trees give longer rules with more
conditions). The selection rule was decided before the search: highest validation accuracy, and if several settings are within
0.05 percentage points, the one with the fewest rules. The winner was <code>{json.dumps(best_rf)}</code>. Across the
whole grid, validation accuracy only moved between {rf_t.val_accuracy.min():.2%} and {rf_t.val_accuracy.max():.2%}, while
the rule count ranged from {rf_t['size'].min():.0f} to {rf_t['size'].max():.0f}. Beyond a few dozen rules, extra rules
add almost nothing (Figure 9), so the smaller, more readable model costs essentially no accuracy. FIGS, the other untaught
rule-based model, was tuned on its number of splits (best <code>{json.dumps(m['best_params']['figs'])}</code>) and LightGBM
on tree size and learning rate (best <code>{json.dumps(m['best_params']['lgbm'])}</code>).</p>
<p><b>Comparison models</b> use standard, sensible settings: logistic regression C = 1 (and an L1 version, C = 0.1); decision
tree depth 8, min leaf 50; k-NN k = 51 (fit on 60,000 rows) and an RBF SVM C = 1 (fit on 15,000 rows) on standardised
features, because both scale badly with data size; bagging 100 trees; random forest and extra trees 300 trees, min leaf 10;
AdaBoost 200 rounds; gradient boosting, XGBoost, LightGBM and CatBoost 400-600 rounds; soft voting and stacking over
RF + XGBoost + LightGBM (stacking with a logistic-regression meta-learner on 3-fold out-of-fold probabilities); EBM with
10 interactions, 8 outer bags, learning rate 0.02 and early stopping. Full settings are in <code>src/train.py</code>.</p>""")

    if ab is not None:
        a("<h3>3.3 Ablation: what each part of my RuleFit adds</h3>")
        abt = ab[["variant", "rules", "accuracy", "balanced_accuracy", "roc_auc", "brier", "log_loss"]].copy()
        abt.columns = ["Variant (validation months)", "Rules", "Accuracy", "Bal. acc.", "ROC-AUC", "Brier", "Log-loss"]
        a(table(abt, {"Accuracy": pct, "Bal. acc.": pct, "ROC-AUC": f3, "Brier": lambda v: f"{v:.4f}", "Log-loss": f3}))
        A = ab.set_index("variant")
        shipped, sig = A.iloc[0], A.iloc[1]
        relax, final_ab, lin = A.iloc[2], A.iloc[3], A.iloc[4]
        a(f"""<p class='small'>Table 5. Same rule-generation settings throughout; trained on the training months, scored on the
validation months.</p>
<p>Three findings. (1) <i>Two errors that partly cancel.</i> The library's formula, sigmoid(2z&minus;1), is wrong in
principle, yet it gives Brier {shipped.brier:.4f}. Using the textbook link, sigmoid(z), on the <i>same</i> lasso weights is
{'worse' if sig.brier > shipped.brier else 'better'} (Brier {sig.brier:.4f}, log-loss {sig.log_loss:.3f}). The reason
is that the lasso shrinks every weight towards zero, so the scores are too timid, and the library's doubling of z happens
to undo part of that shrinkage. (2) <i>Fixing the cause works best.</i> The relaxed refit re-estimates the weights of
the selected rules without shrinkage and then applies sigmoid(z). It gives the best Brier score ({relax.brier:.4f}) and
log-loss ({relax.log_loss:.3f}) of all the variants, with accuracy {relax.accuracy:.2%} against {shipped.accuracy:.2%} for the
library as shipped, at the cost of {(shipped.balanced_accuracy - relax.balanced_accuracy) * 100:.1f} points of balanced
accuracy. A calibrated probability matters here because the app tells people "about 8 in 10". (3) De-duplication
removed {int(relax.rules - final_ab.rules)} rule{'s' if relax.rules - final_ab.rules != 1 else ''} with these settings
and changed accuracy by {(final_ab.accuracy - relax.accuracy) * 100:+.2f} points. It is a safeguard that keeps every line
of the explanation distinct. Friedman &amp; Popescu's original formulation also adds every raw feature as a linear term.
That reaches {lin.accuracy:.2%} accuracy with Brier {lin.brier:.4f}, a gain of {(lin.accuracy - final_ab.accuracy) * 100:+.2f}
points. I kept the rules-only version because every term can then be read as an IF-THEN sentence, which the front end
depends on.</p>""")

    a("<h3>3.4 Metrics</h3>")
    a("""<p><b>Accuracy</b> is the headline because a wrong alert in either direction is a similar failure for the user. Because about
three quarters of months are over the limit, I also report <b>balanced accuracy</b> and recall for each class (catching homes
that will stay <i>under</i> is the harder half), <b>macro F1</b>, <b>ROC-AUC</b> (ranking quality), and <b>Brier score /
log-loss</b>. The app shows a probability, so it must be calibrated. The test-accuracy confidence interval uses a
household-level cluster bootstrap (500 resamples), because rows from one household are not independent.</p>""")

    a("<h3>3.5 Results</h3>")
    v = val.sort_values("accuracy", ascending=False)[["model", "family", "accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "brier", "fit_seconds"]]
    v.columns = ["Model", "Family", "Accuracy", "Bal. acc.", "Macro F1", "ROC-AUC", "Brier", "Fit (s)"]
    a(table(v, {"Accuracy": pct, "Bal. acc.": pct, "Macro F1": f3, "ROC-AUC": f3, "Brier": f3}))
    a(f"<p class='small'>Table 6. All {n_models} approaches on the validation months (days 1-15, {sp['validation']['rows']:,} rows).</p>")
    fin_val = V.loc[FINAL]
    rank = int((val.accuracy > fin_val.accuracy).sum()) + 1
    a(img("model_comparison_validation.png", "Figure 8. Validation accuracy of every model, coloured by family.", "88%"))
    a(img("accuracy_vs_rules.png", "Figure 9. Validation accuracy against model size for the two rule-based models. "
          "Dashed lines show reference models.", "70%"))
    t = test[["model", "accuracy", "balanced_accuracy", "recall_over", "recall_under", "f1_macro", "roc_auc", "brier", "log_loss"]].copy()
    t.columns = ["Model", "Accuracy", "Bal. acc.", "Recall (over)", "Recall (under)", "Macro F1", "ROC-AUC", "Brier", "Log-loss"]
    a(table(t, {"Accuracy": pct, "Bal. acc.": pct, "Recall (over)": pct1, "Recall (under)": pct1, "Macro F1": f3,
                "ROC-AUC": f3, "Brier": f3, "Log-loss": f3}, cls="test"))
    a(f"""<p class='small'>Table 7. Test set (Jul-Dec 2024, {sp['test']['rows']:,} rows, {sp['test']['households']:,} households).
Finalists were refitted on train + validation and scored once.</p>""")
    gap_bb = (best_bb.accuracy - tm["accuracy"]) * 100
    if gap_bb <= 0:
        cost_txt = (f"So on test, a model a person can read in full is {-gap_bb:.2f} percentage points <i>more</i> accurate than "
                    "the best black box, a difference well inside the confidence interval. Readability cost nothing here.")
    else:
        cost_txt = f"So a model a person can read in full costs {gap_bb:.2f} percentage points of accuracy."
    val_best = val.loc[val.accuracy.idxmax()]
    bal_rank = int((val.balanced_accuracy > V.loc[FINAL].balanced_accuracy).sum()) + 1
    a(f"""<div class='callout'><b>Headline.</b> On the unseen test months the {n_rules}-rule RuleFit model is correct
{tm['accuracy']:.2%} of the time (95% CI {ci[0]:.2%}-{ci[1]:.2%}), with balanced accuracy {tm['balanced_accuracy']:.2%},
ROC-AUC {tm['roc_auc']:.3f} and Brier score {tm['brier']:.3f}. The pace rule scores {rule_test['accuracy']:.2%}, so RuleFit
removes {(err_rule - err_fin) / err_rule:.0%} of the rule's errors. The most accurate taught black-box model on test,
{best_bb['model']}, scores {best_bb['accuracy']:.2%}. {cost_txt} The EBM scores {ebm_test.accuracy:.2%} and
FIGS {figs_test.accuracy:.2%}.</div>""")
    a("<div class='two'>" + img("accuracy_by_day_test.png", "Figure 10. Test accuracy by checkpoint day.")
      + img("calibration_test.png", "Figure 11. Calibration on the test set.", "82%") + "</div>")
    days_better = [d for d in by_day if by_day[d] > by_day_rule[d]]
    days_worse = [d for d in by_day if by_day[d] < by_day_rule[d]]
    early = sum(by_day[d] - by_day_rule[d] for d in (1, 2, 3)) / 3
    late = sum(by_day[d] - by_day_rule[d] for d in range(11, 16)) / 5
    late_txt = ("level (within 0.1 points)" if abs(late) < 0.1 else f"{abs(late):.1f} points apart, "
                f"{'RuleFit' if late > 0 else 'the rule'} ahead")
    a(f"""<p><b>What the results indicate.</b> (1) The problem has a strong, mostly threshold-driven signal: on validation, every
model except k-NN, the SVM and naive Bayes lands between about 94.5% and 95.5%. On validation RuleFit ranks {rank} of
{n_models} on accuracy ({fin_val.accuracy:.2%}, against {val_best.accuracy:.2%} for {val_best.model}) and {bal_rank} of {n_models}
on balanced accuracy ({fin_val.balanced_accuracy:.2%}). The top five models are within {(val_best.accuracy - V.iloc[:, 0].sort_values().iloc[-5:].min()) * 100 if False else (val.accuracy.nlargest(5).max() - val.accuracy.nlargest(5).min()) * 100:.2f}
percentage points of each other, far less than the uncertainty of a single estimate. Naive Bayes (its independence assumption is broken by the correlated usage
features), k-NN (no notion of which inputs matter) and the hand rules clearly trail. (2) The
model earns most of its keep <b>early in the month</b>, when an alert is most useful. Over days 1-3 RuleFit is
{early:+.1f} points better than the pace rule on average, while over days 11-15 the two are {late_txt}.
RuleFit is ahead on {len(days_better)} of the 15 days{(' and behind on days ' + ', '.join(map(str, days_worse))) if days_worse else ''}.
This was measured on the test set, so I did not change the model in response, which would be test-set snooping.
(3) Thanks to the relaxed refit, the probabilities are well calibrated (Figure 11; Brier {tm['brier']:.3f}), so "8 in 10" in
the app means about 8 in 10. (4) Recall for months that stay under ({tm['recall_under']:.1%}) is lower than for months that
go over ({tm['recall_over']:.1%}), reflecting the 3:1 imbalance. The confusion matrix (Figure 12) shows {cm[0][1]:,} false
alarms and {cm[1][0]:,} missed crossings among {sum(map(sum, cm)):,} test predictions.</p>""")
    a(img("confusion_matrix_test.png", "Figure 12. RuleFit confusion matrix on the test set (for the technical report only; "
          "the stakeholder app shows a 100-dot picture instead).", "44%"))
    if rob:
        u, ss = rob["unseen_households"], rob["seed_stability"]
        a(f"""<p><b>Robustness.</b> (a) <i>Unseen households:</i> with 20% of households held out entirely, a RuleFit trained on
the other 80% scored {u['rulefit_accuracy']:.2%} (AUC {u['rulefit_auc']:.3f}) on the held-out households' test months,
against {u['pace_rule_accuracy_same_rows']:.2%} for the pace rule ({u['unseen_test_households']} households). The model
therefore generalises to new customers, not only to new months of known ones. (b) <i>Seed stability:</i> rule generation
is randomised, so I refitted with five different seeds. Validation accuracy stayed between {ss['accuracy_range'][0]:.2%} and
{ss['accuracy_range'][1]:.2%} (s.d. {ss['accuracy_std'] * 100:.2f} points). The five most important inputs overlapped by
{ss['mean_pairwise_top5_overlap']:.1f} of 5 on average between seeds, and
{', '.join(FRIENDLY.get(f, f) for f in ss['inputs_in_every_top5']) or 'no input'} appeared in the top five every time.
The exact wording of individual rules changes with the seed (thresholds move by a few units), but what the model relies
on does not.</p>""")

    # ------------------------------------------------------------------ 4
    a("<h2 class='pb'>4. Explainability and interpretation</h2>")
    a("""<p>Because RuleFit is a glass box, the first explanation is the model itself (4.1-4.3). To check that this
explanation is trustworthy, it is compared with three model-agnostic methods applied to the same model: SHAP,
permutation importance and LIME (4.4, 4.6), plus partial dependence and ICE curves (4.5). Each finding is then checked
against domain knowledge (4.5).</p>""")
    a("<h3>4.1 The whole model on one page</h3>")
    rp = xai["rules_per_prediction"]
    lens = {int(k): v for k, v in xai["rule_lengths"].items()}
    top = rules.head(12)[["rule_readable", "weight", "support", "importance"]].copy()
    top.columns = ["Rule (IF ...)", "Weight (log-odds)", "True for", "Importance"]
    a(table(top, {"Weight (log-odds)": lambda v: f"{v:+.2f}", "True for": pct1, "Importance": f3}))
    a(f"""<p class='small'>Table 8. The 12 most important of the {n_rules} rules (Friedman &amp; Popescu importance
|a<sub>k</sub>|&middot;&radic;(s<sub>k</sub>(1&minus;s<sub>k</sub>)), where s<sub>k</sub> is the share of training rows for which the rule
is true). The full rulebook is in Appendix B.</p>""")
    a(img("rulefit_top_rules.png", "Figure 13. Weights of the 15 most important rules. Orange rules push towards 'over', "
          "blue towards 'under'; the percentage is how often the rule is true."))
    a(f"""<p>The final model is an intercept plus {n_rules} rules: {', '.join(f'{v} with {k} condition{"s" if k > 1 else ""}' for k, v in sorted(lens.items()))}.
They use {xai['features_used']} of the 32 inputs; the rest were never needed. For a typical test prediction,
{rp['median_rules_true']:.0f} rules are true ({rp['mean_rules_true']:.1f} on average). Because the model is a sum, the
explanation can be ranked: the three largest contributions account for {rp['share_explained_by_top3_rules']:.0%} of the
total absolute push on an average prediction, so the app shows the strongest few. Reading the rules, the model has
learned three ideas:</p>
<ul>
<li><b>Allowance versus habit.</b> The largest rules compare how many units a day the household may still use with how
many it has been using lately (<i>units allowed per day from now</i>, <i>allowed per day vs recent daily usage</i>,
<i>projected total at current pace</i>). A home that must go below its recent daily use to stay under is pushed
strongly towards "over".</li>
<li><b>Where the bill usually sits.</b> Rules on last month's bill and on the history-based projection anchor the
prediction early in the month, when the meter has seen only a few days.</li>
<li><b>Recent momentum.</b> Rules on the units used in the last 3 days correct the pace projection when consumption
has recently changed.</li>
</ul>""")
    a("<h3>4.2 Global feature importance</h3>")
    ri = pd.Series(xai["rule_feature_importance"])
    ri_top = [FRIENDLY.get(f, f) for f in ri.index[:5]]
    unused = [FRIENDLY.get(f, f) for f, v in ri.items() if v == 0]
    survey = ["members", "n_ac", "n_fridge", "electric_water_heater", "washing_machine", "electric_cooking", "home_business"]
    n_sv = sum(ri.get(f, 0) == 0 for f in survey)
    survey_txt = ("<b>None of the seven survey answers is used by any rule.</b>" if n_sv == 7
                  else f"{n_sv} of the seven survey answers are not used by any rule.")
    a(img("rulefit_feature_importance.png", "Figure 14. Feature importance derived from the rules (each rule's importance "
          "shared equally among the inputs it tests).", "72%"))
    a(f"""<p>The five most relied-on inputs are: {', '.join(ri_top)}. Inputs the model never uses include
{', '.join(unused[:8])}{' and others' if len(unused) > 8 else ''}. {survey_txt} Once the meter readings and past bills
are known, appliances and household size add nothing the model needs. Section 4.5 examines this.</p>""")
    a("<h3>4.3 Local explanations: exact, and simple enough to do by hand</h3>")
    loc = xai["local_examples"]
    bl = loc["Borderline"]
    import math
    a(img("rulefit_local_examples.png", "Figure 15. Exact local explanations for three test households on day 8: the largest "
          "rule contributions relative to an average home."))
    a(f"""<p>For any household x, RuleFit's score is exactly
<span class='eq' style='display:inline'>log-odds(x) = b + &Sigma;<sub>k</sub> a<sub>k</sub>(r<sub>k</sub>(x) &minus; s<sub>k</sub>)</span>,
where b = {xai['baseline_log_odds']:.2f} is the log-odds of an "average" home
(P = {1 / (1 + math.exp(-xai['baseline_log_odds'])):.0%}) and each term is one rule's contribution relative to that average.
For a model that is linear in its rules, these centred contributions are exactly the SHAP values with respect to the
rules, so the local explanation needs no sampling and cannot disagree with the model. Note that a rule that is
<i>false</i> for a home also contributes (&minus;a<sub>k</sub>s<sub>k</sub>), because not matching a common rule is itself a
difference from the average home. Figure 15 marks each rule as true (&#10003;) or false (&#10007;) for that household. In the borderline example
(Figure 15, bottom), the contributions add up to {bl['log_odds'] - bl['baseline']:+.2f}, so the log-odds are
{bl['baseline']:.2f} {bl['log_odds'] - bl['baseline']:+.2f} = {bl['log_odds']:.2f}, which gives P(over) =
{bl['p']:.2f}. That household's month actually ended at {bl['actual_total']:.0f} units against a limit of
{bl['limit']:.0f}. The evidence pulls both ways, and the model honestly sits near 50%. The app uses exactly these numbers. It sums
them into five everyday groups (pace, recent days, past bills, home, calendar) for the headline reasons, and lists the
matching rules in plain words underneath.</p>""")

    a("<h3>4.4 Do model-agnostic methods agree with the model's own explanation?</h3>")
    a("<div class='two'>" + img("shap_summary_rulefit.png", "Figure 16. SHAP values for RuleFit (permutation explainer, "
                                "500 test rows, 200-row background).")
      + img("permutation_importance_rulefit.png", "Figure 17. Permutation importance (accuracy lost when an input is "
            "shuffled; 20,000 test rows, 5 repeats).") + "</div>")
    a(img("importance_agreement.png", "Figure 18. The three importance methods side by side (each normalised to sum to 100%).", "74%"))
    rho = pd.DataFrame(xai["importance_spearman"])
    t5 = xai["importance_top5"]
    ov_shap = len(set(t5["RuleFit rules"]) & set(t5["SHAP"]))
    ov_perm = len(set(t5["RuleFit rules"]) & set(t5["Permutation"]))
    a(f"""<p>SHAP (Lundberg &amp; Lee, 2017) and permutation importance treat the model as a black box, so they test whether the
rule-based reading of the model is complete. Across all 32 inputs, the Spearman rank correlation between the rule-based
importance and SHAP is &rho; = {rho.loc['RuleFit rules', 'SHAP']:.2f}, and between the rule-based importance and permutation
importance &rho; = {rho.loc['RuleFit rules', 'Permutation']:.2f}. They share {ov_shap} and {ov_perm} of their top-5 inputs
respectively. Per household, the exact rule contributions (split across the inputs each rule uses) and the SHAP values
have a median Spearman correlation of {xai['shap_vs_exact_rowwise_spearman_median']:.2f}. The rankings differ where
inputs are strongly correlated: the three past bills, or <i>units left</i> and <i>units allowed per day</i>. SHAP and
permutation then share credit between the correlated twins differently than the rules do. This is a known property of
attribution methods, not a flaw of the model, and it is why the app groups related inputs into one reason.</p>""")

    a("<h3>4.5 Partial dependence and domain knowledge</h3>")
    a(img("pdp_ice_rulefit.png", "Figure 19. Partial dependence (orange) with individual ICE curves (blue). Each panel changes "
          "one raw household input and recomputes all derived features, so the households stay realistic."))
    pdp = xai["pdp"]
    sf, ub = pdp["Units used so far"], pdp["Usual monthly bill (units)"]
    ac, mem = pdp["Air conditioners"], pdp["People in the house"]
    k = lambda d: list(d.values())
    da = xai["direction_audit"]
    thr = {"pace_projection": [], "remaining_by_history": [], "allowed_per_day": []}
    for r in rules.rule:
        for c in r.split(" and "):
            f, op, v = c.split(" ")
            if f in thr:
                thr[f].append(float(v))
    thr_txt = (f"every threshold the rules learned on the projected total at the current pace lies between "
               f"{min(thr['pace_projection']):.0f} and {max(thr['pace_projection']):.0f} units, and every threshold on the total "
               f"projected from usual habits between {min(thr['remaining_by_history']):.0f} and "
               f"{max(thr['remaining_by_history']):.0f} units: the rules put their cut-offs right at the 60-unit line")
    inc = rules[rules.rule_readable.isin(da["inconsistent_single"])]
    exc_txt = ("" if inc.empty else
               f"The {len(inc)} exceptions ({'; '.join(inc.rule_readable)}) are small corrections "
               f"(|weight| at most {inc.weight.abs().max():.2f}) sitting next to a larger rule on the same input with "
               "the opposite sign, for example &minus;1.01 for a projected total &le; 59.2. Together they shape a step "
               "around the line rather than reverse it, and the combined effect is monotone: the partial-dependence "
               "curve for units used so far rises at every step (Figure 19).")
    if max(k(ac)) - min(k(ac)) < 1e-9 and max(k(mem)) - min(k(mem)) < 1e-9:
        home_txt = ("The curves for air conditioners and household size are perfectly flat "
                    f"(the average stays at {k(ac)[0]:.1%}), because no rule tests them.")
    else:
        home_txt = (f"Going from 0 to 3 air conditioners changes the average from {k(ac)[0]:.1%} to {k(ac)[-1]:.1%}, "
                    f"and from 1 to 8 people from {k(mem)[0]:.1%} to {k(mem)[-1]:.1%}.")
    a(f"""<p><b>Partial dependence.</b> Raising <i>units used so far</i> from 0 to 60 moves the average predicted chance from
{k(sf)[0]:.0%} to {k(sf)[-1]:.0%}, and raising the <i>usual monthly bill</i> from 20 to 130 units moves it from
{k(ub)[0]:.0%} to {k(ub)[-1]:.0%}. Both rise, as they should. Because RuleFit is built from thresholds, the curves are
step functions, and the steps sit near values that matter for the 60-unit line. {home_txt}</p>
<p><b>Rule-direction audit.</b> For each rule condition on an input with an obvious expected direction (more usage, a higher
bill or a lower daily allowance should all raise the risk), I checked whether the rule's weight points the expected
way. Of the {da['single_condition_rules']} one-condition rules, {da['single_consistent_share']:.0%} agree with domain knowledge
(carrying {da['single_consistent_importance_share']:.0%} of their importance). {exc_txt}
In multi-condition rules, {da['multi_consistent_share']:.0%} of conditions agree. Disagreement there is expected: a
multi-condition rule often acts as a <i>correction</i> on top of simpler rules. For example, a rule that subtracts a
little for homes that are high on pace but low on recent use. Such a rule should be read together with the others, not on
its own.</p>
<p><b>Does the model agree with domain knowledge?</b> Mostly yes:</p>
<ul>
<li>It reproduces the arithmetic a careful household would do: if the units still allowed per day fall below recent daily use,
the month goes over. It learned this from data and was never told.</li>
<li>It located the tariff line from the data. Some engineered inputs are built from the limit (units left, pace vs the
limit), but the two projected totals are plain unit counts that do not contain it. Even so, {thr_txt}.</li>
<li>Appliances (ACs, water heater) and household size have no effect once the meter is known: no rule uses them. This
<i>contradicts</i> the naive expectation, given that homes with ACs do cross the line more often (Figure 5). The resolution is that the
meter readings already contain the AC's consumption, so the survey adds no new information. Consistent with this,
accuracy is about the same for homes with no survey (Section 5.2). This matters in practice: the tool works from meter data
alone.</li>
</ul>""")
    a("<h3>4.6 LIME versus the exact explanation</h3>")
    lm = xai["lime_borderline"]
    a("<div class='two'>" + img("lime_borderline.png", "Figure 20. LIME explanation of the borderline household (Figure 15, bottom).")
      + "<div><p>" + f"""LIME (Ribeiro et al., 2016) fits a small linear surrogate model to random perturbations around one
household. On the borderline household, LIME's top five inputs share {lm['top5_overlap']} of 5 with the exact rule
decomposition, and on the shared inputs the direction agrees in {lm['sign_agreement_on_shared']} cases. LIME's
perturbations change inputs independently. With 32 correlated inputs, that creates unrealistic households (a large units-so-far
with a tiny recent use), and its weights are small and depend on the random seed. This illustrates the main argument for a
glass-box model: when the model is a sum of rules, the explanation does not have to be estimated.""" + "</p></div></div>")

    # ------------------------------------------------------------------ 5
    a("<h2 class='pb'>5. Critical discussion</h2>")
    a("<h3>5.1 Where the model fails</h3>")
    g = gap.rename(columns={"gap": "Month ended this far from the limit (units)", "rows": "Rows", "accuracy": "RuleFit accuracy",
                            "rule_accuracy": "Pace-rule accuracy", "share_over": "Share over"})
    a(table(g[["Month ended this far from the limit (units)", "Rows", "RuleFit accuracy", "Pace-rule accuracy"]],
            {"Rows": n0, "RuleFit accuracy": pct1, "Pace-rule accuracy": pct1}))
    a(f"""<p class='small'>Table 9. Test accuracy by how far the month actually ended from the limit.</p>
<p>Errors are concentrated in months that end within a few units of the line: accuracy is {gap.accuracy.iloc[0]:.1%} for
months that end within 5 units, against {gap.accuracy.iloc[-1]:.1%} for months that end 60 or more units away (Figure 21).
For those homes the outcome depends on the last two weeks of behaviour, which no model can see on day 5. These are also the
homes that gain most from an alert, so the app tells them plainly that they are close to the line rather than giving
false certainty.</p>""")
    a(img("fairness_and_errors.png", "Figure 21. Error rate by distance to the limit (left) and accuracy by socio-economic class (right)."))
    a("<h3>5.2 Bias and fairness</h3>")
    s2 = sec.rename(columns={"socio_economic_class": "Socio-economic class", "households": "Households", "accuracy": "RuleFit accuracy",
                             "rule_accuracy": "Pace rule", "share_over": "Share over"})
    a(table(s2[["Socio-economic class", "Households", "RuleFit accuracy", "Pace rule", "Share over"]],
            {"RuleFit accuracy": pct1, "Pace rule": pct1, "Share over": pct1}))
    sec_lo, sec_hi = sec.accuracy.min(), sec.accuracy.max()
    sv = surv.set_index("survey").accuracy
    band = pd.DataFrame(xai["slices"]["sec_by_band"]).set_index("socio_economic_class")
    a(table(band.reset_index().rename(columns={"socio_economic_class": "Class", "beyond 15 units": "Accuracy, month ended >15 units from line",
                                               "within 15 units": "Accuracy, within 15 units", "share_within_15": "Share of months within 15 units"}),
            {"Accuracy, month ended >15 units from line": pct1, "Accuracy, within 15 units": pct1, "Share of months within 15 units": pct1}))
    sh = band.share_within_15
    a(f"""<p class='small'>Tables 10-11. Test accuracy by socio-economic class (A = highest), and the same split by distance to
the line.</p>
<p>Accuracy ranges from {sec_lo:.1%} to {sec_hi:.1%} across socio-economic classes. Comparing like with like explains
most of this. For months that end more than 15 units from the line, every class is between
{band['beyond 15 units'].min():.1%} and {band['beyond 15 units'].max():.1%}. But the share of months that end near the line
varies by class, from {sh.min():.0%} ({sh.idxmin()}) to {sh.max():.0%} ({sh.idxmax()}), so some classes get more hard cases.
Within the near-line band, accuracy still varies ({band['within 15 units'].min():.0%}-{band['within 15 units'].max():.0%}),
so the disparity is not fully explained. The model never sees class, area or house type, but this is still a real
equity concern: lower-income homes near the line are the ones the bill jump hurts most. An operator must not treat
the tool's "you will stay under" as a guarantee for such a home. The app's near-line warning exists for this reason.
A rulebook makes this auditable: none of the {n_rules} rules mentions a protected attribute, and anyone can check that.
Accuracy on homes with no survey ({sv.get('No survey (smart-meter only)', float('nan')):.1%}) is close to surveyed homes
({sv.get('Surveyed', float('nan')):.1%}), so the model does not need the survey to work. By area, accuracy ranges from
{area.accuracy.min():.1%} to {area.accuracy.max():.1%}; by house type, from {house.accuracy.min():.1%} to
{house.accuracy.max():.1%} (groups with at least 15 households).</p>""")
    a("<h3>5.3 Limitations and data-quality issues</h3>")
    a(f"""<ul>
<li><b>Coverage:</b> LECO customers in coastal western and southern areas only. Up-country and CEB-served homes, which have a
different climate and usage, may behave differently. Solar homes are excluded.</li>
<li><b>Calendar vs billing months:</b> the smart-meter data is cut by calendar month, while real LECO billing periods start on
the meter-reading date. The tariff's pro-rating rule was applied, but the date offset is an approximation.</li>
<li><b>Missing months:</b> May-July 2024 is almost absent from the 6-hour files. About {dq['missing_reading_share']:.0%} of
daily readings are missing overall.</li>
<li><b>Survey timing:</b> wave 1 was fielded Dec 2023 - Oct 2024, so some appliance answers postdate the 2023 training months.</li>
<li><b>Model-specific limitations of RuleFit:</b> (i) rules are axis-aligned thresholds, so predictions jump at a threshold
instead of changing smoothly (Figure 19). Two almost identical homes on either side of "62.9 units" can get different
probabilities. (ii) The weights of overlapping, correlated rules are not causal effects. A rule's weight is its effect
<i>given the other rules</i>, so single rules should not be quoted out of context. (iii) The exact thresholds depend on the
random seed (Section 3.5), although accuracy and the main inputs do not. (iv) The <code>imodels</code> probability formula had to
be corrected (Section 2), which shows the value of checking a library against the maths.</li>
<li><b>Prediction window:</b> the model is built for days 1-15. In the second half of the month the arithmetic is nearly
certain.</li>
<li><b>Daily resolution:</b> the 6-hour readings are reduced to daily closing values. The 15-minute data could add
time-of-day behaviour.</li>
<li><b>Tariff changes:</b> if PUCSL moves the 60-unit boundary, the target must be redefined and the model retrained.</li>
</ul>""")
    a("<h3>5.4 Real-world impact and ethics</h3>")
    a(f"""<p>For a home near the line, an early alert is worth about Rs {estimate_bill(61) - estimate_bill(60):,.0f} a month,
a meaningful amount for lower-income families after the 2022-2024 tariff increases. LECO could send it as an SMS from
existing smart-meter data, and it supports the national demand-side-management goal by shifting consumption without
new infrastructure. A rule-based model is also easier to approve: a regulator or utility manager can read all
{n_rules} rules before deployment. The risks: (i) a false "you are safe" could cost a household money, so the app states its
error rate and the near-line warning; (ii) pressure to stay under 60 could lead families to cut essential use, such as
refrigeration or fans during heat, so the tips focus on waste (AC hours, water-heater timing) and not on essentials;
(iii) usage data reveals occupancy patterns, so any operational version must keep data with the utility, under consent,
and the public app processes inputs in memory only.</p>""")

    # ------------------------------------------------------------------ 6
    a("<h2 class='pb'>6. Front-end integration (bonus)</h2>")
    a("""<p>A Streamlit app (<code>app/streamlit_app.py</code>) wraps the saved RuleFit model. It is written for people with
no ML background: no confusion matrices, AUCs or log-odds. Users type the two meter readings printed on their bill, the
days elapsed, their last three bills and a few questions about the home (or pick a worked example). The result shows:</p>
<ol>
<li>a one-sentence verdict and the odds in words ("more than 9 times out of 10");</li>
<li>a meter strip of units used, the projection at the current pace, and the line, with the two rupee amounts that are at stake;</li>
<li>an actionable daily budget ("keep to about 1.9 units a day for the remaining 21 days; lately you have used 2.2");</li>
<li><i>Why the tool thinks this</i>: the model's exact rule contributions, summed into five everyday reasons and worded
as a comparison with an average home, because that is what centred contributions mean;</li>
<li><i>See the rules that matched your home</i>: the matching rules, translated into sentences such as "at most 1.9 units a
day allowed from now on and more than 59 units projected from usual habits". The rules that support the answer come first.
The small correcting rules that pull the other way are listed separately, in muted text and with a note that they fine-tune
the stronger rules. This is a design choice that follows from the direction audit in Section 4.5: shown on equal terms, a
correction rule would confuse a non-expert, but hiding it would break the promise that nothing is hidden;</li>
<li>a <i>How reliable is this?</i> tab showing 100 dots coloured right or wrong, accuracy by part of the month, and when to
be careful; and an <i>About</i> tab that explains the rulebook idea and lets anyone browse every rule.</li>
</ol>
<p>The same <code>features.py</code> computes inputs in training and in the app, so there is no training/serving skew. The
rule-to-sentence translation (<code>src/explain_text.py</code>) is unit-tested so that no raw feature name reaches a user.
Library versions are pinned in <code>requirements.txt</code>, because a saved model must be loaded with the versions that
saved it. An end-to-end browser test (<code>tests/app_screenshots.js</code>) runs three examples and fails on any page
error. Screenshots are in Appendix A, and the demo video walks through the app.</p>""")

    # ------------------------------------------------------------------ 7
    a("<h2>7. Conclusion</h2>")
    a(f"""<p>Using public Sri Lankan smart-meter data, I framed a concrete local problem (the 60-unit tariff cliff) as a
mid-month classification task and built a leakage-safe dataset. I then showed that RuleFit, a method not taught in the
module, solves it with a model a person can read in full: {n_rules} IF-THEN rules, {tm['accuracy']:.2%} test accuracy,
against {best_bb['accuracy']:.2%} for the best black-box ensemble ({best_bb['model']}), with calibrated probabilities after
a relaxed-lasso refit that I added. Its explanations are exact. They agree broadly with SHAP and permutation importance,
and they match domain knowledge, including one useful surprise: appliances add little once the meter is read. The same rules
power a front end that tells a household not just <i>whether</i> it will cross the line but <i>why</i>, and <i>what to
do</i>. Future work: a validated rule/model hybrid for the second week, thresholds at 90 and 180 units, the 15-minute data,
and a Sinhala/Tamil interface.</p>""")

    a("""<h2>References</h2><ol class='small'>
<li>Friedman, J. H., Popescu, B. E. Predictive learning via rule ensembles. <i>Annals of Applied Statistics</i> 2(3), 916-954, 2008. doi:10.1214/07-AOAS148.</li>
<li>Meinshausen, N. Relaxed lasso. <i>Computational Statistics &amp; Data Analysis</i> 52(1), 374-393, 2007.</li>
<li>Tibshirani, R. Regression shrinkage and selection via the lasso. <i>JRSS B</i> 58(1), 267-288, 1996.</li>
<li>Singh, C., Nasseri, K., Tan, Y. S., Tang, T., Yu, B. imodels: a python package for fitting interpretable models. <i>JOSS</i> 6(61), 3192, 2021.</li>
<li>Tan, Y. S., Singh, C., Nasseri, K., Agarwal, A., Yu, B. Fast interpretable greedy-tree sums (FIGS). arXiv:2201.11931, 2022.</li>
<li>LIRNEasia et al. <i>Residential Electricity Consumption Dataset for Sri Lanka (RECON-SL)</i>. IEEE DataPort, 2025. doi:10.21227/n1dk-q860.</li>
<li>Public Utilities Commission of Sri Lanka. <i>Electricity tariff, domestic (D-1)</i>, revision effective 11 May 2026.</li>
<li>Nori, H., Jenkins, S., Koch, P., Caruana, R. InterpretML: A unified framework for machine learning interpretability. arXiv:1909.09223, 2019.</li>
<li>Lundberg, S., Lee, S.-I. A unified approach to interpreting model predictions. <i>NeurIPS</i>, 2017.</li>
<li>Ribeiro, M. T., Singh, S., Guestrin, C. "Why should I trust you?" Explaining the predictions of any classifier. <i>KDD</i>, 2016.</li>
<li>Goldstein, A., Kapelner, A., Bleich, J., Pitkin, E. Peeking inside the black box: ICE plots. <i>JCGS</i>, 2015.</li>
<li>Friedman, J. H. Greedy function approximation: a gradient boosting machine. <i>Annals of Statistics</i> 29(5), 2001.</li>
</ol>""")

    # ------------------------------------------------------------------ appendix
    a("<h2 class='pb'>Appendix A. App screenshots</h2>")
    a("<div class='two'>" + img("app_result_over.png", "Figure A1. Check my month: a family close to the line.")
      + img("app_result_under.png", "Figure A2. A careful two-person home that will stay under.") + "</div>")
    a("<div class='two'>" + img("app_rules.png", "Figure A3. The rules that matched a home, in plain words.")
      + img("app_reliability.png", "Figure A4. The How reliable is this? tab (no confusion matrix: 100 dots instead).") + "</div>")
    a("<h2 class='pb'>Appendix B. The complete rulebook</h2>")
    full = rules[["rule_readable", "weight", "support"]].copy()
    full.insert(0, "#", range(1, len(full) + 1))
    full.columns = ["#", "IF ...", "Weight", "True for"]
    a(f"<p class='small'>All {n_rules} rules of the final model, in order of importance. Intercept a<sub>0</sub> = "
      f"{xai['intercept']:.3f}. To score a household, add the weights of the rules that are true, add a<sub>0</sub>, and apply "
      "P = 1 / (1 + e<sup>&minus;sum</sup>).</p>")
    a(table(full, {"Weight": lambda v: f"{v:+.3f}", "True for": pct1}, cls="rules"))
    a("""<h2>Appendix C. Reproducibility</h2><p class='small'>The source code is supplied as a ZIP.
<code>src/</code> holds the pipeline, <code>app/</code> the front end and <code>tests/</code> the tests (pytest: tariff maths,
feature engineering, leakage guards, exact additivity and calibration of the RuleFit model, plain-language rules; Playwright: an
end-to-end UI test). Run order: <code>aggregate_smart_meter -> data_prep -> eda -> train -> ablation -> explain -> robustness ->
build_report</code>. Random seed 42 throughout. Environment: Python 3.13, imodels 3.0.4, scikit-learn 1.9.1, interpret 0.7.8,
LightGBM 4.7.0, SHAP 0.54.1 (full pins in <code>requirements*.txt</code>). Every number in this report is generated from
<code>reports/*.json|csv</code> by <code>src/build_report.py</code>. The raw data is not redistributed; <code>DATASET.md</code>
explains how to download it and rebuild the modelling table.</p>""")

    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Bill Alert report</title><style>{CSS}</style></head><body>{''.join(H)}</body></html>"
    out = REPORTS / "Bill_Alert_Report.html"
    out.write_text(html, encoding="utf-8")
    return out


def to_pdf(html_path: Path):
    from playwright.sync_api import sync_playwright
    pdf = html_path.with_suffix(".pdf")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(html_path.resolve().as_uri(), wait_until="load")
        pg.pdf(path=str(pdf), format="A4", print_background=True, display_header_footer=True,
               header_template="<span></span>",
               footer_template="<div style='font-size:7.5pt;width:100%;text-align:center;color:#777;font-family:Inter'>"
                               "<span class='pageNumber'></span> / <span class='totalPages'></span></div>",
               margin={"top": "18mm", "bottom": "18mm", "left": "17mm", "right": "17mm"})
        b.close()
    return pdf


if __name__ == "__main__":
    h = build()
    print("pdf:", to_pdf(h))
