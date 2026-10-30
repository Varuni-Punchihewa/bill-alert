"""Step 3 - evaluation figures, explainability (XAI) and fairness slices for RuleFit.

Methods
  * Intrinsic (glass-box) - RuleFit's own rule list: weights, support, global rule
    importance (Friedman & Popescu), feature importance derived from rules,
    exact local explanations (baseline + rule contributions = log-odds)
  * SHAP (permutation explainer, model-agnostic) on RuleFit itself
  * Permutation feature importance
  * Partial dependence + ICE curves (on raw household inputs, derived features recomputed)
  * LIME for a borderline household, compared with the exact explanation
  * Domain-knowledge checks: rule-direction audit + PDP directions
  * Fairness / error analysis by socio-economic class, area, house type, distance to the limit

    python -m src.explain
"""
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

from src import plotstyle as ps
from src.config import DATA_PROC, FIGURES, MODELS, REPORTS, SEED, VALID_END
from src.explain_text import parse, rule_plain, rule_short
from src.features import FEATURES, friendly, make_X

warnings.filterwarnings("ignore")
ps.apply()
TARGET = "over_limit"
FINAL = "RuleFit (relaxed)"

# Direction domain knowledge expects: +1 = more of it -> more likely to go over.
EXPECTED = {"so_far": 1, "pace_projection": 1, "pace_ratio": 1, "units_left": -1, "allowed_per_day": -1,
            "recent_rate": 1, "recent3_rate": 1, "last7": 1, "last3": 1, "allowed_vs_recent": -1,
            "prev1": 1, "prev2": 1, "prev3": 1, "history_avg": 1, "history_ratio": 1, "history_max": 1,
            "remaining_by_history": 1, "n_ac": 1, "members": 1, "n_fridge": 1, "electric_water_heater": 1,
            "electric_cooking": 1, "home_business": 1, "washing_machine": 1}


def load():
    model = joblib.load(MODELS / "rulefit_bill_alert.joblib")["model"]
    cp = pd.read_parquet(DATA_PROC / "checkpoints.parquet")
    preds = pd.read_parquet(REPORTS / "test_predictions.parquet").reset_index(drop=True)
    te = preds[["hid", "ym", "day"]].merge(cp[cp.period >= VALID_END], on=["hid", "ym", "day"], how="left")
    return model, te, preds


# ------------------------------------------------------------ evaluation plots
FAMILY_COLOR = {"Baseline rule": ps.INK2, "Taught: single model": ps.BLUE, "Taught: bagging": ps.AQUA,
                "Taught: boosting": ps.YELLOW, "Taught: ensemble": ps.VIOLET, "Taught: glass-box": ps.GREEN,
                "Not taught": ps.MAGENTA, "New algorithm (final)": ps.ORANGE}


def fig_model_comparison():
    v = pd.read_csv(REPORTS / "validation_comparison.csv").sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.barh(v.model, v.accuracy * 100, color=[FAMILY_COLOR[f] for f in v.family], height=0.68)
    for i, a in enumerate(v.accuracy):
        ax.text(a * 100 + 0.05, i, f"{a:.2%}", va="center", fontsize=8, color=ps.INK)
    lo = min(80, np.floor(v.accuracy.min() * 100) - 2)
    ax.set(xlim=(lo, 100), xlabel="Validation accuracy (%)  - Jan-Jun 2024, days 1-15",
           title="Model comparison on the validation months")
    ax.legend(handles=[Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIGURES / "model_comparison_validation.png")


def fig_tradeoff():
    """Accuracy vs model size: how many rules does a household-readable model need?"""
    t = pd.read_csv(REPORTS / "tuning_results.csv")
    v = pd.read_csv(REPORTS / "validation_comparison.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for name, col, lab in [("RuleFit", ps.ORANGE, "RuleFit (number of rules)"),
                           ("FIGS", ps.MAGENTA, "FIGS (number of splits)")]:
        d = t[t.model == name].groupby("size").val_accuracy.max().sort_index()
        ax.plot(d.index, d.values * 100, marker="o", color=col, label=lab, lw=2)
    for name, col in [("LightGBM", ps.YELLOW), ("Explainable Boosting Machine", ps.GREEN),
                      ("Decision Tree", ps.BLUE)]:
        if name in v.index:
            ax.axhline(v.loc[name, "accuracy"] * 100, color=col, ls="--", lw=1.3, label=f"{name} (black-box size)"
                       if name == "LightGBM" else name)
    ax.axhline(95, color=ps.GRID, lw=1.2)
    ax.set(xlabel="Model size (rules / splits a person would have to read)", ylabel="Validation accuracy (%)",
           title="Accuracy vs readability: a few dozen rules is enough")
    ax.legend(fontsize=8, loc="lower right")
    ps.save(fig, FIGURES / "accuracy_vs_rules.png")


def fig_accuracy_by_day(preds):
    cols = {"p": ("RuleFit (final)", ps.ORANGE), "p_Explainable Boosting Machine": ("EBM", ps.GREEN),
            "p_LightGBM": ("LightGBM", ps.YELLOW), "p_Logistic Regression": ("Logistic regression", ps.BLUE),
            "p_Rule: pace projection": ("Pace rule", ps.INK2)}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    out = {}
    for c, (lab, col) in cols.items():
        acc = preds.groupby("day").apply(lambda g: accuracy_score(g[TARGET], g[c] >= .5)) * 100
        ax.plot(acc.index, acc.values, marker="o", ms=4, color=col, label=lab, lw=2.5 if c == "p" else 1.5)
        out[lab] = acc.round(2).to_dict()
    ax.axhline(95, color=ps.GRID, lw=1.5, ls="--")
    ax.set(xlabel="Day of the billing month when the alert is issued", ylabel="Test accuracy (%)",
           title="Accuracy by day of the month (test set)")
    ax.legend(loc="lower right", fontsize=8)
    ps.save(fig, FIGURES / "accuracy_by_day_test.png")
    return out


def fig_calibration(preds):
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], color=ps.GRID, lw=1.5)
    for c, lab, col in [("p", "RuleFit (final)", ps.ORANGE), ("p_FIGS (greedy tree sums)", "FIGS", ps.MAGENTA),
                        ("p_LightGBM", "LightGBM", ps.YELLOW)]:
        f, m = calibration_curve(preds[TARGET], preds[c], n_bins=10, strategy="quantile")
        ax.plot(m, f, marker="o", ms=4, color=col, label=lab)
    ax.set(xlabel="Predicted chance of going over", ylabel="Observed share that went over",
           title="Calibration (test set)")
    ax.legend(loc="upper left")
    ps.save(fig, FIGURES / "calibration_test.png")


def fig_confusion(preds):
    """Confusion matrix for the technical report only (NOT shown in the app)."""
    y, yh = preds[TARGET].to_numpy(), (preds.p >= .5).astype(int).to_numpy()
    cm = np.array([[((y == 0) & (yh == 0)).sum(), ((y == 0) & (yh == 1)).sum()],
                   [((y == 1) & (yh == 0)).sum(), ((y == 1) & (yh == 1)).sum()]])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap=plt.matplotlib.colors.ListedColormap(ps.SEQ[:5]), vmin=0, vmax=cm.max() * 1.6)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                    color=ps.INK, fontsize=10)
    ax.set_xticks([0, 1], ["Predicted under", "Predicted over"])
    ax.set_yticks([0, 1], ["Actually under", "Actually over"])
    ax.grid(False)
    ax.set_title("RuleFit confusion matrix (test)")
    ps.save(fig, FIGURES / "confusion_matrix_test.png")
    return cm.tolist()


# ------------------------------------------------------------ RuleFit glass-box
def rule_list(model):
    t = model.rule_table()
    t["rule_readable"] = t.rule.map(rule_short)
    t["rule_plain"] = t.rule.map(rule_plain)
    t["n_conditions"] = t.rule.map(lambda r: len(parse(r)))
    t[["rule_readable", "weight", "support", "importance", "n_conditions", "rule_plain", "rule"]].to_csv(
        REPORTS / "rulefit_rules_explained.csv", index=False)
    top = t.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.barh(range(len(top)), top.weight, color=[ps.ORANGE if w > 0 else ps.BLUE for w in top.weight], height=0.62)
    ax.set_yticks(range(len(top)), [f"{r}   ({s:.0%} of cases)" for r, s in zip(top.rule_readable, top.support)],
                  fontsize=8)
    ax.axvline(0, color=ps.INK2, lw=0.8)
    ax.set(xlabel="Rule weight (change in log-odds of going over when the rule is true)",
           title=f"The {len(t)} learned rules - top 15 by importance")
    ax.legend(handles=[Patch(color=ps.ORANGE, label="Rule pushes towards 'over'"),
                       Patch(color=ps.BLUE, label="Rule pushes towards 'under'")], loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIGURES / "rulefit_top_rules.png")
    return t


def rule_feature_importance(model):
    imp = model.feature_importance()
    top = imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.barh([friendly(f) for f in top.index], top.values * 100, color=ps.ORANGE, height=0.62)
    ax.set(xlabel="Share of total rule importance (%)", title="Which inputs RuleFit's rules rely on")
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIGURES / "rulefit_feature_importance.png")
    return imp


def rules_per_prediction(model, te):
    X = make_X(te)
    R = model._rules_true(model._impute(X))
    C = np.abs(model.contributions(X))
    active = R.sum(1)
    big = (C >= 0.1).sum(1)
    return {"mean_rules_true": float(active.mean()), "median_rules_true": float(np.median(active)),
            "mean_rules_with_contribution_ge_0.1": float(big.mean()),
            "share_explained_by_top3_rules": float(np.mean(np.sort(C, 1)[:, -3:].sum(1) / C.sum(1).clip(1e-9)))}


def local_examples(model, te, preds):
    X = make_X(te)
    p = preds.p.to_numpy()
    d8 = (te.day == 8).to_numpy()
    cases = {"Clearly over": int(np.argmax(np.where(d8, p, -1))),
             # lowest-scoring day-8 home that did stay under (the very lowest ended 2 units over: a near-line miss)
             "Clearly under": int(np.argmin(np.where(d8 & (te[TARGET] == 0).to_numpy(), p, 2))),
             "Borderline": int(np.argmin(np.abs(p - 0.5) + 10 * (te.day != 8)))}
    idx = list(cases.values())
    C = model.contributions(X.iloc[idx])
    z = model.decision_function(X.iloc[idx])
    assert np.allclose(model.baseline_ + C.sum(1), z)          # exact additivity
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 13))
    out = {}
    for k, (ax, (name, i)) in enumerate(zip(axes, cases.items())):
        true_k = model._rules_true(model._impute(X.iloc[[i]]))[0]
        s = pd.Series(C[k], index=[("✓ " if t else "✗ ") + rule_short(r) for r, t in zip(model.rules_, true_k)])
        s = s[s.abs() >= 1e-6]
        s = s.reindex(s.abs().sort_values(ascending=False).index)[:7][::-1]
        ax.barh(range(len(s)), s.values, color=[ps.ORANGE if v > 0 else ps.BLUE for v in s.values], height=0.6)
        import textwrap
        ax.set_yticks(range(len(s)), [textwrap.fill(t, 62) for t in s.index], fontsize=7.5)
        ax.axvline(0, color=ps.INK2, lw=0.8)
        r = te.iloc[i]
        ax.set_title(f"{name}: P(over) = {p[i]:.2f}   (day {int(r.day)}, {r.so_far:.0f} units so far, last bill "
                     f"{r.prev1:.0f}; month actually ended at {r.total:.0f} vs limit {r.limit:.0f})", fontsize=9)
        ax.grid(axis="y", visible=False)
        out[name] = {"row": i, "p": float(p[i]), "log_odds": float(z[k]), "baseline": model.baseline_,
                     "actual_total": float(r.total), "limit": float(r.limit),
                     "top_rules": s[::-1].round(3).to_dict()}
    axes[-1].set_xlabel("Contribution to log-odds vs an average home (exact)")
    fig.suptitle("Local explanations: baseline + contributions = the prediction, exactly  (✓ rule true for this home, ✗ false)",
                 x=0.01, y=1.0, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ps.save(fig, FIGURES / "rulefit_local_examples.png")
    return out, cases


# --------------------------------------------------------------- model-agnostic
def shap_rulefit(model, te):
    import shap
    X = make_X(te)
    bg = X.sample(200, random_state=SEED)
    xs = X.sample(500, random_state=SEED + 1)
    f = lambda a: model.decision_function(pd.DataFrame(a, columns=FEATURES))
    ex = shap.PermutationExplainer(f, shap.maskers.Independent(bg, max_samples=200))
    sv = ex(xs, max_evals=2 * len(FEATURES) + 1, silent=True)
    plt.figure()
    shap.summary_plot(sv.values, xs.rename(columns=friendly), max_display=15, show=False, plot_size=(8, 6))
    plt.title("SHAP values for RuleFit (permutation explainer, test sample)", loc="left", fontsize=12,
              fontweight="bold")
    plt.savefig(FIGURES / "shap_summary_rulefit.png", bbox_inches="tight", dpi=200)
    plt.close("all")
    shap_imp = pd.Series(np.abs(sv.values).mean(0), index=FEATURES).sort_values(ascending=False)
    # Per-row agreement between SHAP and the exact rule contributions split across features
    F = model.feature_contributions(xs).to_numpy()
    row_rho = np.nanmedian([spearmanr(F[i], sv.values[i]).correlation for i in range(len(xs))])
    return shap_imp, float(row_rho)


def perm_importance(model, te):
    s = te.sample(min(20000, len(te)), random_state=SEED)
    r = permutation_importance(model, make_X(s), s[TARGET], scoring="accuracy", n_repeats=5,
                               random_state=SEED, n_jobs=1)
    imp = pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh([friendly(t) for t in top.index], top.values * 100, color=ps.AQUA, height=0.6)
    ax.set(xlabel="Accuracy lost when the input is shuffled (percentage points)",
           title="Permutation importance (RuleFit, test sample)")
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIGURES / "permutation_importance_rulefit.png")
    return imp


def importance_agreement(rule_imp, shap_imp, perm_imp):
    d = pd.DataFrame({"RuleFit rules": rule_imp, "SHAP": shap_imp, "Permutation": perm_imp.clip(lower=0)}).fillna(0)
    d = d / d.sum()
    top = d.sort_values("RuleFit rules", ascending=False).head(10)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    h = 0.26
    for j, (c, col) in enumerate([("RuleFit rules", ps.ORANGE), ("SHAP", ps.VIOLET), ("Permutation", ps.AQUA)]):
        ax.barh(np.arange(len(top)) + (j - 1) * h, top[c] * 100, height=h, color=col, label=c)
    ax.set_yticks(range(len(top)), [friendly(f) for f in top.index], fontsize=8.5)
    ax.set(xlabel="Share of importance (%)", title="Three importance methods agree on what matters")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIGURES / "importance_agreement.png")
    rho = d.corr(method="spearman").round(3)
    top5 = {c: list(d[c].sort_values(ascending=False).index[:5]) for c in d}
    return rho.to_dict(), top5


def pdp_ice(model, te):
    """Vary one raw household input, recompute the derived features, average the predictions."""
    s = te.sample(min(1500, len(te)), random_state=SEED).reset_index(drop=True)
    specs = {
        "Units used so far": ("so_far", np.arange(0, 61, 5), s),
        "Usual monthly bill (units)": ("prev*", np.arange(20, 131, 10), s),
        "Air conditioners": ("n_ac", np.arange(0, 4), s.dropna(subset=["members"])),
        "People in the house": ("members", np.arange(1, 9), s.dropna(subset=["members"])),
    }
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.5), sharey=True)
    out = {}
    for ax, (lab, (f, grid, base)) in zip(axes, specs.items()):
        base = base.sample(min(len(base), 400), random_state=SEED)
        curves = []
        for v in grid:
            b = base.copy()
            if f == "prev*":
                b[["prev1", "prev2", "prev3"]] = v
            else:
                b[f] = v
            curves.append(model.predict_proba(make_X(b))[:, 1])
        C = np.array(curves)
        for j in range(0, C.shape[1], 8):
            ax.plot(grid, C[:, j], color=ps.SEQ[1], lw=0.6, alpha=0.55)
        ax.plot(grid, C.mean(1), color=ps.ORANGE, lw=2.5, marker="o", ms=4)
        ax.set(title=lab, xlabel="value")
        out[lab] = dict(zip([int(g) for g in grid], C.mean(1).round(4).tolist()))
    axes[0].set_ylabel("Predicted chance of going over")
    fig.suptitle("Partial dependence (orange) with individual ICE curves (blue)", x=0.01, y=1.05, ha="left",
                 fontsize=12, fontweight="bold")
    ps.save(fig, FIGURES / "pdp_ice_rulefit.png")
    return out


def lime_case(model, te, row):
    from lime.lime_tabular import LimeTabularExplainer
    X = make_X(te)
    bg = X.sample(min(5000, len(X)), random_state=SEED)
    med = bg.median()
    ex = LimeTabularExplainer(bg.fillna(med).to_numpy(), feature_names=FEATURES, class_names=["under", "over"],
                              discretize_continuous=True, random_state=SEED)
    x = X.iloc[row].fillna(med).to_numpy()
    e = ex.explain_instance(x, lambda a: model.predict_proba(pd.DataFrame(a, columns=FEATURES)), num_features=8)
    lst = e.as_list()
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    lab, val = zip(*lst[::-1])
    pretty = [l for l in lab]
    for f in sorted(FEATURES, key=len, reverse=True):
        pretty = [p.replace(f, friendly(f)) if f in p and friendly(f) not in p else p for p in pretty]
    ax.barh(pretty, val, color=[ps.ORANGE if v > 0 else ps.BLUE for v in val], height=0.6)
    ax.axvline(0, color=ps.INK2, lw=0.8)
    ax.set(title="LIME explanation of the borderline household", xlabel="Local weight towards 'over'")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ps.save(fig, FIGURES / "lime_borderline.png")
    lime_feats = [next(f for f in sorted(FEATURES, key=len, reverse=True) if f in l) for l, _ in lst]
    exact = model.feature_contributions(X.iloc[[row]]).iloc[0]
    exact_top = list(exact.abs().sort_values(ascending=False).index[:5])
    lime_sign = {f: np.sign(v) for f, v in zip(lime_feats, [v for _, v in lst])}
    sign_agree = [bool(np.sign(exact[f]) == lime_sign[f]) for f in lime_feats[:5] if abs(exact[f]) > 1e-9]
    return {"lime": [(a, round(b, 4)) for a, b in lst], "lime_top5": lime_feats[:5], "exact_top5": exact_top,
            "top5_overlap": len(set(lime_feats[:5]) & set(exact_top)),
            "sign_agreement_on_shared": f"{sum(sign_agree)}/{len(sign_agree)}"}


# ------------------------------------------------------------- domain knowledge
def direction_audit(rules):
    """Does each rule push in the direction an energy officer would expect?

    For a one-condition rule, 'feature > t' should have the expected sign and
    'feature <= t' the opposite. Multi-condition rules are reported separately
    because their weights often act as corrections on top of simpler rules.
    """
    rows = []
    for _, r in rules.iterrows():
        conds = parse(r.rule)
        for f, op, v in conds:
            if f not in EXPECTED:
                continue
            expected = EXPECTED[f] * (1 if op == ">" else -1)
            rows.append({"rule": r.rule_readable, "feature": f, "single": len(conds) == 1,
                         "consistent": bool(np.sign(r.weight) == expected), "importance": r.importance})
    d = pd.DataFrame(rows)
    one = d[d.single]
    many = d[~d.single]
    return {"single_condition_rules": int(len(one)),
            "single_consistent_share": float(one.consistent.mean()) if len(one) else None,
            "single_consistent_importance_share": float(one[one.consistent].importance.sum() / one.importance.sum())
            if len(one) else None,
            "multi_condition_terms": int(len(many)),
            "multi_consistent_share": float(many.consistent.mean()) if len(many) else None,
            "inconsistent_single": one[~one.consistent].rule.tolist()}


# ------------------------------------------------------------- fairness / errors
def slices(preds):
    d = preds.copy()
    d["correct"] = ((d.p >= .5).astype(int) == d[TARGET]).astype(int)
    d["rule_correct"] = ((d["p_Rule: pace projection"] >= .5).astype(int) == d[TARGET]).astype(int)
    d["gap"] = pd.cut((d.total - d.limit).abs(), [-0.1, 5, 15, 30, 60, np.inf],
                      labels=["0-5", "5-15", "15-30", "30-60", "60+"])
    d["survey"] = np.where(d.socio_economic_class.isna(), "No survey (smart-meter only)", "Surveyed")
    out = {}
    for col in ["socio_economic_class", "type_of_house", "csc_area", "gap", "survey"]:
        t = d.groupby(col, observed=True).agg(rows=("correct", "size"), households=("hid", "nunique"),
                                              accuracy=("correct", "mean"), rule_accuracy=("rule_correct", "mean"),
                                              share_over=(TARGET, "mean"))
        if col != "gap":
            t = t[t.households >= 15]
        out[col] = t.round(4)
    d["band"] = np.where((d.total - d.limit).abs() <= 15, "within 15 units", "beyond 15 units")
    band = d.groupby(["socio_economic_class", "band"]).correct.mean().unstack()
    band["share_within_15"] = d.groupby("socio_economic_class").band.apply(lambda s: (s == "within 15 units").mean())
    out["sec_by_band"] = band.round(4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    g = out["gap"]
    axes[0].bar(g.index.astype(str), (1 - g.accuracy) * 100, color=ps.ORANGE, width=0.6, label="RuleFit")
    axes[0].plot(g.index.astype(str), (1 - g.rule_accuracy) * 100, color=ps.INK2, marker="o", label="Pace rule")
    for i, v in enumerate(1 - g.accuracy):
        axes[0].text(i, v * 100 + 0.4, f"{v:.1%}", ha="center", fontsize=8.5)
    axes[0].set(xlabel="How far the month's total ended from the limit (units)", ylabel="Error rate (%)",
                title="Errors are concentrated right at the line")
    axes[0].legend()
    s = out["socio_economic_class"]
    axes[1].bar(s.index, s.accuracy * 100, color=ps.BLUE, width=0.6)
    for i, v in enumerate(s.accuracy):
        axes[1].text(i, v * 100 + 0.2, f"{v:.1%}", ha="center", fontsize=8.5)
    axes[1].set(ylim=(85, 100), ylabel="Accuracy (%)", title="Accuracy by socio-economic class (A = highest)")
    ps.save(fig, FIGURES / "fairness_and_errors.png")
    return {k: v.reset_index().to_dict(orient="records") for k, v in out.items()}


def main():
    model, te, preds = load()
    assert len(te) == len(preds)
    assert np.allclose(model.predict_proba(make_X(te))[:, 1], preds.p)    # the saved model made these predictions
    res = {}
    fig_model_comparison()
    fig_tradeoff()
    res["accuracy_by_day"] = fig_accuracy_by_day(preds)
    fig_calibration(preds)
    res["confusion_matrix"] = fig_confusion(preds)
    rules = rule_list(model)
    res["n_rules"] = len(rules)
    res["rule_lengths"] = rules.n_conditions.value_counts().sort_index().to_dict()
    res["baseline_log_odds"] = model.baseline_
    res["intercept"] = model.intercept_
    res["top_rules"] = rules.head(15)[["rule_readable", "weight", "support", "importance"]].round(4).to_dict("records")
    rule_imp = rule_feature_importance(model)
    res["rule_feature_importance"] = rule_imp.round(4).to_dict()
    res["features_used"] = int((rule_imp > 0).sum())
    res["rules_per_prediction"] = rules_per_prediction(model, te)
    res["local_examples"], cases = local_examples(model, te, preds)
    shap_imp, row_rho = shap_rulefit(model, te)
    res["shap_importance"] = shap_imp.round(4).to_dict()
    res["shap_vs_exact_rowwise_spearman_median"] = row_rho
    perm = perm_importance(model, te)
    res["permutation_importance"] = perm.round(5).to_dict()
    res["importance_spearman"], res["importance_top5"] = importance_agreement(rule_imp, shap_imp, perm)
    res["pdp"] = pdp_ice(model, te)
    res["lime_borderline"] = lime_case(model, te, cases["Borderline"])
    res["direction_audit"] = direction_audit(rules)
    res["slices"] = slices(preds)
    (REPORTS / "explainability.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("slices",)}, indent=1, default=str)[:7000])


if __name__ == "__main__":
    main()
