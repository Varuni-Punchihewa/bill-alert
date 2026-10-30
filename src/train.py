"""Step 2 - tune, compare and evaluate; the final model is RuleFit (not taught in class).

Protocol (chronological, no shuffling across time):
    train      : billing months Mar-2023 .. Dec-2023
    validation : Jan-2024 .. Jun-2024   -> tuning + model comparison
    test       : Jul-2024 .. Dec-2024   -> touched once, after every choice is fixed

    python -m src.train            # full run (about an hour on a 2-CPU laptop)
    python -m src.train --quick    # small sample, to smoke-test the code
"""
import argparse
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imodels import FIGSClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import DATA_PROC, MODELS, REPORTS, SEED, TRAIN_END, VALID_END
from src.features import FEATURES, make_X
from src.rulefit_model import RuleFitBillAlert

warnings.filterwarnings("ignore")
TARGET = "over_limit"
FINAL = "RuleFit (relaxed)"

RULEFIT_GRID = [dict(max_rules=30), dict(max_rules=40), dict(max_rules=60), dict(max_rules=80),
                dict(max_rules=40, n_estimators=300), dict(max_rules=60, n_estimators=300),
                dict(max_rules=40, tree_size=6), dict(max_rules=60, n_estimators=300, tree_size=6)]
FIGS_GRID = [dict(max_rules=8), dict(max_rules=12), dict(max_rules=16), dict(max_rules=20), dict(max_rules=25)]
LGBM_GRID = [dict(num_leaves=31, learning_rate=0.05, n_estimators=400),
             dict(num_leaves=63, learning_rate=0.03, n_estimators=600),
             dict(num_leaves=127, learning_rate=0.03, n_estimators=800, min_child_samples=50)]


# ---------------------------------------------------------------- data
def load_splits(quick=False):
    cp = pd.read_parquet(DATA_PROC / "checkpoints.parquet")
    if quick:
        hh = cp.hid.drop_duplicates().sample(80, random_state=SEED)
        cp = cp[cp.hid.isin(hh)]
    return (cp[cp.period < TRAIN_END], cp[(cp.period >= TRAIN_END) & (cp.period < VALID_END)],
            cp[cp.period >= VALID_END])


def training_sample(tr, n_per_month=None):
    """Each household-month contributes up to n of its 15 checkpoint days.

    Consecutive days of one month are highly correlated; sampling keeps every
    month while fitting a 2-CPU budget.
    """
    if n_per_month is None:
        return tr
    shuffled = tr.sample(frac=1.0, random_state=SEED)
    keep = shuffled.groupby(["hid", "ym"]).cumcount() < n_per_month
    return shuffled[keep].sort_index()


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "precision_over": precision_score(y, yhat, zero_division=0),
        "recall_over": recall_score(y, yhat, zero_division=0),
        "recall_under": recall_score(1 - y, 1 - yhat, zero_division=0),
        "f1_macro": f1_score(y, yhat, average="macro"),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(p)) > 2 else np.nan,
        "brier": brier_score_loss(y, np.clip(p, 0, 1)),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
    }


# ------------------------------------------------------------- baselines
class PaceRule:
    """What a careful household does by hand: extrapolate the month-to-date average."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = (X["pace_projection"] > X["limit"]).astype(float).to_numpy()
        return np.c_[1 - p, p]


class LastBillRule:
    """'You crossed 60 last month, so you will this month.'"""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        p = (X["prev1"] > X["limit"]).astype(float).to_numpy()
        return np.c_[1 - p, p]


class FIGSWrapped:
    """FIGS needs a complete numeric matrix: median-impute first."""

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        self.imp_ = SimpleImputer(strategy="median").fit(X)
        self.m_ = FIGSClassifier(random_state=SEED, **self.kw).fit(self.imp_.transform(X), y, feature_names=list(X.columns))
        return self

    def predict_proba(self, X):
        return self.m_.predict_proba(self.imp_.transform(X))


def impute(model, scale=False):
    steps = [SimpleImputer(strategy="median", add_indicator=True)]
    if scale:
        steps.append(StandardScaler())
    return make_pipeline(*steps, model)


def rf():
    return impute(RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_samples=0.3, n_jobs=-1,
                                         random_state=SEED))


def xgb():
    return XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         n_jobs=-1, random_state=SEED)


def lgbm(params):
    return LGBMClassifier(verbose=-1, random_state=SEED, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, **params)


def ebm():
    return ExplainableBoostingClassifier(interactions=10, outer_bags=8, inner_bags=0, learning_rate=0.02,
                                         max_rounds=5000, early_stopping_rounds=100, n_jobs=-1, random_state=SEED)


def ensemble(kind, lgbm_params):
    members = [("rf", rf()), ("xgb", xgb()), ("lgbm", lgbm(lgbm_params))]
    if kind == "voting":
        return VotingClassifier(members, voting="soft")
    return StackingClassifier(members, final_estimator=LogisticRegression(max_iter=2000), cv=3,
                              stack_method="predict_proba", n_jobs=1)


def candidate_models(best):
    """name -> (family, model). Families follow the lecture coverage table."""
    return {
        "Rule: pace projection": ("Baseline rule", PaceRule()),
        "Rule: last month's bill": ("Baseline rule", LastBillRule()),
        "Logistic Regression": ("Taught: single model", impute(LogisticRegression(max_iter=3000), scale=True)),
        "Logistic Regression (Lasso / L1)": ("Taught: single model",
                                             impute(LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                                                       max_iter=3000), scale=True)),
        "k-Nearest Neighbours": ("Taught: single model", impute(KNeighborsClassifier(n_neighbors=51, n_jobs=-1), scale=True)),
        "SVM (RBF kernel)": ("Taught: single model", impute(SVC(C=1.0, probability=True, random_state=SEED), scale=True)),
        "Decision Tree": ("Taught: single model", impute(DecisionTreeClassifier(max_depth=8, min_samples_leaf=50,
                                                                               random_state=SEED))),
        "Bagging (decision trees)": ("Taught: bagging", impute(BaggingClassifier(DecisionTreeClassifier(min_samples_leaf=20),
                                                                                 n_estimators=100, max_samples=0.3,
                                                                                 n_jobs=-1, random_state=SEED))),
        "Random Forest": ("Taught: bagging", rf()),
        "AdaBoost": ("Taught: boosting", impute(AdaBoostClassifier(n_estimators=200, random_state=SEED))),
        "Gradient Boosting": ("Taught: boosting", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                                                 random_state=SEED)),
        "XGBoost": ("Taught: boosting", xgb()),
        "LightGBM": ("Taught: boosting", lgbm(best["lgbm"])),
        "CatBoost": ("Taught: boosting", CatBoostClassifier(iterations=600, learning_rate=0.08, depth=6, verbose=0,
                                                             random_seed=SEED, allow_writing_files=False)),
        "Voting (RF + XGBoost + LightGBM)": ("Taught: ensemble", ensemble("voting", best["lgbm"])),
        "Stacking (RF + XGBoost + LightGBM -> LR)": ("Taught: ensemble", ensemble("stacking", best["lgbm"])),
        "Explainable Boosting Machine": ("Taught: glass-box", ebm()),
        "Gaussian Naive Bayes": ("Not taught", impute(GaussianNB())),
        "Extra Trees": ("Not taught", impute(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=10, max_samples=0.3,
                                                                   bootstrap=True, n_jobs=-1, random_state=SEED))),
        "FIGS (greedy tree sums)": ("Not taught", FIGSWrapped(**best["figs"])),
        FINAL: ("New algorithm (final)", RuleFitBillAlert(random_state=SEED, **best["rulefit"])),
    }


SUBSAMPLE = {"k-Nearest Neighbours": 60_000, "SVM (RBF kernel)": 15_000,   # cost grows with training size
             "Logistic Regression (Lasso / L1)": 30_000}                 # liblinear L1 stalls on the full set


# ------------------------------------------------------------------- tuning
def tune(name, make, grid, Xtr, ytr, Xva, yva, size=None):
    rows = []
    for g in grid:
        t = time.time()
        m = make(g).fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        r = {"model": name, "params": json.dumps(g), "val_accuracy": accuracy_score(yva, p >= .5),
             "val_balanced_accuracy": balanced_accuracy_score(yva, p >= .5), "val_auc": roc_auc_score(yva, p),
             "val_brier": brier_score_loss(yva, p), "size": size(m) if size else None,
             "seconds": round(time.time() - t)}
        rows.append(r)
        print("  tune", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    return pick(df, size is not None), df


def pick(df, by_size):
    """Best validation accuracy; a tie (within 0.0005) goes to the smaller model."""
    top = df.val_accuracy.max()
    cand = df[df.val_accuracy >= top - 0.0005]
    cand = (cand.sort_values(["size", "val_accuracy"], ascending=[True, False], kind="stable") if by_size
            else cand.sort_values("val_accuracy", ascending=False, kind="stable"))
    return json.loads(cand.iloc[0]["params"])


# -------------------------------------------------------------- evaluation
def cluster_bootstrap_ci(df, col_y, col_p, n=500, seed=SEED):
    """95% CI for accuracy, resampling *households* (rows of one household are correlated)."""
    rng = np.random.default_rng(seed)
    g = df.assign(c=((df[col_p] >= .5).astype(int) == df[col_y]).astype(int)).groupby("hid").c.agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    accs = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main(quick=False, reuse_tuning=False):
    t0 = time.time()
    tr, va, te = load_splits(quick)
    n_per_month = 3 if quick else 5
    trs = training_sample(tr, n_per_month)
    Xtr, ytr = make_X(trs), trs[TARGET].to_numpy()
    Xva, yva = make_X(va), va[TARGET].to_numpy()
    Xte, yte = make_X(te), te[TARGET].to_numpy()
    split_info = {k: {"rows": int(len(d)), "households": int(d.hid.nunique()),
                      "months": f"{d.ym.min()} .. {d.ym.max()}", "share_over_limit": round(float(d[TARGET].mean()), 4)}
                  for k, d in [("train", tr), ("train_sample_used", trs), ("validation", va), ("test", te)]}
    print(json.dumps(split_info, indent=2), flush=True)

    # ---------------- 1. tuning on the validation months
    q = (lambda g: g[:2]) if quick else (lambda g: g)
    if reuse_tuning:   # same grids, same data, same selection rule - just not recomputed
        t = pd.read_csv(REPORTS / "tuning_results.csv")
        best = {"rulefit": pick(t[t.model == "RuleFit"], True), "figs": pick(t[t.model == "FIGS"], True),
                "lgbm": pick(t[t.model == "LightGBM"], False)}
        print("reusing tuning:", best, flush=True)
    else:
        best = tune_all(q, Xtr, ytr, Xva, yva)
    run_rest(best, t0, tr, va, te, trs, n_per_month, split_info, Xtr, ytr, Xva, yva, Xte, yte)


def tune_all(q, Xtr, ytr, Xva, yva):
    best = {}
    best["rulefit"], g1 = tune("RuleFit", lambda g: RuleFitBillAlert(random_state=SEED, **g), q(RULEFIT_GRID),
                               Xtr, ytr, Xva, yva, size=lambda m: len(m.rules_))
    best["figs"], g2 = tune("FIGS", lambda g: FIGSWrapped(**g), q(FIGS_GRID), Xtr, ytr, Xva, yva,
                            size=lambda m: m.m_.complexity_)
    best["lgbm"], g3 = tune("LightGBM", lgbm, q(LGBM_GRID), Xtr, ytr, Xva, yva)
    pd.concat([g1, g2, g3]).to_csv(REPORTS / "tuning_results.csv", index=False)
    return best


def run_rest(best, t0, tr, va, te, trs, n_per_month, split_info, Xtr, ytr, Xva, yva, Xte, yte):

    # ---------------- 2. every model on the validation months
    rows, val_pred = [], {}
    for name, (family, m) in candidate_models(best).items():
        t = time.time()
        if name in SUBSAMPLE:
            sel = np.random.default_rng(SEED).choice(len(Xtr), min(SUBSAMPLE[name], len(Xtr)), replace=False)
            m.fit(Xtr.iloc[sel], ytr[sel])
        else:
            m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        val_pred[name] = p
        rows.append({"model": name, "family": family, **metrics(yva, p), "fit_seconds": round(time.time() - t, 1)})
        print(f"  {name:42s} acc={rows[-1]['accuracy']:.4f} bal={rows[-1]['balanced_accuracy']:.4f} "
              f"({rows[-1]['fit_seconds']}s)", flush=True)
    pd.DataFrame(rows).to_csv(REPORTS / "validation_comparison.csv", index=False)
    pd.DataFrame(val_pred).assign(day=va.day.to_numpy(), y=yva).to_parquet(REPORTS / "validation_predictions.parquet")

    # ---------------- 3. finalists refitted on train + validation, scored ONCE on test
    print("Refitting finalists on train+validation ...", flush=True)
    full = pd.concat([trs, training_sample(va, n_per_month)])
    Xf, yf = make_X(full), full[TARGET].to_numpy()
    cm = candidate_models(best)
    finalists = {k: cm[k][1] for k in ["Rule: pace projection", "Logistic Regression", "Random Forest", "XGBoost",
                                       "LightGBM", "Stacking (RF + XGBoost + LightGBM -> LR)",
                                       "Explainable Boosting Machine", "FIGS (greedy tree sums)", FINAL]}
    test_rows, test_pred = [], {}
    for name, m in finalists.items():
        m.fit(Xf, yf)
        p = m.predict_proba(Xte)[:, 1]
        test_pred[name] = p
        test_rows.append({"model": name, "family": cm[name][0], **metrics(yte, p)})
        print(f"  TEST {name:42s} acc={test_rows[-1]['accuracy']:.4f}", flush=True)
    test = pd.DataFrame(test_rows)
    test.to_csv(REPORTS / "test_results.csv", index=False)

    final = finalists[FINAL]
    te_out = te[["hid", "ym", "day", "so_far", "total", "limit", TARGET, "prev1", "socio_economic_class",
                 "csc_area", "type_of_house", "n_ac", "members"]].copy()
    for name, p in test_pred.items():
        te_out["p_" + name] = p
    te_out["p"] = test_pred[FINAL]
    te_out.to_parquet(REPORTS / "test_predictions.parquet")
    lo, hi = cluster_bootstrap_ci(te_out, TARGET, "p")
    joblib.dump({"model": final, "features": FEATURES, "params": best["rulefit"],
                 "trained_on": f"{full.ym.min()} .. {full.ym.max()}"}, MODELS / "rulefit_bill_alert.joblib")
    final.rule_table().to_csv(REPORTS / "rulefit_rules.csv", index=False)

    summary = {
        "splits": split_info,
        "best_params": best,
        "final_model": FINAL,
        "n_rules": len(final.rules_),
        "test_metrics": test.set_index("model").loc[FINAL].drop("family").to_dict(),
        "test_accuracy_95ci_household_bootstrap": [lo, hi],
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reuse-tuning", action="store_true", help="read the grid-search results instead of recomputing")
    main(**vars(ap.parse_args()))
