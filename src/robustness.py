"""Step 4 - robustness checks for the final RuleFit model.

1. Unseen households: the main test set (Jul-Dec 2024) holds future months of
   households that also appear in training - the realistic case for an existing
   customer. Here 20% of households are held out entirely (GroupShuffleSplit).
2. Seed stability: rules come from a randomised tree ensemble, so we refit with
   different random seeds and check that accuracy and the most important inputs
   stay the same (a model whose explanation changed with the seed would not be
   trustworthy).

    python -m src.robustness
"""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.config import DATA_PROC, REPORTS, SEED, TRAIN_END, VALID_END
from src.features import make_X
from src.rulefit_model import RuleFitBillAlert
from src.train import TARGET, training_sample


def main():
    cp = pd.read_parquet(DATA_PROC / "checkpoints.parquet")
    params = json.loads((REPORTS / "metrics.json").read_text())["best_params"]["rulefit"]

    # 1. households never seen in training
    hh = cp.hid.drop_duplicates().to_numpy()
    tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(hh, groups=hh))
    tr_h, te_h = set(hh[tr_i]), set(hh[te_i])
    tr = training_sample(cp[(cp.period < VALID_END) & cp.hid.isin(tr_h)], 5)
    te = cp[(cp.period >= VALID_END) & cp.hid.isin(te_h)]
    X = make_X(te)
    p = RuleFitBillAlert(random_state=SEED, **params).fit(make_X(tr), tr[TARGET]).predict_proba(X)[:, 1]
    res = {"unseen_households": {
        "train_households": len(tr_h), "unseen_test_households": int(te.hid.nunique()), "test_rows": int(len(te)),
        "rulefit_accuracy": float(accuracy_score(te[TARGET], p >= .5)),
        "rulefit_auc": float(roc_auc_score(te[TARGET], p)),
        "pace_rule_accuracy_same_rows": float(accuracy_score(te[TARGET], X.pace_projection > X.limit))}}

    # 2. seed stability (train on the training months, score on validation months)
    trs = training_sample(cp[cp.period < TRAIN_END], 5)
    va = cp[(cp.period >= TRAIN_END) & (cp.period < VALID_END)]
    Xtr, Xva = make_X(trs), make_X(va)
    runs = []
    for seed in [0, 1, 2, 3, 4]:
        m = RuleFitBillAlert(random_state=seed, **params).fit(Xtr, trs[TARGET])
        runs.append({"seed": seed, "val_accuracy": float(accuracy_score(va[TARGET], m.predict(Xva))),
                     "n_rules": len(m.rules_), "top5": list(m.feature_importance().index[:5])})
        print(runs[-1], flush=True)
    tops = [set(r["top5"]) for r in runs]
    accs = pd.Series([r["val_accuracy"] for r in runs])
    res["seed_stability"] = {"runs": runs, "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
                             "accuracy_range": [float(accs.min()), float(accs.max())],
                             "inputs_in_every_top5": sorted(set.intersection(*tops)),
                             "mean_pairwise_top5_overlap": float(pd.Series(
                                 [len(a & b) for i, a in enumerate(tops) for b in tops[i + 1:]]).mean())}
    (REPORTS / "robustness.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
