"""Step 2b - ablation: what each design decision in the final RuleFit contributes.

All variants: trained on the training months, scored on the validation months,
same rule-generation settings (the tuned ones). The test set is not used.

    python -m src.ablation
"""
import json
import warnings

import numpy as np
import pandas as pd
from imodels import RuleFitClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss, roc_auc_score

from src.config import DATA_PROC, REPORTS, SEED, TRAIN_END, VALID_END
from src.features import make_X
from src.rulefit_model import RuleFitBillAlert
from src.train import TARGET, training_sample

warnings.filterwarnings("ignore")


def score(name, y, p, n_rules, note):
    return {"variant": name, "rules": n_rules, "accuracy": accuracy_score(y, p >= .5),
            "balanced_accuracy": balanced_accuracy_score(y, p >= .5), "roc_auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)), "note": note}


def main():
    cp = pd.read_parquet(DATA_PROC / "checkpoints.parquet")
    params = json.loads((REPORTS / "metrics.json").read_text())["best_params"]["rulefit"]
    tr = training_sample(cp[cp.period < TRAIN_END], 5)
    va = cp[(cp.period >= TRAIN_END) & (cp.period < VALID_END)]
    Xtr, ytr, Xva, yva = make_X(tr), tr[TARGET].to_numpy(), make_X(va), va[TARGET].to_numpy()
    imp = SimpleImputer(strategy="median").fit(Xtr)
    A, B = imp.transform(Xtr).astype(np.float32), imp.transform(Xva).astype(np.float32)
    names = list(Xtr.columns)
    rows = []

    lib = RuleFitClassifier(include_linear=False, random_state=SEED, **params).fit(A, ytr, feature_names=names)
    n_lib = len(lib.rules_without_feature_names_)
    rows.append(score("imodels RuleFitClassifier, as shipped", yva, lib.predict_proba(B)[:, 1], n_lib,
                      "probability = softmax([1-z, z]) = sigmoid(2z-1)"))
    z = lib._predict_continuous_output(B)
    rows.append(score("Same rules and lasso weights, sigmoid(z)", yva, 1 / (1 + np.exp(-z)), n_lib,
                      "correct link function, weights still shrunk by the lasso"))
    codes = [r.rule for r in lib.rules_without_feature_names_]
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(lib.transform(A, codes), ytr)
    rows.append(score("Relaxed refit, no de-duplication", yva, lr.predict_proba(lib.transform(B, codes))[:, 1], n_lib,
                      "lasso selects, ordinary logistic regression re-estimates the weights"))
    final = RuleFitBillAlert(random_state=SEED, **params).fit(Xtr, ytr)
    rows.append(score("Relaxed refit + de-duplication (final)", yva, final.predict_proba(Xva)[:, 1],
                      len(final.rules_), "identical / constant rules removed before the refit"))
    lin = RuleFitClassifier(include_linear=True, random_state=SEED, **params).fit(A, ytr, feature_names=names)
    zl = lin._predict_continuous_output(B)
    n_lin = len(lin._get_rules(exclude_zero_coef=True))
    rows.append(score("Rules + linear terms (original RuleFit), sigmoid(z)", yva, 1 / (1 + np.exp(-zl)), n_lin,
                      "adds raw features as linear terms; terms are no longer all IF-THEN sentences"))
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "ablation.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
