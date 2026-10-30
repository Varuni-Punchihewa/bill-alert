"""RuleFit with a relaxed-lasso refit: the model behind Bill Alert.

RuleFit (Friedman & Popescu, 2008) learns a short list of IF-THEN rules and a
weight for each:

    log-odds(over the limit) = b0 + sum_k  w_k * [rule_k is true for this household]

Three stages:
  1. Rule generation - an ensemble of small trees of varying size is grown with
     gradient boosting (imodels' implementation). Every node of every tree
     becomes a candidate rule, e.g. "allowed_per_day <= 2.0 and prev1 > 61".
  2. Rule selection - an L1 (lasso) penalised logistic regression over all
     candidate rules keeps at most `max_rules` of them; the rest get weight 0.
  3. Relaxed refit - an ordinary (lightly L2-penalised) logistic regression is
     refitted on the selected rules only. The lasso shrinks every weight towards
     zero to choose rules; refitting removes that shrinkage, so the weights are
     true log-odds and the probabilities are calibrated (relaxed lasso,
     Meinshausen 2007).

Why our own wrapper: imodels' RuleFitClassifier turns its linear score z into a
probability with softmax([1 - z, z]) = sigmoid(2z - 1), which is not the fitted
logistic model's probability. We use sigmoid(z) of the refitted model.

Explanations are exact: a prediction's log-odds is the intercept plus the
weights of the rules that are true for that household, nothing else.
"""
import numpy as np
import pandas as pd
from imodels import RuleFitClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


class RuleFitBillAlert(BaseEstimator, ClassifierMixin):
    def __init__(self, max_rules=40, n_estimators=100, tree_size=4, C=1.0, random_state=42):
        self.max_rules = max_rules
        self.n_estimators = n_estimators
        self.tree_size = tree_size
        self.C = C
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y).astype(int)
        self.feature_names_ = list(X.columns)
        self.imputer_ = SimpleImputer(strategy="median").fit(X)
        A = self._impute(X)
        self.generator_ = RuleFitClassifier(max_rules=self.max_rules, n_estimators=self.n_estimators,
                                            tree_size=self.tree_size, include_linear=False,
                                            random_state=self.random_state)
        self.generator_.fit(A, y, feature_names=self.feature_names_)
        codes = [r.rule for r in self.generator_.rules_without_feature_names_]   # "X_3 <= 2.0 and X_7 > 61"
        names = [str(r) for r in self.generator_.rules_]                          # same rules, real names
        R = self.generator_.transform(A, codes)
        # Drop rules that are always/never true, and rules true for exactly the
        # same homes as a simpler rule (they carry no extra information and would
        # split one weight between two lines of the explanation).
        keep, seen = [], set()
        for k in sorted(range(len(codes)), key=lambda k: (codes[k].count(" and "), k)):
            key = np.packbits(R[:, k].astype(bool)).tobytes()
            if 0 < R[:, k].mean() < 1 and key not in seen:
                seen.add(key)
                keep.append(k)
        keep.sort()
        self.rule_codes_ = [codes[k] for k in keep]
        self.rules_ = [names[k] for k in keep]
        R = R[:, keep]
        self.support_ = R.mean(axis=0)
        self.lr_ = LogisticRegression(C=self.C, max_iter=5000).fit(R, y)
        self.coef_ = self.lr_.coef_.ravel()
        self.intercept_ = float(self.lr_.intercept_[0])
        self.classes_ = np.array([0, 1])
        return self

    # -------------------------------------------------------------- predict
    def _impute(self, X):
        X = X[self.feature_names_] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_)
        return self.imputer_.transform(X).astype(np.float32)

    def _rules_true(self, A):
        return self.generator_.transform(A, self.rule_codes_)

    def decision_function(self, X):
        return self.intercept_ + self._rules_true(self._impute(X)) @ self.coef_

    def predict_proba(self, X):
        z = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # ---------------------------------------------------------- explanation
    def rule_table(self):
        """The whole model: every rule, its weight (log-odds) and how often it is true.

        importance = |w| * sqrt(s (1 - s)) is Friedman & Popescu's global rule
        importance: a rule matters when its weight is large AND it splits the
        data (s = share of households for which it is true).
        """
        t = pd.DataFrame({"rule": self.rules_, "weight": self.coef_, "support": self.support_})
        t["importance"] = t.weight.abs() * np.sqrt(t.support * (1 - t.support))
        t["features"] = [sorted({c.split(" ")[0] for c in r.split(" and ")}) for r in t.rule]
        return t.sort_values("importance", ascending=False).reset_index(drop=True)

    def explain(self, X):
        """Per row: which rules are true and what each adds to the log-odds (exact)."""
        R = self._rules_true(self._impute(X))
        return [{"intercept": self.intercept_,
                 "rules": [(self.rules_[k], float(self.coef_[k])) for k in np.flatnonzero(row)],
                 "log_odds": float(self.intercept_ + row @ self.coef_)} for row in R]

    @property
    def baseline_(self):
        """Log-odds of an 'average' household: b0 + sum_k w_k * support_k."""
        return float(self.intercept_ + self.support_ @ self.coef_)

    def contributions(self, X):
        """Exact, centred contribution of every rule for every row.

        c_k = w_k * (r_k - s_k), so  log-odds = baseline_ + sum_k c_k  exactly.
        A positive c_k means "this rule makes the home look MORE likely to go
        over than an average home"; this equals the exact SHAP value of a model
        that is linear in the rule indicators.
        """
        R = self._rules_true(self._impute(X)).astype(float)
        return (R - self.support_) * self.coef_

    def feature_contributions(self, X):
        """Rule contributions shared equally among the features each rule tests."""
        C = self.contributions(X)
        F = pd.DataFrame(0.0, index=range(len(C)), columns=self.feature_names_)
        for k, feats in enumerate(self.rule_table().set_index("rule").loc[self.rules_, "features"]):
            for f in feats:
                F[f] += C[:, k] / len(feats)
        return F

    def feature_importance(self):
        """Rule importance shared equally among the features each rule uses."""
        imp = {f: 0.0 for f in self.feature_names_}
        for _, r in self.rule_table().iterrows():
            for f in r.features:
                imp[f] += r.importance / len(r.features)
        s = pd.Series(imp)
        return (s / s.sum()).sort_values(ascending=False)
