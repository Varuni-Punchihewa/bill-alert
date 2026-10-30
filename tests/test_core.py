"""Unit tests: tariff maths, feature engineering, and leakage guards.

    pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, add_derived, make_X, month_limit
from src.tariff import estimate_bill


# ------------------------------------------------------------------ tariff
@pytest.mark.parametrize("units,expected", [(30, 230), (60, 630), (61, 1260), (90, 1840), (180, 6420)])
def test_tariff_blocks(units, expected):
    assert estimate_bill(units) == pytest.approx(expected)


def test_crossing_60_doubles_the_bill():
    assert estimate_bill(61) >= 2 * estimate_bill(60)


def test_limit_is_prorated():
    assert month_limit(30) == pytest.approx(60)
    assert month_limit(31) == pytest.approx(62)
    assert month_limit(28) == pytest.approx(56)


# ---------------------------------------------------------------- features
def _row(**kw):
    base = dict(day=10, days_in_month=30, so_far=20.0, last7=14.0, last3=6.0, weekend_days_left=6,
                holidays_left=1, prev1=70.0, prev2=65.0, prev3=60.0, month=4, members=4, n_ac=0, n_fridge=1,
                electric_water_heater=0, washing_machine=1, electric_cooking=1, home_business=0)
    base.update(kw)
    return pd.DataFrame([base])


def test_feature_matrix_shape_and_order():
    X = make_X(_row())
    assert list(X.columns) == FEATURES
    assert X.shape == (1, len(FEATURES))


def test_derived_values():
    d = add_derived(_row()).iloc[0]
    assert d.pace_projection == pytest.approx(60.0)      # 20 units in 10 days -> 60 in 30
    assert d.units_left == pytest.approx(40.0)
    assert d.allowed_per_day == pytest.approx(2.0)       # 40 units over 20 days
    assert d.history_avg == pytest.approx(65.0)
    assert d.allowed_vs_recent == pytest.approx(1.0)     # allowed 2/day, recent 2/day


def test_missing_history_is_tolerated():
    X = make_X(_row(prev2=np.nan, prev3=np.nan, last7=np.nan))
    assert X.history_avg.iloc[0] == pytest.approx(70.0)
    assert np.isnan(X.recent_rate.iloc[0])


# -------------------------------------------------------- leakage guards
def test_no_target_columns_in_features():
    for leaky in ["total", "over_limit", "hid", "ym", "period", "socio_economic_class"]:
        assert leaky not in FEATURES


def test_processed_data_respects_time(tmp_path_factory):
    """If processed data exists: usage so far never exceeds the month total and
    the chronological splits do not overlap."""
    from src.config import DATA_PROC, TRAIN_END, VALID_END
    p = DATA_PROC / "checkpoints.parquet"
    if not p.exists():
        pytest.skip("run python -m src.data_prep first")
    cp = pd.read_parquet(p, columns=["so_far", "total", "day", "period", "prev1"])
    assert (cp.so_far <= cp.total + 1e-6).all()
    assert cp.day.between(1, 15).all()
    assert (cp.prev1 >= 5).all()
    tr = cp[cp.period < TRAIN_END].period.max()
    te = cp[cp.period >= VALID_END].period.min()
    assert tr < pd.Timestamp(TRAIN_END) <= pd.Timestamp(VALID_END) <= te


# ------------------------------------------------------------ RuleFit model
@pytest.fixture(scope="module")
def small_model():
    from src.rulefit_model import RuleFitBillAlert
    rng = np.random.default_rng(0)
    n = 3000
    df = pd.DataFrame(dict(day=rng.integers(1, 16, n), days_in_month=30, so_far=rng.uniform(1, 60, n),
                           last7=rng.uniform(5, 40, n), last3=rng.uniform(1, 15, n), weekend_days_left=6,
                           holidays_left=1, prev1=rng.uniform(20, 150, n), prev2=rng.uniform(20, 150, n),
                           prev3=rng.uniform(20, 150, n), month=rng.integers(1, 13, n), members=rng.integers(1, 7, n),
                           n_ac=rng.integers(0, 3, n), n_fridge=1, electric_water_heater=rng.integers(0, 2, n),
                           washing_machine=1, electric_cooking=1, home_business=0))
    X = make_X(df)
    y = ((X.pace_projection + 0.3 * X.history_avg + rng.normal(0, 8, n)) > 75).astype(int).to_numpy()
    return RuleFitBillAlert(max_rules=20, n_estimators=50, random_state=0).fit(X, y), X


def test_rulefit_explanation_is_exact(small_model):
    m, X = small_model
    z = m.decision_function(X)
    assert np.allclose(m.baseline_ + m.contributions(X).sum(1), z)
    assert np.allclose(m.feature_contributions(X).sum(1), m.contributions(X).sum(1))


def test_rulefit_probability_is_logistic(small_model):
    m, X = small_model
    z = m.decision_function(X)
    assert np.allclose(m.predict_proba(X)[:, 1], 1 / (1 + np.exp(-z)))


def test_rulefit_rules_are_unique_and_informative(small_model):
    m, X = small_model
    R = m._rules_true(m._impute(X))
    assert len({R[:, k].tobytes() for k in range(R.shape[1])}) == R.shape[1]
    assert ((m.support_ > 0) & (m.support_ < 1)).all()


def test_rules_read_as_plain_language(small_model):
    from src.explain_text import rule_plain, rule_short
    m, _ = small_model
    for r in m.rules_:
        assert "_" not in rule_plain(r)         # no raw feature names leak to users
        assert rule_short(r)
