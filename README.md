# Bill Alert: will a Sri Lankan home cross the 60-unit electricity line this month?

Applied Machine Learning assignment, MSc in Artificial Intelligence, University of Moratuwa.

In the PUCSL domestic tariff, a household that uses more than **60 units** in a 30-day billing period
moves to the "above 60" rate structure: **60 units cost about Rs 630 and 61 units about Rs 1,260**.
Most homes only find out when the bill arrives. This project predicts, **on any of the first 15 days of
the billing month**, whether the month will end above the (pro-rated) 60-unit limit. It uses the household's
meter readings so far, its last three bills, and a few facts about the home.

* **Model:** **RuleFit** (Friedman & Popescu, 2008), a sparse logistic model over IF-THEN rules mined from a
  boosted tree ensemble, with a relaxed-lasso refit for calibrated probabilities (`src/rulefit_model.py`).
  It was not covered in the lectures. It is compared with every model family taught (linear models, SVM, trees,
  k-NN, bagging, random forest, AdaBoost, gradient boosting, XGBoost, LightGBM, CatBoost, voting, stacking, EBM),
  plus FIGS, extra trees, naive Bayes and two hand rules.
* **Front end:** a Streamlit app for people with no ML background (`app/streamlit_app.py`).
* **Report:** `Bill_Alert_Report.pdf`. **Demo:** `Bill_Alert_demo.mp4`. **Data:** see `DATASET.md`.

## Results

Test set: Jul-Dec 2024, 89,232 checkpoints, 2,123 households, used once after every design choice was fixed on the validation months.

| Model | Family | Accuracy | Balanced acc. | ROC-AUC | Brier |
|---|---|---|---|---|---|
| **RuleFit (relaxed)** | New algorithm (final) | **95.57%** (95% CI 95.05%-96.11%) | 92.60% | 0.978 | 0.035 |
| Random Forest | Taught: bagging | 95.45% | 91.90% | 0.988 | 0.033 |
| FIGS (greedy tree sums) | Not taught | 95.41% | 92.31% | 0.978 | 0.096 |
| XGBoost | Taught: boosting | 95.39% | 92.03% | 0.987 | 0.034 |
| Stacking (RF + XGBoost + LightGBM -> LR) | Taught: ensemble | 95.37% | 91.64% | 0.987 | 0.036 |
| LightGBM | Taught: boosting | 95.32% | 91.86% | 0.986 | 0.036 |
| Explainable Boosting Machine | Taught: glass-box | 95.21% | 92.23% | 0.984 | 0.036 |
| Logistic Regression | Taught: single model | 95.18% | 91.57% | 0.985 | 0.036 |
| Rule: pace projection | Baseline rule | 94.40% | 89.93% | - | 0.056 |

* The final model is **36 IF-THEN rules** using 8 of the 32 inputs; the full rulebook is in `reports/rulefit_rules_explained.csv` and Appendix B of the report.
* On 421 households never seen in training it scores 96.40%.
* Its exact rule explanations agree with SHAP (rank correlation 0.999).
* The full comparison of 21 approaches on the validation months is in `reports/validation_comparison.csv` and the report.

## Quick start (the app)

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py     # -> http://localhost:8501
```

The trained model (`models/rulefit_bill_alert.joblib`) is included, so the app runs without the raw data.

## Reproduce everything

Download RECON-SL as described in `DATASET.md`, then:

```bash
pip install -r requirements-dev.txt
python -m src.aggregate_smart_meter    # ~15 min, once (6-hour files -> daily register readings)
python -m src.data_prep                # ~1 min
python -m src.eda
python -m src.train                    # 2-CPU laptop: about an hour (--quick: ~2 min smoke test)
python -m src.ablation                 # what each part of the RuleFit design adds
python -m src.explain                  # rules, SHAP, permutation importance, PDP/ICE, LIME, fairness
python -m src.robustness               # unseen households + seed stability
python -m src.build_report             # regenerates reports/Bill_Alert_Report.pdf
pytest -q
```

## Project layout

```
src/
  config.py                  paths, seeds, split dates, tariff - single source of truth
  aggregate_smart_meter.py   6-hour raw files -> one register reading per household-day
  data_prep.py               cleaning, leakage-safe checkpoints, survey features
  eda.py                     exploratory figures
  features.py                feature engineering shared by training and the app (no train/serve skew)
  rulefit_model.py           RuleFit + relaxed refit + exact explanations (the model)
  train.py                   tuning, comparison of 21 approaches, final test evaluation, saved model
  ablation.py                probability link / relaxed refit / de-duplication / linear terms
  explain.py                 rulebook, rule importance, local explanations, SHAP, PDP/ICE, LIME, fairness
  robustness.py              unseen-household check and seed stability
  explain_text.py            rules -> plain-language sentences and grouped reasons
  tariff.py, plotstyle.py    bill estimate; figure style
  build_report.py            the PDF report, generated from saved results
app/streamlit_app.py         front end for households / LECO staff
models/                      the final RuleFit model
reports/                     metrics, comparison tables, the full rulebook (CSV), figures, report
tests/                       pytest unit tests + Playwright UI check (app_screenshots.js)
```

## Methodology at a glance

* **Unit of prediction:** household x billing month x checkpoint day (1-15).
* **Target:** month total > 60 x days/30 units (the tariff pro-rates blocks by period length).
* **Leakage control:** usage so far comes only from the register reading on the checkpoint day (no interpolation).
  Bill history uses only earlier complete months. The split is chronological: train Mar-Dec 2023, validation Jan-Jun 2024,
  test Jul-Dec 2024 (scored once).
* **Exclusions:** rooftop-solar (net-metered) homes, months after a vacant month (<5 units), meter resets and glitches.
* **Ethics:** socio-economic class, area and house type are never model inputs. They are used only to audit errors.
