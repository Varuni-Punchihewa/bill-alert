"""Central configuration: paths, constants, split dates and tariff.

Every script imports from here so that paths, random seeds and business
rules live in exactly one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
RECON = DATA_RAW / "recon_sl"                      # unzipped RECON-SL "data" folder
SURVEY = RECON / "survey_data"
MONTHLY_BILLS = RECON / "consumption_data" / "non_smart_meter" / "monthly_consumption.csv"
SMART_6H_DIR = RECON / "consumption_data" / "smart_meter" / "6hour_interval"
DAILY_REGISTER = DATA_RAW / "daily6h_all.csv.gz"     # output of 00_aggregate_smart_meter.py

# Convenience: also accept the layout of the original Assignment folder
# (Assignment/data/data/... and Assignment/work/daily6h_all.csv.gz) next to this project.
if not RECON.exists() and (ROOT.parent / "data" / "data").exists():
    RECON = ROOT.parent / "data" / "data"
    SURVEY = RECON / "survey_data"
    MONTHLY_BILLS = RECON / "consumption_data" / "non_smart_meter" / "monthly_consumption.csv"
    SMART_6H_DIR = RECON / "consumption_data" / "smart_meter" / "6hour_interval"
if not DAILY_REGISTER.exists() and (ROOT.parent / "work" / "daily6h_all.csv.gz").exists():
    DAILY_REGISTER = ROOT.parent / "work" / "daily6h_all.csv.gz"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SEED = 42

# ---------------------------------------------------------------- business rules
# PUCSL domestic (D-1) tariff block boundary. A household that uses more than
# 60 units in a 30-day billing period is moved to the "above 60" tariff where
# every unit is charged at a higher rate. Block sizes are pro-rated for billing
# periods that are not 30 days long.
THRESHOLD_UNITS = 60
BILLING_BASE_DAYS = 30

# Checkpoints at which the alert is issued: days 1..15 of the billing month,
# i.e. the first half of the month while the household can still act.
CHECKPOINT_DAYS = range(1, 16)

# Cleaning rules for the cumulative kWh register
MAX_DAILY_KWH = 200     # a single household day above this is a meter glitch
MIN_PREV_UNITS = 5      # last month < 5 units = house was empty; no alert needed

# Chronological split (by billing month) -> no future information in training
TRAIN_END = "2024-01-01"   # train: Mar 2023 .. Dec 2023
VALID_END = "2024-07-01"   # validation: Jan 2024 .. Jun 2024 ; test: Jul 2024 ..

# Domestic tariff (D-1) effective 11 May 2026 (PUCSL Annex-2). Used only by the
# app to translate units into rupees for the user; NOT used by the model.
TARIFF_EFFECTIVE = "11 May 2026"
TARIFF_LOW = [  # used when month total <= 60 units
    # (block upper limit, LKR per unit, fixed charge if the bill ends in this block)
    (30, 5.00, 80.0),
    (60, 9.00, 210.0),
]
TARIFF_HIGH = [  # used when month total is 61..180 units
    (60, 14.00, 0.0),
    (90, 20.00, 400.0),
    (120, 28.00, 1000.0),
    (180, 44.00, 1500.0),
]
TARIFF_VERY_HIGH = [  # used when month total > 180 units
    (180, 32.50, 0.0),
    (float("inf"), 100.00, 2500.0),
]

for _p in (DATA_PROC, MODELS, FIGURES):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except OSError:          # read-only deployments only need to read models/
        pass
