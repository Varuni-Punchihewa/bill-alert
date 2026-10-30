# Dataset: how it was obtained and compiled

The raw data is **not redistributed** with this submission. It is public and free to download, and the steps below
rebuild the exact modelling table used in the report.

## 1. Source

**RECON-SL: Residential Electricity Consumption Dataset for Sri Lanka**

* Publisher: IEEE DataPort, open access. DOI: [10.21227/n1dk-q860](https://doi.org/10.21227/n1dk-q860)
* Page: <https://ieee-dataport.org/open-access/residential-electricity-consumption-dataset-sri-lanka-recon-sl>
* Collected by LIRNEasia with Lanka Electricity Company (LECO), funded by the Lacuna Fund.
* Coverage: 4,063 LECO households in the western and southern coastal service areas (Moratuwa, Panadura, Galle,
  Kelaniya, Negombo and others).
* Anonymisation: households are identified only by a random ID. There are no names, addresses or account/meter numbers.
  Participants consented to the LIRNEasia survey.

| Part of RECON-SL | What it contains | Used here |
|---|---|---|
| `consumption_data/smart_meter/6hour_interval/` | Cumulative import-register readings every 6 hours, 2,950 metered households, Jan 2023 - Dec 2024 (about 1 GB, 4.4 M rows) | **Yes**: the target and all meter features |
| `consumption_data/smart_meter/15min_interval/` | Same meters at 15-minute resolution (4.6 GB) | No (future work) |
| `consumption_data/non_smart_meter/monthly_consumption.csv` | Monthly billed units for all 4,063 households, Oct 2022 - Oct 2024 | Exploratory analysis only |
| `survey_data/` (wave 1) | Household roster, dwelling and appliance tables | **Yes**: household features and fairness-audit fields |

## 2. How to get it

1. Create a free IEEE account and open the DataPort page above.
2. Download and unzip the dataset.
3. Place the unzipped `data` folder so that `data/raw/recon_sl/` contains `consumption_data/` and `survey_data/`.

## 3. How the modelling table is compiled

```bash
python -m src.aggregate_smart_meter   # 6-hour files -> one closing register reading per household-day (~15 min)
python -m src.data_prep               # cleaning, leakage-safe checkpoints, survey features (~1 min)
```

| Step | What happens | Numbers |
|---|---|---|
| Aggregate | 6-hour readings reduced to the closing register reading of each day | 1,813,466 household-days, 2,950 households |
| Clean | Days with a negative step (meter reset) or more than 200 kWh removed; months containing one invalidated | 4,232 glitch days removed |
| Exclude solar | Net-metered rooftop-solar homes removed (their import register is not their billed consumption) | 515 households |
| Monthly totals | Closing register minus the previous month's closing register (no interpolation, so a missing day inside the month costs nothing) | 42,293 complete household-months |
| Exclude vacant months | Rows of months following a month under 5 units (house empty) | 37,393 rows dropped |
| Checkpoints | For each household-month, one row per day 1-15, using only readings up to that day | **507,643 rows, 2,325 households, Mar 2023 - Dec 2024** |
| Survey join | Wave-1 answers encoded as counts and 0/1 flags; 1,122 smart-meter-only households have no survey (left missing) | |

## 4. The modelling table (`data/processed/checkpoints.parquet`)

One row = **(household, billing month, checkpoint day 1-15)**.

| Column | Meaning | Role |
|---|---|---|
| `hid`, `ym`, `period` | Household ID, billing month | Identifiers, never model inputs |
| `day`, `days_in_month` | Checkpoint day; length of the month | Input |
| `so_far`, `last7`, `last3` | Units used so far, in the last 7 days, in the last 3 days | Input |
| `prev1`, `prev2`, `prev3` | Units in each of the previous three complete months | Input |
| `month`, `weekend_days_left`, `holidays_left` | Calendar; Sri Lankan public and Poya holidays left | Input |
| `members`, `n_ac`, `n_fridge`, `electric_water_heater`, `washing_machine`, `electric_cooking`, `home_business` | Survey wave 1 | Input |
| `total` | Units used in the whole month | Used only to build the target |
| `limit` | 60 x days_in_month / 30 (the tariff pro-rates blocks) | Input (derived) |
| **`over_limit`** | **1 if `total` > `limit`** | **Target** |
| `socio_economic_class`, `csc_area`, `type_of_house` | Survey and utility fields | **Fairness audit only, never model inputs** |
| `prev_month_share_by_day` | Helper used in exploration | Not a model input |

The model also uses 14 engineered features computed from these inputs (`src/features.py`), shared by training and the app.

Share of household-months over the limit: 76.2%. 17.4% of months end within 15 units of the line.
