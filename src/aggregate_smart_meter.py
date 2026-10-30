"""Step 0 - collapse the raw 6-hour smart-meter files (~1 GB) to one row per
household per day.

The meter reports *cumulative* registers (total import, export and the three
time-of-use registers). The largest value seen on a day is that day's closing
reading. Run once; output is data/raw/daily6h_all.csv.gz.

    python -m src.aggregate_smart_meter
"""
import time

import pandas as pd

from src.config import SMART_6H_DIR, DAILY_REGISTER

RAW_COLS = ["household_ID", "DATE", "TOTAL_IMPORT (kWh)", "TOTAL_EXPORT (kWh)",
            "TR1_TOTAL_IMPORT (kWh)", "TR2_TOTAL_IMPORT (kWh)", "TR3_TOTAL_IMPORT (kWh)"]
NEW_COLS = ["hid", "date", "imp", "exp", "tr1", "tr2", "tr3"]


def aggregate_file(path, chunksize=400_000):
    parts = []
    for ch in pd.read_csv(path, usecols=RAW_COLS, chunksize=chunksize, dtype=str):
        ch.columns = NEW_COLS
        for c in NEW_COLS[2:]:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        parts.append(ch.groupby(["hid", "date"]).max())
    return pd.concat(parts).groupby(level=[0, 1]).max()


def main():
    files = sorted(SMART_6H_DIR.glob("smart_6hour_*.csv"))
    if not files:
        raise FileNotFoundError(f"No 6-hour files found in {SMART_6H_DIR}")
    out = []
    for f in files:
        t = time.time()
        d = aggregate_file(f)
        print(f"{f.name}: {d.shape[0]:,} household-days in {time.time() - t:.0f}s")
        out.append(d)
    daily = pd.concat(out).groupby(level=[0, 1]).max().reset_index()
    daily.to_csv(DAILY_REGISTER, index=False, compression="gzip")
    print("saved", DAILY_REGISTER, daily.shape)


if __name__ == "__main__":
    main()
