"""Build the hourly + daily tables used by the XGBoost models.

Inputs:
  Storage/HKO/hourly/archive/hko_hourly_<element>.csv   (hko_hourly_scraper.py)
  Storage/HKO/processed/hko_daily_observations.csv      (hko_build_training_set.py)
  Storage/PM/cleaned/hko_daily_extract_cleaned.csv      (clean_pm_data.py)
Outputs:
  Storage/HKO/hourly/hourly_frame.csv   one row per hour (HKO HQ + King's Park + engineered hourly features)
  Storage/HKO/hourly/daily_table.csv    one row per day
"""
import argparse
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
import xgb_features as F  # noqa: E402

HOURLY_DIR = os.path.join(SCRIPT_DIR, "Storage", "HKO", "hourly")
DAILY_OBS = os.path.join(PROJECT_ROOT, "Storage", "HKO", "processed", "hko_daily_observations.csv")
EXTRACT = os.path.join(PROJECT_ROOT, "Storage", "PM", "cleaned", "hko_daily_extract_cleaned.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive-dir", default=os.path.join(HOURLY_DIR, "archive"))
    a = ap.parse_args()

    os.makedirs(HOURLY_DIR, exist_ok=True)
    long = {}
    for e in F.ELEMENTS:
        long[e] = F.normalise_long(pd.read_csv(os.path.join(a.archive_dir, f"hko_hourly_{e}.csv")))
        print(f"{e:22s} {len(long[e]):>10,} on-the-hour rows")
    hourly = F.build_hourly_frame(long)
    hourly.to_csv(os.path.join(HOURLY_DIR, "hourly_frame.csv"))

    extracts = [pd.read_csv(EXTRACT)] if os.path.exists(EXTRACT) else []
    daily = F.build_daily_table(DAILY_OBS, extracts)
    daily.index.name = "date"
    daily.to_csv(os.path.join(HOURLY_DIR, "daily_table.csv"))
    print(f"hourly_frame.csv: {len(hourly):,} hours over {hourly.date.nunique():,} days")
    print(f"daily_table.csv: {len(daily):,} days ({daily.index.min():%Y-%m-%d} to {daily.index.max():%Y-%m-%d})")


if __name__ == "__main__":
    main()
