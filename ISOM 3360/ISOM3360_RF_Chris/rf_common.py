"""Shared paths and helpers for Chris's Random Forest replication track."""
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_SUITE_DIR = os.path.join(PROJECT_ROOT, "model_suite_code")
sys.path.insert(0, MODEL_SUITE_DIR)

import suite_common as SUITE  # noqa: E402

XGB_REPO = os.path.join(PROJECT_ROOT, "ISOM3360_XGBoost_for_repo", "ISOM 3360")
TRAINING_SET = os.path.join(PROJECT_ROOT, "Storage", "HKO", "processed", "hko_training_set.csv")
PM_PRICE_10MIN = os.path.join(PROJECT_ROOT, "Storage", "PM", "pm_price_10min.csv")
PM_PRICE_10MIN_LEGACY = SUITE.PM_PRICES_10MIN
PM_PRICE_HOURLY = os.path.join(PROJECT_ROOT, "Storage", "PM", "cleaned", "pm_price_history_hourly_cleaned.csv")
OUT = os.path.join(SCRIPT_DIR, "out")
DAILY_OUT = os.path.join(OUT, "daily")
HOURLY_OUT = os.path.join(OUT, "hourly")
BACKTEST_OUT = os.path.join(OUT, "backtest")


def load_prices():
    for path in (PM_PRICE_10MIN, PM_PRICE_10MIN_LEGACY):
        if os.path.isfile(path):
            prices = pd.read_csv(path, parse_dates=["obs_date", "time_hkt"])
            return prices.rename(columns={"obs_date": "date"}), 10, "10-minute"
    if os.path.isfile(PM_PRICE_HOURLY):
        prices = pd.read_csv(PM_PRICE_HOURLY, parse_dates=["obs_date", "time_hkt"])
        prices = prices.rename(columns={"obs_date": "date"})
        return prices, 70, "hourly (70-minute lag)"
    raise FileNotFoundError(
        "No Polymarket price history found. Expected either "
        f"{PM_PRICE_10MIN} or {PM_PRICE_HOURLY}."
    )


def ensure_files(*paths):
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Required input file(s) not found:\n" + "\n".join(missing))
