# XGBoost hourly model

This copied project contains an hourly XGBoost bracket model. It predicts each Polymarket bracket probability from HKO observations available during the day; a separate script backtests those predictions.

All commands below are run from the repository root in PowerShell. The project's Python modules are nested at `ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\`. The project resolves its own outputs relative to those modules; it reads the canonical daily HKO and Polymarket sources from the main `ISOM 3360\Storage\` folder.

## 1. Download hourly HKO data

The hourly historical archive starts in mid-2020. A multi-year first download takes substantial time and disk space; the archive is ignored by Git. This workspace does not currently contain the archive or the derived hourly tables.

```powershell
.\.venv\Scripts\python.exe "ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\hko_hourly_scraper.py" 2020-07-01 2026-11-25 "ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\Storage\HKO\hourly\archive"
```

The scraper writes six `hko_hourly_<element>.csv` files into the archive directory and keeps its downloaded snapshots under its `raw\` subfolder.

## 2. Build the hourly and daily tables

```powershell
.\.venv\Scripts\python.exe "ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\xgb_build_hourly_dataset.py"
```

The script reads the hourly archive plus `ISOM 3360\Storage\HKO\processed\hko_daily_observations.csv` and, when available, `ISOM 3360\Storage\PM\cleaned\hko_daily_extract_cleaned.csv`. It writes:

```text
ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\Storage\HKO\hourly\hourly_frame.csv
ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\Storage\HKO\hourly\daily_table.csv
```

## 3. Train XGBoost and write predictions/metrics

```powershell
.\.venv\Scripts\python.exe "ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\xgb_models.py"
```

- **Daily prior:** regression predicts the day's maximum from data available through the previous day.
- **Hourly model:** predicts `d = floor(final max) - floor(running max)` for classes 0 through 8+, which maps to the 1 °C brackets.
- The default time split trains through 2026-06-30, validates July–August 2026, and tests from 2026-09-01.
- No Polymarket prices are used during model fitting.

Outputs are saved under `ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\RESULT\xgboost\`, including predictions, metrics, tuning logs, feature importance, and fitted models.

## 4. Run the buy-point backtest

For a 10-minute price history:

```powershell
.\.venv\Scripts\python.exe "ISOM 3360\hourly_benchmark_code\fetch_pm_10min.py" "ISOM 3360\Storage\PM\cleaned\pm_markets_cleaned.csv" "ISOM 3360\Storage\PM\cleaned\pm_price_10min.csv"
.\.venv\Scripts\python.exe "ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\xgb_backtest.py" --prices "ISOM 3360\Storage\PM\cleaned\pm_price_10min.csv" --lag-minutes 10
```

Or, with the checked hourly prices:

```powershell
.\.venv\Scripts\python.exe "ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\xgb_backtest.py"
```

The default 70-minute lag uses the next hourly snapshot after HKO publishes the hourly reading. The 10-minute version uses a 10-minute lag. Backtest outputs are under the nested project's `RESULT\xgboost\backtest\` directory.

## Previously reported results

The copied project previously reported results for 2026-09-01 through 2026-11-04 (58 days): test log-loss 0.943 versus 1.025 for its per-hour baseline; hourly daily-maximum RMSE 0.865 °C; and first-signal P&L +6.80 over 403 trades using 10-minute prices, with a bootstrap 95% range of -3.36 to +18.30. The one-hour-late hourly-price run reported +0.37 over 402 trades, with a 95% range of -8.34 to +9.64.

These are historical reported figures, not a result reproduced in the current checkout. Reproducing them requires the hourly archive and derived tables, which are not present.
