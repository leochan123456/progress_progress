# ISOM 3360: Hong Kong Temperature Prediction

This repository contains the original Random Forest baseline, the newer isolated Chris RF evaluation track, the four-member model suite, hourly benchmark code, and the supporting XGBoost project. The original `random_forest_models.py` and `polymarket_backtest.py` remain unchanged as baseline evidence.

## 60-second teacher fast track

From the repository root in PowerShell, with the environment already installed, run the two baseline scripts against the checked-in HKO and Polymarket data:

```powershell
.\.venv\Scripts\python.exe "ISOM 3360\random_forest_models.py"
.\.venv\Scripts\python.exe "ISOM 3360\polymarket_backtest.py"
```

The first command prints the test RMSE/MAE and cutoff Brier/log-loss scores. The second prints the buy/sell P&L summary. These commands do not refresh external data or require the large hourly HKO archive.

## Environment

Python 3.12 is the tested runtime. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The `.venv` is local and ignored by Git. In VS Code, select `.venv\Scripts\python.exe` as the workspace interpreter.

## Transparent replication guide: Steps 1–10

Run commands from the repository root unless a step says otherwise.

1. **Set up Python.** Create `.venv` and install `requirements.txt` using the commands above.

2. **Refresh HKO daily data and build features.**

   ```powershell
   .\.venv\Scripts\python.exe "ISOM 3360\hko_build_training_set.py"
   ```

   The builder caches the 13 monthly HKO CSV series under `ISOM 3360\Storage\HKO\raw_hko\` and writes the observations and lagged training set under `ISOM 3360\Storage\HKO\processed\`. The current checked monthly series end on 2026-10-31; the Daily Extract fills through 2026-11-25. Recent King's Park inputs stay blank until HKO publishes them.

3. **Clean the training set.**

   ```powershell
   .\.venv\Scripts\python.exe "ISOM 3360\cleaning.py"
   ```

   Correlations use available observations. VIF and Lasso use complete predictor rows, and rows with missing selected predictors are left out of the cleaned modeling file. This avoids imputing unpublished sunshine/wind data.

4. **Refresh Polymarket source files** (network required).

   ```powershell
   .\.venv\Scripts\python.exe "ISOM 3360\polymarket_hk_temp_scraper.py"
   ```

5. **Clean Polymarket markets, prices, and settlements.**

   ```powershell
   .\.venv\Scripts\python.exe "ISOM 3360\clean_pm_data.py"
   ```

6. **Run the preserved Step 5a baseline model.**

   ```powershell
   .\.venv\Scripts\python.exe "ISOM 3360\random_forest_models.py"
   ```

   This evaluates the continuous maximum-temperature model and 1-degree cutoff classifiers. Results are saved under `ISOM 3360\RESULT\`.

7. **Run the preserved Step 6a baseline backtest.**

   ```powershell
   .\.venv\Scripts\python.exe "ISOM 3360\polymarket_backtest.py"
   ```

   The backtest uses the configured fee and edge threshold and saves its trade ledger and summary under `ISOM 3360\RESULT\backtest\`. It counts repeated hourly quotes; its total P&L is not a capital-weighted portfolio return.

8. **Run Chris's isolated daily RF and monthly walk-forward evaluations.**

   ```powershell
   .\.venv\Scripts\python.exe "ISOM 3360\ISOM3360_RF_Chris\rf_daily.py"
   .\.venv\Scripts\python.exe "ISOM 3360\ISOM3360_RF_Chris\rf_walkforward.py"
   ```

   These use the shared year-based split (train through 2022, validation 2023, test 2024 onward). Outputs go to the ignored `ISOM 3360\ISOM3360_RF_Chris\out\` folder.

9. **Prepare the hourly HKO dataset, then run the hourly RF.**

   The hourly archive is deliberately not included in Git. It spans multiple years and can be large; the current checkout does not contain `hourly_frame.csv` or `daily_table.csv`, so `rf_hourly.py` cannot run until these are built.

   ```powershell
   .\.venv\Scripts\python.exe "ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\hko_hourly_scraper.py" 2020-07-01 2026-11-25 "ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\Storage\HKO\hourly\archive"
   .\.venv\Scripts\python.exe "ISOM 3360\ISOM3360_XGBoost_for_repo\ISOM 3360\xgb_build_hourly_dataset.py"
   .\.venv\Scripts\python.exe "ISOM 3360\ISOM3360_RF_Chris\rf_hourly.py"
   ```

   The archive, raw hourly snapshots, and derived hourly tables are ignored. The scraper is a long-running first-time download; do not commit its archive.

10. **Run the full hourly benchmark/model-suite reports** (after completing Step 9).

    ```powershell
    .\.venv\Scripts\python.exe "ISOM 3360\hourly_benchmark_code\fetch_flw.py" 20260301 20261114 "ISOM 3360\hourly_benchmark_code\flw_raw"
    .\.venv\Scripts\python.exe "ISOM 3360\hourly_benchmark_code\fetch_pm_10min.py" "ISOM 3360\Storage\PM\cleaned\pm_markets_cleaned.csv" "ISOM 3360\hourly_benchmark_code\pm_price_10min.csv"
    .\.venv\Scripts\python.exe "ISOM 3360\hourly_benchmark_code\benchmarks.py"
    .\.venv\Scripts\python.exe "ISOM 3360\hourly_benchmark_code\score.py"
    .\.venv\Scripts\python.exe "ISOM 3360\hourly_benchmark_code\report.py"
    .\.venv\Scripts\python.exe "ISOM 3360\model_suite_code\suite_daily.py"
    .\.venv\Scripts\python.exe "ISOM 3360\model_suite_code\suite_hourly.py"
    .\.venv\Scripts\python.exe "ISOM 3360\model_suite_code\suite_upgrades.py"
    .\.venv\Scripts\python.exe "ISOM 3360\model_suite_code\suite_excel.py"
    ```

    The benchmark and full suite require the hourly frame, HKO forecast bulletin archive, cleaned PM markets, and 10-minute prices. The 10-minute backtests use snapshots at the decision time; the RF-only backtest falls back to hourly prices with a 70-minute lag when no 10-minute file exists.

## Evidence map

| Evidence | Purpose | Location |
| --- | --- | --- |
| Step 5a baseline model | Original RF regression and cutoff-classifier implementation | `ISOM 3360/random_forest_models.py` |
| Step 6a baseline backtest | Original RF probability/market backtest | `ISOM 3360/polymarket_backtest.py` |
| Chris RF extraction | RF-only shared helpers, daily, walk-forward, hourly, and backtest scripts | `ISOM 3360/ISOM3360_RF_Chris/` |
| Full member model suite | Shared daily/hourly models and upgrades | `ISOM 3360/model_suite_code/` |
| Hourly benchmark | HKO, Polymarket, and XGBoost benchmark/report scripts | `ISOM 3360/hourly_benchmark_code/` |
| XGBoost project | Hourly features, scraper, model, and archived prediction artifacts | `ISOM 3360/ISOM3360_XGBoost_for_repo/` |
| Canonical daily data | HKO observations and lagged training set | `ISOM 3360/Storage/HKO/processed/` |
| Checked-in workbooks | Model-suite and hourly-benchmark results | `HK_Temp_Model_Suite.xlsx`, `HK_Temp_Hourly_Benchmarks.xlsx` |

All scripts derive project paths from `__file__`; no `/home/claude/...` paths are required. The baseline source files and `.github/workflows/daily-data-update.yml` are retained.

## Chris RF results from this checkout

| Run | Test rows/days | RMSE (°C) | MAE (°C) | Bracket log-loss | P&L |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step 5a baseline RF | 1,036 | 1.7024 | 1.3429 | Per-cutoff in `rf_metrics.csv` | — |
| RF daily, original settings | 1,060 | 1.6980 | 1.3405 | 1.9616 | +3.40 |
| RF daily, validation-tuned | 1,060 | 1.7018 | 1.3506 | 1.9647 | +3.75 |
| RF monthly walk-forward | 1,060 | 1.7009 | 1.3475 | 1.9641 | Not separately backtested |

The baseline cleaned test covers 2024-01-01 through 2026-11-01. The isolated daily RF covers 2024-01-01 through 2026-11-25; its input pipeline imputes predictor gaps, while the baseline cleaner excludes 24 incomplete recent rows. Validation tuning did not improve the daily test RMSE.

The isolated RF backtest P&L is per share over 232 market days (837 tuned first-signal trades) using hourly prices at a 70-minute lag: the more realistic 10-minute prices were not present in this checkout. Its 95% bootstrap range is -15.27 to +23.76, so the positive total is not evidence of a reliable edge. The preserved Step 6a backtest reported +46.91 across 27,850 trades from 55,300 hourly quote rows at a 3-cent threshold; repeated quotes mean this is not portfolio return.

The hourly RF smoke test is **blocked**, not passed: the required `hourly_frame.csv` and `daily_table.csv` are absent because the raw hourly HKO archive has not been downloaded. No substitute data was used.
