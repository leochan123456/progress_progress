# ISOM 3360: Hong Kong Temperature Prediction

This repository contains the original Random Forest baseline, the newer isolated Chris RF evaluation track, the four-member model suite, hourly benchmark code, and the supporting XGBoost project. The original `random_forest_models.py` and `polymarket_backtest.py` remain unchanged as baseline evidence.

## 60-second teacher fast track

From the repository root, with the environment activated, run the baseline and Chris RF scripts against the checked-in data:

```bash
python "ISOM 3360/random_forest_models.py"
python "ISOM 3360/polymarket_backtest.py"
python "ISOM 3360/ISOM3360_RF_Chris/rf_daily.py"
python "ISOM 3360/ISOM3360_RF_Chris/rf_walkforward.py"
python "ISOM 3360/ISOM3360_RF_Chris/rf_backtest.py"
```

The baseline scripts print model scores and buy/sell P&L. Chris's scripts generate daily, walk-forward, and backtest evidence; tuning and monthly refits can take longer than 60 seconds. They do not require the large hourly HKO archive.

## Environment

Python 3.12 is the tested runtime. From the repository root:

```bash
python -m venv .venv
```

Activate the environment for your shell:

```text
PowerShell: .\.venv\Scripts\Activate.ps1
macOS/Linux: source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

The `.venv` is local and ignored by Git.

## Transparent replication guide: Steps 1–10

Run commands from the repository root unless a step says otherwise.

### Step 1: Set up Python

Create and activate `.venv` and install `requirements.txt` using the commands above.

### Step 2: Refresh HKO daily data and build features

```bash
python "ISOM 3360/hko_build_training_set.py"
```

The builder caches 13 monthly HKO CSV series under `ISOM 3360/Storage/HKO/raw_hko/` and writes observations and lagged features under `ISOM 3360/Storage/HKO/processed/`. The checked monthly series end on 2026-10-31; the Daily Extract fills through 2026-11-25. Recent King's Park inputs stay blank until HKO publishes them.

### Step 3: Clean the training set

```bash
python "ISOM 3360/cleaning.py"
```

Correlations use available observations. VIF and Lasso use complete predictor rows, and rows with missing selected predictors are excluded from the cleaned modeling file.

### Step 4: Refresh Polymarket source files

This step requires network access.

```bash
python "ISOM 3360/polymarket_hk_temp_scraper.py"
```

### Step 5: Clean Polymarket markets, prices, and settlements

```bash
python "ISOM 3360/clean_pm_data.py"
```

### Step 6: Run the preserved Step 5a baseline model

```bash
python "ISOM 3360/random_forest_models.py"
```

This evaluates the continuous maximum-temperature model and 1-degree cutoff classifiers. Results are saved under `ISOM 3360/RESULT/`.

### Step 7: Run the preserved Step 6a baseline backtest

```bash
python "ISOM 3360/polymarket_backtest.py"
```

The backtest saves its trade ledger and summary under `ISOM 3360/RESULT/backtest/`. It counts repeated hourly quotes; its total P&L is not a capital-weighted portfolio return.

### Step 8: Run Chris's daily RF, walk-forward, and backtest

To refresh 10-minute prices (network required), then run the evaluations:

```bash
python "ISOM 3360/hourly_benchmark_code/fetch_pm_10min.py" "ISOM 3360/Storage/PM/cleaned/pm_markets_cleaned.csv" "ISOM 3360/Storage/PM/pm_price_10min.csv"
python "ISOM 3360/ISOM3360_RF_Chris/rf_daily.py"
python "ISOM 3360/ISOM3360_RF_Chris/rf_walkforward.py"
python "ISOM 3360/ISOM3360_RF_Chris/rf_backtest.py"
```

The split is train through 2022, validation 2023, and test 2024 onward. CSV evidence is saved under `ISOM 3360/ISOM3360_RF_Chris/out/`. The 10-minute price file occupies 43,615,659 bytes in Git (44,583,636 bytes in this Windows checkout). Walk-forward probabilities use the tuned validation sigma. The backtest uses the latest 10-minute price at or before 00:10; without that file it falls back to hourly prices with a 70-minute lag.

### Step 9: Prepare the hourly HKO dataset and run the hourly RF

Obtain `hourly_frame.csv` and `daily_table.csv` from the team's hourly HKO pipeline, or build them with the commands below. The multi-year archive is large and excluded from Git.

```bash
python "ISOM 3360/ISOM3360_XGBoost_for_repo/ISOM 3360/hko_hourly_scraper.py" 2020-07-01 2026-11-25 "ISOM 3360/ISOM3360_XGBoost_for_repo/ISOM 3360/Storage/HKO/hourly/archive"
python "ISOM 3360/ISOM3360_XGBoost_for_repo/ISOM 3360/xgb_build_hourly_dataset.py"
python "ISOM 3360/ISOM3360_RF_Chris/rf_hourly.py"
```

The derived tables can be tracked; do not commit the archive.

### Step 10: Run the full hourly benchmark and model-suite reports

After Step 9, fetch the full 20200101–20261123 bulletin range for `suite_upgrades.py` retraining. The checked-in `HK_Temp_Hourly_Benchmarks.xlsx` already stores parsed bulletins; reproducing that evidence alone does not require the download.

```bash
python "ISOM 3360/hourly_benchmark_code/fetch_flw.py" 20200101 20261123 "ISOM 3360/hourly_benchmark_code/flw_raw"
python "ISOM 3360/hourly_benchmark_code/fetch_pm_10min.py" "ISOM 3360/Storage/PM/cleaned/pm_markets_cleaned.csv" "ISOM 3360/hourly_benchmark_code/pm_price_10min.csv"
python "ISOM 3360/hourly_benchmark_code/benchmarks.py"
python "ISOM 3360/hourly_benchmark_code/score.py"
python "ISOM 3360/hourly_benchmark_code/report.py"
python "ISOM 3360/model_suite_code/suite_daily.py"
python "ISOM 3360/model_suite_code/suite_hourly.py"
python "ISOM 3360/model_suite_code/suite_upgrades.py"
python "ISOM 3360/model_suite_code/suite_excel.py"
```

The benchmark and full suite require the hourly frame, forecast bulletin archive, cleaned PM markets, and 10-minute prices. They currently read the price file from `hourly_benchmark_code/`, unlike Chris's backtest in Step 8.

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
| RF daily, original settings | 1,060 | 1.6980 | 1.3405 | 1.9616 | +4.58 |
| RF daily, validation-tuned | 1,060 | 1.7018 | 1.3506 | 1.9647 | +5.80 |
| RF monthly walk-forward | 1,060 | 1.7009 | 1.3475 | 1.9641 | +4.16 |

The live checkout's 1,060 test days include November 24–25, two more than the workbook's frozen 1,058-day snapshot ending 2026-11-23.

The baseline cleaned test covers 2024-01-01 through 2026-11-01. The isolated daily RF covers 2024-01-01 through 2026-11-25; its input pipeline imputes predictor gaps, while the baseline cleaner excludes 24 incomplete recent rows. Validation tuning did not improve the daily test RMSE.

The isolated RF backtest P&L is per share over 233 market days, using 10-minute prices at or before 00:10. The tuned model's 870 first-signal trades have a 95% bootstrap range of -13.37 to +25.42, so the positive total is not evidence of a reliable edge. The preserved Step 6a backtest reported +46.91 across 27,850 trades from 55,300 hourly quote rows at a 3-cent threshold; repeated quotes mean this is not portfolio return.

## Optional Hourly Extension

The team's hourly HKO pipeline provides `hourly_frame.csv` and `daily_table.csv` for running the hourly RF locally. For verified hourly RF results, see tabs B1–B5 in [HK_Temp_Model_Suite.xlsx](HK_Temp_Model_Suite.xlsx): 64.8% bracket hit rate, 1.003 log-loss, and +3.97 first-signal P&L.
