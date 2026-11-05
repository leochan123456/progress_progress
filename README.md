# Hong Kong Temperature Prediction Market

## Overview

This project uses HKO weather data and Polymarket Hong Kong temperature markets to evaluate two Random Forest models:

- A regression model for the continuous daily maximum temperature.
- Binary cutoff models for probabilities such as `P(max temperature >= 31 C)`.

Polymarket contracts use one-degree brackets. For example, the `31 C` bracket represents temperatures from `31.0` through `31.9` C.

## Environment

The project uses this Python interpreter:

```bash
/home/codespace/.python/current/bin/python3
```

Required packages include `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, and `pypdf`.

Run all commands from the repository root:

```bash
cd /workspaces/progress_progress
```

## Run Order

### 1. Refresh HKO data

This downloads or reuses the cached HKO source files, builds lagged weather features, and writes the processed HKO tables.

```bash
/home/codespace/.python/current/bin/python3 "ISOM 3360/hko_build_training_set.py"
```

Outputs:

```text
ISOM 3360/Storage/HKO/raw_hko/
ISOM 3360/Storage/HKO/processed/hko_daily_observations.csv
ISOM 3360/Storage/HKO/processed/hko_training_set.csv
```

The default historical period is `2010-01-01` through `2026-09-30`. Custom dates can be supplied with `--start` and `--end`.

### 2. Clean the HKO training data

This performs correlation screening, VIF reporting, and Lasso feature selection.

```bash
/home/codespace/.python/current/bin/python3 "ISOM 3360/cleaning.py"
```

Output:

```text
ISOM 3360/Storage/HKO/cleaned/hko_training_set_cleaned.csv
```

### 3. Refresh Polymarket data

This downloads market metadata, hourly Yes prices, and HKO settlement data.

```bash
/home/codespace/.python/current/bin/python3 "ISOM 3360/polymarket_hk_temp_scraper.py"
```

Raw scraper outputs:

```text
ISOM 3360/Storage/PM/pm_markets.csv
ISOM 3360/Storage/PM/pm_price_history_hourly.csv
ISOM 3360/Storage/PM/hko_daily_extract.csv
```

### 4. Clean Polymarket data

This normalizes dates, timestamps, numeric fields, and prices; removes duplicate market keys; and keeps the original exports unchanged.

```bash
/home/codespace/.python/current/bin/python3 "ISOM 3360/clean_pm_data.py"
```

Outputs:

```text
ISOM 3360/Storage/PM/cleaned/pm_markets_cleaned.csv
ISOM 3360/Storage/PM/cleaned/pm_price_history_hourly_cleaned.csv
ISOM 3360/Storage/PM/cleaned/hko_daily_extract_cleaned.csv
```

### 5. Train both Random Forest models

This uses the cleaned HKO table. The time-based split is:

- Training: 2010 through 2022
- Validation: 2023
- Test: 2024 onward

```bash
/home/codespace/.python/current/bin/python3 "ISOM 3360/random_forest_models.py"
```

Results are written to:

```text
ISOM 3360/RESULT/rf_predictions.csv
ISOM 3360/RESULT/rf_metrics.csv
ISOM 3360/RESULT/rf_feature_importance.csv
```

The model currently trains cutoff classifiers from `8 C` through the highest observed temperature cutoff. Cutoff probabilities are made monotonic so a higher temperature cutoff cannot have a larger probability than a lower cutoff.

## Interpreting Results

The regression model is evaluated with RMSE and MAE. The cutoff models are evaluated with Brier score and log-loss. Lower values are better.

Polymarket prices are not model inputs. They are reserved for a later trading comparison and backtest, where model probabilities can be compared with market prices after fees and a minimum edge.

## Future Forecasting

The current Random Forest script evaluates dates that already have known target temperatures. It does not yet generate an unlabeled forecast after the latest HKO date.

To forecast a future date, the project needs:

1. New HKO observations through the day before the forecast date.
2. A prediction mode that builds features without requiring the future target.
3. A model retrained on all labelled historical data available at that time.

Missing future target values do not prevent forecasting, but future input weather observations are required for lagged and rolling features.

## Daily GitHub Action

The workflow at `.github/workflows/daily-data-update.yml` runs every day at `01:15 UTC`. It can also be started manually from the repository's **Actions** tab.

The daily workflow:

1. Installs the Python dependencies.
2. Refreshes Polymarket markets, hourly prices, and HKO settlement extracts.
3. Cleans the PM files.
4. Commits and pushes changed files under `ISOM 3360/Storage/PM`.

The workflow does not retrain the Random Forest every day. The HKO historical CSV source is updated less frequently, and model training requires completed target temperatures. When new labelled HKO data becomes available, run the HKO builder, HKO cleaner, and Random Forest script manually before publishing new model results.

For the scheduled workflow to push commits, the repository Actions setting **Workflow permissions** must allow read and write access to repository contents.