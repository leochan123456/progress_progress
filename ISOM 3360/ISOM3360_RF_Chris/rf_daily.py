"""Fit and evaluate Chris's day-ahead Random Forest on the shared train/validation/test split."""
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import rf_common as C

Y = "target_max_temp_c"
CLEANED_22 = [
    "minT_lag1", "meanT_lag1", "minT_lag2", "cos_day", "maxT_ma3", "solar_lag1",
    "pres_lag2", "maxT_ma7", "evap_lag1", "sin_day", "maxT_change_1d",
    "pres_change_1d", "wind_speed_lag1", "rh_lag2", "wind_dir_cos_lag1",
    "rain_lag2", "rain_lag1", "year", "wind_dir_sin_lag1", "maxT_lag7",
    "rh_ma7", "pres_lag7",
]


def build_model(params, n_estimators=300):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1, **params),
    )


def score(predicted, actual, sigma):
    metrics = C.SUITE.point_metrics(predicted, actual)
    metrics["bracket_logloss"] = C.SUITE.gauss_bracket_logloss(predicted, sigma, actual)
    return metrics


def main():
    C.ensure_files(C.TRAINING_SET)
    data = pd.read_csv(C.TRAINING_SET, parse_dates=["date"]).dropna(subset=[Y]).reset_index(drop=True)
    missing_features = sorted(set(CLEANED_22) - set(data.columns))
    if missing_features:
        raise ValueError(f"Training set is missing Chris's selected features: {missing_features}")

    features = [column for column in data.columns if column not in ("date", Y)]
    train = data[data.date.dt.year <= 2022]
    validation = data[data.date.dt.year == 2023]
    test = data[data.date.dt.year >= 2024]
    train_validation = pd.concat([train, validation])
    if train.empty or validation.empty or test.empty:
        raise ValueError("The training set must contain 2010-2022, 2023, and 2024+ rows.")

    default_model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=100, max_features=0.7, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
    )
    default_model.fit(train[CLEANED_22], train[Y])
    default_val = default_model.predict(validation[CLEANED_22])
    default_sigma = float(np.std(default_val - validation[Y].to_numpy()))
    default_test_model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=100, max_features=0.7, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
    ).fit(train_validation[CLEANED_22], train_validation[Y])
    default_test = default_test_model.predict(test[CLEANED_22])

    tuning_log = [{
        "version": "default",
        "params": "100 trees; max_features=0.7; min_samples_leaf=3; 22 selected features",
        "train_rmse": C.SUITE.rmse(default_model.predict(train[CLEANED_22]), train[Y]),
        "val_rmse": C.SUITE.rmse(default_val, validation[Y]),
        "chosen": False,
    }]
    grid = [
        dict(max_features=max_features, min_samples_leaf=leaf, max_depth=depth)
        for max_features, leaf, depth in itertools.product(
            [0.33, 0.5, 0.7, 1.0], [1, 3, 5, 10, 20], [None, 12]
        )
    ]
    best = None
    started = time.time()
    for params in grid:
        model = build_model(params).fit(train[features], train[Y])
        predicted = model.predict(validation[features])
        result = {
            "version": "tuning",
            "params": str(params),
            "train_rmse": C.SUITE.rmse(model.predict(train[features]), train[Y]),
            "val_rmse": C.SUITE.rmse(predicted, validation[Y]),
            "chosen": False,
        }
        tuning_log.append(result)
        if best is None or result["val_rmse"] < best[0]["val_rmse"]:
            best = (result, params, model, predicted)
    best_result, best_params, best_model, tuned_val = best
    best_result["chosen"] = True
    tuned_sigma = float(np.std(tuned_val - validation[Y].to_numpy()))
    tuned_test_model = build_model(best_params).fit(train_validation[features], train_validation[Y])
    tuned_test = tuned_test_model.predict(test[features])

    predictions = pd.DataFrame({
        "date": test.date.to_numpy(),
        "actual": test[Y].to_numpy(),
        "rf_default": default_test,
        "rf_tuned": tuned_test,
    })
    val_predictions = pd.DataFrame({
        "date": validation.date.to_numpy(),
        "actual": validation[Y].to_numpy(),
        "rf_default": default_val,
        "rf_tuned": tuned_val,
    })
    leaderboard = pd.DataFrame([
        {"model": "Random Forest (Chris)", "version": "default", **score(default_test, test[Y], default_sigma),
         "val_rmse": tuning_log[0]["val_rmse"], "train_rmse": tuning_log[0]["train_rmse"]},
        {"model": "Random Forest (Chris)", "version": "tuned", **score(tuned_test, test[Y], tuned_sigma),
         "val_rmse": best_result["val_rmse"], "train_rmse": best_result["train_rmse"]},
    ])

    os.makedirs(C.DAILY_OUT, exist_ok=True)
    predictions.to_csv(os.path.join(C.DAILY_OUT, "daily_test_predictions.csv"), index=False)
    val_predictions.to_csv(os.path.join(C.DAILY_OUT, "daily_val_predictions.csv"), index=False)
    leaderboard.to_csv(os.path.join(C.DAILY_OUT, "daily_leaderboard.csv"), index=False)
    pd.DataFrame(tuning_log).to_csv(os.path.join(C.DAILY_OUT, "daily_tuning_log.csv"), index=False)
    pd.DataFrame([
        {"model": "Random Forest (Chris)", "version": "default", "params": tuning_log[0]["params"], "sigma": default_sigma},
        {"model": "Random Forest (Chris)", "version": "tuned", "params": str(best_params), "sigma": tuned_sigma},
    ]).to_csv(os.path.join(C.DAILY_OUT, "daily_final_settings.csv"), index=False)

    print(f"train {len(train)} | validation {len(validation)} | test {len(test)} "
          f"({test.date.min():%Y-%m-%d} .. {test.date.max():%Y-%m-%d})")
    print(f"Chris RF tuned parameters: {best_params} | validation RMSE {best_result['val_rmse']:.3f} "
          f"| tuning time {time.time() - started:.0f}s")
    print(leaderboard.round(4).to_string(index=False))
    print("saved:", C.DAILY_OUT)


if __name__ == "__main__":
    main()
