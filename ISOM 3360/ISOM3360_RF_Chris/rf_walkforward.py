"""Evaluate monthly walk-forward refits of Chris's day-ahead Random Forest."""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import rf_common as C

Y = "target_max_temp_c"


def main():
    static_path = os.path.join(C.DAILY_OUT, "daily_test_predictions.csv")
    settings_path = os.path.join(C.DAILY_OUT, "daily_final_settings.csv")
    C.ensure_files(C.TRAINING_SET, static_path, settings_path)
    data = pd.read_csv(C.TRAINING_SET, parse_dates=["date"]).dropna(subset=[Y]).reset_index(drop=True)
    features = [column for column in data.columns if column not in ("date", Y)]
    test = data[data.date.dt.year >= 2024].copy().reset_index(drop=True)
    if test.empty:
        raise ValueError("No 2024+ test rows are available for the walk-forward evaluation.")

    static = pd.read_csv(static_path, parse_dates=["date"])
    settings = pd.read_csv(settings_path)
    tuned_setting = settings[settings.version == "tuned"]
    if tuned_setting.empty:
        raise ValueError("The daily RF settings file has no tuned model sigma.")
    sigma = float(tuned_setting.iloc[0].sigma)

    params = dict(
        n_estimators=300, max_features=0.5, min_samples_leaf=1,
        random_state=42, n_jobs=-1,
    )
    walkforward = np.full(len(test), np.nan)
    months = pd.period_range(test.date.min(), test.date.max(), freq="M")
    for month in months:
        start = month.to_timestamp()
        history = data[data.date < start]
        mask = test.date.dt.to_period("M") == month
        if history.empty:
            raise ValueError(f"No historical training rows are available before {month}.")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(**params),
        ).fit(history[features], history[Y])
        walkforward[mask.to_numpy()] = model.predict(test.loc[mask, features])
        print(f"{month}: trained on {len(history)} days, predicted {int(mask.sum())} days")

    predictions = test[["date", Y]].rename(columns={Y: "actual"})
    predictions["rf_walkforward"] = walkforward
    predictions = predictions.merge(static[["date", "rf_tuned"]], on="date", how="left", validate="one_to_one")
    if predictions.rf_tuned.isna().any():
        raise ValueError("Some walk-forward test dates are missing static RF predictions.")

    rows = []
    for name, column in [("static tuned", "rf_tuned"), ("monthly walk-forward", "rf_walkforward")]:
        values = C.SUITE.point_metrics(predictions[column], predictions.actual)
        values["bracket_logloss"] = C.SUITE.gauss_bracket_logloss(predictions[column], sigma, predictions.actual)
        rows.append({"model": name, **values})
    metrics = pd.DataFrame(rows)

    annual = []
    predictions["year"] = predictions.date.dt.year
    for year, frame in predictions.groupby("year"):
        annual.append({
            "year": year,
            "days": len(frame),
            "static_rmse": C.SUITE.rmse(frame.rf_tuned, frame.actual),
            "walkforward_rmse": C.SUITE.rmse(frame.rf_walkforward, frame.actual),
            "static_bias": float((frame.rf_tuned - frame.actual).mean()),
            "walkforward_bias": float((frame.rf_walkforward - frame.actual).mean()),
        })

    os.makedirs(C.OUT, exist_ok=True)
    predictions.to_csv(os.path.join(C.OUT, "rf_walkforward_test_predictions.csv"), index=False)
    metrics.to_csv(os.path.join(C.OUT, "rf_walkforward_metrics.csv"), index=False)
    pd.DataFrame(annual).to_csv(os.path.join(C.OUT, "rf_walkforward_drift.csv"), index=False)
    print(f"\nWalk-forward refits: {len(months)} months")
    print(metrics.round(4).to_string(index=False))
    print("saved:", C.OUT)


if __name__ == "__main__":
    main()
