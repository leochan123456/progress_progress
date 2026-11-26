"""Fit and evaluate Chris's within-day Random Forest classifier."""
import itertools
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import rf_common as C

import sys
sys.path.insert(0, C.XGB_REPO)
import xgb_features as F  # noqa: E402

HOURLY_DIR = os.path.join(C.XGB_REPO, "Storage", "HKO", "hourly")
HOURLY_FRAME = os.path.join(HOURLY_DIR, "hourly_frame.csv")
DAILY_TABLE = os.path.join(HOURLY_DIR, "daily_table.csv")
CLASS_COUNT = F.CAP + 1


def full_proba(model, features):
    probabilities = np.zeros((len(features), CLASS_COUNT))
    probabilities[:, model.classes_.astype(int)] = model.predict_proba(features)
    return probabilities


def logloss(probabilities, outcomes):
    return float(-np.mean(np.log(np.clip(probabilities[np.arange(len(outcomes)), outcomes], 1e-6, 1))))


def make_model(params, n_estimators=200):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1, **params),
    )


def main():
    C.ensure_files(HOURLY_FRAME, DAILY_TABLE)
    hourly = pd.read_csv(HOURLY_FRAME, index_col=0, parse_dates=[0])
    hourly["date"] = pd.to_datetime(hourly["date"])
    daily = pd.read_csv(DAILY_TABLE, index_col=0, parse_dates=[0])
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))

    prior_features = F.prior_features(daily)
    prior_dates = prior_features.index[
        (prior_features.index.year >= 2010) & (prior_features.index <= "2019-12-31")
    ]
    valid_prior = (
        prior_features.loc[prior_dates].notna().all(axis=1)
        & daily.loc[prior_dates, "max_temp_c"].notna()
    )
    from xgboost import XGBRegressor
    prior_model = XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, random_state=42, n_jobs=4,
    )
    prior_model.fit(
        prior_features.loc[prior_dates][valid_prior],
        daily.loc[prior_dates, "max_temp_c"][valid_prior],
    )
    prior_mean = pd.Series(prior_model.predict(prior_features), index=prior_features.index)
    data = F.assemble(hourly, F.yesterday_features(daily), prior_mean)
    data = data.dropna(subset=["temp", "rmax", "final_max"])
    data["d_raw"] = np.floor(data.final_max) - np.floor(data.rmax)
    data = data[data.d_raw >= 0].copy()
    data["target"] = data.d_raw.clip(upper=F.CAP).astype(int)

    train = data[data.date <= "2026-06-30"]
    validation = data[(data.date > "2026-06-30") & (data.date < "2026-09-01")]
    test = data[data.date >= "2026-09-01"].reset_index(drop=True)
    train_validation = pd.concat([train, validation])
    if train.empty or validation.empty or test.empty:
        raise ValueError("Hourly data must include train through 2026-06-30, validation Jul-Aug 2026, and test from 2026-09-01.")
    features = F.HOURLY_FEATURES
    outcomes = validation.target.to_numpy()

    default_model = make_model(
        dict(max_features=0.7, min_samples_leaf=3, class_weight="balanced_subsample"),
        n_estimators=100,
    ).fit(train[features], train.target)
    default_val = full_proba(default_model, validation[features])

    started = time.time()
    tuning_log = []
    best = None
    grid = [
        dict(n_estimators=200, max_features=max_features, min_samples_leaf=leaf)
        for max_features, leaf in itertools.product(["sqrt", 0.5], [5, 20, 60])
    ]
    for params in grid:
        model = make_model({k: v for k, v in params.items() if k != "n_estimators"},
                           n_estimators=params["n_estimators"]).fit(train[features], train.target)
        validation_probabilities = full_proba(model, validation[features])
        result = {
            "params": str(params),
            "val_logloss": logloss(validation_probabilities, outcomes),
            "chosen": False,
        }
        tuning_log.append(result)
        if best is None or result["val_logloss"] < best[0]["val_logloss"]:
            best = (result, params, model)
        print(f"{params}: validation log-loss {result['val_logloss']:.4f}")
    best_result, best_params, best_model = best
    best_result["chosen"] = True
    tuned_val = full_proba(best_model, validation[features])
    final_model = make_model(
        {k: v for k, v in best_params.items() if k != "n_estimators"},
        n_estimators=best_params["n_estimators"],
    ).fit(train_validation[features], train_validation.target)
    tuned_test = full_proba(final_model, test[features])

    default_final = make_model(
        dict(max_features=0.7, min_samples_leaf=3, class_weight="balanced_subsample"),
        n_estimators=100,
    ).fit(train_validation[features], train_validation.target)
    default_test = full_proba(default_final, test[features])

    predictions = test[["date", "hour", "time", "rmax", "final_max"]].copy()
    predictions["rf_default"] = np.floor(test.rmax) + (default_test * np.arange(CLASS_COUNT)).sum(axis=1) + 0.5
    predictions["rf"] = np.floor(test.rmax) + (tuned_test * np.arange(CLASS_COUNT)).sum(axis=1) + 0.5
    metrics = []
    for name, probabilities, point in [
        ("default", default_test, predictions.rf_default),
        ("tuned", tuned_test, predictions.rf),
    ]:
        point_metrics = C.SUITE.point_metrics(point, test.final_max)
        metrics.append({
            "version": name,
            "test_logloss": logloss(probabilities, test.target.to_numpy()),
            "val_logloss": logloss(default_val, outcomes) if name == "default" else best_result["val_logloss"],
            **point_metrics,
        })

    markets = C.SUITE.load_markets()
    probability_rows = []
    for index, row in test.iterrows():
        brackets = markets[markets.obs_date == row.date]
        if brackets.empty:
            continue
        mapped = F.bracket_probs(tuned_test[index], row.rmax, brackets[["bracket", "lo", "hi"]])
        for bracket, probability in mapped.items():
            probability_rows.append({
                "date": row.date, "hour": int(row.hour), "time_hkt": row.time,
                "bracket": bracket, "model_p": float(probability),
            })

    os.makedirs(C.HOURLY_OUT, exist_ok=True)
    predictions.to_csv(os.path.join(C.HOURLY_OUT, "hourly_test_predicted_max.csv"), index=False)
    pd.DataFrame(metrics).to_csv(os.path.join(C.HOURLY_OUT, "hourly_metrics.csv"), index=False)
    pd.DataFrame(tuning_log).to_csv(os.path.join(C.HOURLY_OUT, "hourly_tuning_log.csv"), index=False)
    pd.DataFrame(probability_rows).to_csv(os.path.join(C.HOURLY_OUT, "hourly_bracket_probs_rf.csv"), index=False)
    pd.DataFrame([{
        "version": "tuned", "params": str(best_params), "val_logloss": best_result["val_logloss"],
        "tuning_seconds": time.time() - started,
    }]).to_csv(os.path.join(C.HOURLY_OUT, "hourly_final_settings.csv"), index=False)

    print(f"train {train.date.nunique()} days / {len(train)} rows | "
          f"validation {validation.date.nunique()} days | test {test.date.nunique()} days")
    print(f"chosen parameters: {best_params}")
    print(pd.DataFrame(metrics).round(4).to_string(index=False))
    print("saved:", C.HOURLY_OUT)


if __name__ == "__main__":
    main()
