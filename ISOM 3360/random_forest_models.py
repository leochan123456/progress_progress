"""Train Random Forest temperature and 1-degree cutoff models."""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HKO_CLEANED = os.path.join(SCRIPT_DIR, "Storage", "HKO", "cleaned")
MODEL_OUTPUT = os.path.join(SCRIPT_DIR, "RESULT")

RANDOM_STATE = 42
VALIDATION_YEAR = 2023
TEST_START_YEAR = 2024
N_ESTIMATORS = 100


def positive_probability(model, features):
    probabilities = model.predict_proba(features)
    if 1 not in model.classes_:
        return np.zeros(len(features))
    return probabilities[:, list(model.classes_).index(1)]


def fit_predictions(train, predict, feature_columns, cutoffs):
    train_features = train[feature_columns]
    predict_features = predict[feature_columns]
    result = predict[["date", "target_max_temp_c"]].copy()

    regressor = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_features=0.7,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    regressor.fit(train_features, train["target_max_temp_c"])
    result["predicted_max_temp_c"] = regressor.predict(predict_features)
    importances = pd.DataFrame({
        "feature": feature_columns,
        "regression_importance": regressor.feature_importances_,
    })

    cutoff_importances = []
    for cutoff in cutoffs:
        labels = (train["target_max_temp_c"] >= cutoff).astype(int)
        classifier = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_features=0.7,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        classifier.fit(train_features, labels)
        result[f"p_ge_{cutoff}c_raw"] = positive_probability(classifier, predict_features)
        cutoff_importances.append(classifier.feature_importances_)

    raw_probability_columns = [f"p_ge_{cutoff}c_raw" for cutoff in cutoffs]
    monotonic = result[raw_probability_columns].to_numpy()
    monotonic = np.minimum.accumulate(monotonic, axis=1)
    for index, cutoff in enumerate(cutoffs):
        result[f"p_ge_{cutoff}c"] = monotonic[:, index]

    for index, cutoff in enumerate(cutoffs):
        next_probability = (
            monotonic[:, index + 1]
            if index + 1 < len(cutoffs)
            else np.zeros(len(result))
        )
        result[f"p_bracket_{cutoff}c"] = np.clip(monotonic[:, index] - next_probability, 0, 1)

    importances["classifier_importance_mean"] = np.mean(cutoff_importances, axis=0)
    return result, importances


def evaluate(predictions, cutoffs):
    rows = []
    for split, frame in predictions.groupby("split"):
        actual = frame["target_max_temp_c"]
        predicted = frame["predicted_max_temp_c"]
        rows.append({
            "split": split,
            "metric": "temperature_rmse",
            "target": "continuous_temperature",
            "value": np.sqrt(mean_squared_error(actual, predicted)),
        })
        rows.append({
            "split": split,
            "metric": "temperature_mae",
            "target": "continuous_temperature",
            "value": mean_absolute_error(actual, predicted),
        })
        for cutoff in cutoffs:
            labels = (actual >= cutoff).astype(int)
            probabilities = frame[f"p_ge_{cutoff}c"].clip(1e-6, 1 - 1e-6)
            rows.append({
                "split": split,
                "metric": "brier_score",
                "target": f"temperature_ge_{cutoff}c",
                "value": brier_score_loss(labels, probabilities),
            })
            rows.append({
                "split": split,
                "metric": "log_loss",
                "target": f"temperature_ge_{cutoff}c",
                "value": log_loss(labels, probabilities, labels=[0, 1]),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=os.path.join(HKO_CLEANED, "hko_training_set_cleaned.csv"),
    )
    parser.add_argument("--output-dir", default=MODEL_OUTPUT)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").reset_index(drop=True)
    feature_columns = [column for column in data.columns if column not in {"date", "target_max_temp_c"}]
    cutoffs = list(range(8, int(np.floor(data["target_max_temp_c"].max())) + 1))

    train = data[data.date.dt.year < VALIDATION_YEAR]
    validation = data[data.date.dt.year == VALIDATION_YEAR]
    train_through_validation = data[data.date.dt.year <= VALIDATION_YEAR]
    test = data[data.date.dt.year >= TEST_START_YEAR]

    validation_predictions, validation_importances = fit_predictions(
        train, validation, feature_columns, cutoffs
    )
    validation_predictions["split"] = "validation_2023"
    test_predictions, test_importances = fit_predictions(
        train_through_validation, test, feature_columns, cutoffs
    )
    test_predictions["split"] = "test_2024_onward"

    predictions = pd.concat([validation_predictions, test_predictions], ignore_index=True)
    metrics = evaluate(predictions, cutoffs)
    importances = validation_importances.rename(
        columns={
            "regression_importance": "validation_regression_importance",
            "classifier_importance_mean": "validation_classifier_importance_mean",
        }
    ).merge(
        test_importances.rename(
            columns={
                "regression_importance": "test_regression_importance",
                "classifier_importance_mean": "test_classifier_importance_mean",
            }
        ),
        on="feature",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    predictions.to_csv(os.path.join(args.output_dir, "rf_predictions.csv"), index=False)
    metrics.to_csv(os.path.join(args.output_dir, "rf_metrics.csv"), index=False)
    importances.to_csv(os.path.join(args.output_dir, "rf_feature_importance.csv"), index=False)

    print("training rows:", len(train))
    print("validation rows:", len(validation))
    print("test rows:", len(test))
    print("cutoffs:", cutoffs)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("saved directory:", args.output_dir)


if __name__ == "__main__":
    main()
