"""Clean and reduce the Hong Kong temperature training set.

The workflow mirrors cleaning-2.ipynb: inspect the data, screen features by
Pearson significance, report multicollinearity, and use cross-validated Lasso
to produce a smaller training set.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HKO_STORAGE = os.path.join(SCRIPT_DIR, "Storage", "HKO")
HKO_PROCESSED_STORAGE = os.path.join(HKO_STORAGE, "processed")
CLEANED_STORAGE = os.path.join(HKO_STORAGE, "cleaned")


def feature_correlations(df, features, target):
    results = []
    for column in features:
        values = df[column]
        mask = values.notna() & target.notna()
        x = values[mask]
        y = target[mask]
        if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
            correlation, p_value, t_stat = np.nan, np.nan, np.nan
        else:
            correlation, p_value = stats.pearsonr(x, y)
            t_stat = (correlation * np.sqrt((len(x) - 2) / (1 - correlation**2))
                      if abs(correlation) < 1 else np.inf)
        results.append({
            "feature": column,
            "corr": correlation,
            "t_stat": t_stat,
            "p_value": p_value,
            "significant": bool(pd.notna(p_value) and p_value < 0.05),
        })
    return pd.DataFrame(results).sort_values("p_value", na_position="last")


def print_vif(features):
    vif_input = add_constant(features, has_constant="add")
    vif = pd.DataFrame({
        "feature": vif_input.columns,
        "VIF": [variance_inflation_factor(vif_input.values, i)
                for i in range(vif_input.shape[1])],
    }).sort_values("VIF", ascending=False)
    print("\nVariance inflation factors:")
    print(vif.to_string(index=False))


def clean(input_path, output_path):
    df = pd.read_csv(input_path)
    required = {"date", "target_max_temp_c"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print("Shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nTotal missing values:", int(df.isna().sum().sum()))
    print("Date range:", df["date"].min(), "->", df["date"].max())
    print("\nSummary stats:")
    print(df.describe().T[["min", "max", "mean"]].round(2))

    target = df["target_max_temp_c"]
    features = [column for column in df.columns
                if column not in {"date", "target_max_temp_c"}]
    correlations = feature_correlations(df, features, target)
    print("\nFeature correlations:")
    print(correlations.round(4).to_string(index=False))

    significant = correlations.loc[correlations["significant"], "feature"].tolist()
    print("\nNumber of significant features:", len(significant))
    print("Insignificant features:",
          correlations.loc[~correlations["significant"], "feature"].tolist())

    model_data = df[significant].copy()
    print("X shape after dropping insignificant features:", model_data.shape)
    print_vif(model_data)

    lasso = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LassoCV(cv=5, random_state=42, max_iter=10000, n_jobs=-1)),
    ])
    lasso.fit(model_data, target)

    coefficients = pd.Series(
        lasso.named_steps["model"].coef_, index=model_data.columns
    )
    selected = coefficients[coefficients != 0].sort_values(
        key=abs, ascending=False
    )
    print("\nLasso-selected features (sorted by absolute coefficient):")
    print(selected.round(4).to_string())
    print("\nDropped by Lasso (coefficient = 0):",
          coefficients[coefficients == 0].index.tolist())
    print("Best alpha:", round(lasso.named_steps["model"].alpha_, 6))

    final_features = selected.index.tolist()
    cleaned = df[["date"] + final_features + ["target_max_temp_c"]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print("\nSaved:", output_path)
    print("Cleaned shape:", cleaned.shape)
    print("Final columns:", cleaned.columns.tolist())


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=os.path.join(HKO_PROCESSED_STORAGE, "hko_training_set.csv"))
    parser.add_argument("--output", default=os.path.join(CLEANED_STORAGE, "hko_training_set_cleaned.csv"))
    args = parser.parse_args()
    clean(args.input, args.output)