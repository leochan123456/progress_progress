"""Train the XGBoost models and write predictions, metrics and feature importance (no market prices used).

Model 1  Daily prior (XGBoost regression): day t's max temperature from data up to t-1. Trained 2010 .. --prior-end.
Model 2  Hourly bracket model (XGBoost multi-class). At hour h, HKO's running max since midnight is r and the final
         max can only be >= r, so it predicts d = floor(final max) - floor(r), classes 0..8 (8 = "8 or more").
         Bracket probability = P(d) for bracket floor(r) + d; brackets below the running max get 0.

Time-based split by whole days: train <= --train-end, validation (tuning) until --test-start, test from --test-start.

Inputs : Storage/HKO/hourly/hourly_frame.csv, Storage/HKO/hourly/daily_table.csv (xgb_build_hourly_dataset.py)
         Storage/PM/cleaned/pm_markets_cleaned.csv  (bracket definitions only, to label the predictions)
Outputs: RESULT/xgboost/xgb_predictions.csv          test date x hour x bracket: model probability (used by xgb_backtest.py)
         RESULT/xgboost/xgb_hourly_temperature.csv   test date x hour: predicted final max vs actual
         RESULT/xgboost/xgb_metrics.csv              split, metric, target, value
         RESULT/xgboost/xgb_feature_importance.csv
         RESULT/xgboost/xgb_tuning_log.csv
         RESULT/xgboost/model/  prior_xgb.json, intraday_xgb.json, model_meta.json
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
import xgb_features as F  # noqa: E402

HOURLY_DIR = os.path.join(SCRIPT_DIR, "Storage", "HKO", "hourly")
PM_MARKETS = os.path.join(PROJECT_ROOT, "Storage", "PM", "cleaned", "pm_markets_cleaned.csv")
OUT = os.path.join(SCRIPT_DIR, "RESULT", "xgboost")
EXCLUDED_DATES = {"2026-03-13", "2026-03-14", "2026-05-18", "2026-05-19"}
K = F.CAP + 1


def mlogloss(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1))))


def mbrier(P, y):
    Y = np.zeros_like(P); Y[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def fit_classifier(df, params, n_trees, eval_df=None):
    X, y, w = df[F.HOURLY_FEATURES], df.y.to_numpy(), np.ones(len(df))
    missing = sorted(set(range(K)) - set(y))  # XGBoost needs every class present: tiny-weight dummy rows
    if missing:
        X = pd.concat([X, X.iloc[[0] * len(missing)]]); y = np.r_[y, missing]; w = np.r_[w, [1e-6] * len(missing)]
    m = XGBClassifier(objective="multi:softprob", num_class=K, n_estimators=n_trees, tree_method="hist",
                      eval_metric="mlogloss", subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4, **params)
    if eval_df is not None:
        m.set_params(early_stopping_rounds=50)
        m.fit(X, y, sample_weight=w, eval_set=[(eval_df[F.HOURLY_FEATURES], eval_df.y)], verbose=False)
    else:
        m.fit(X, y, sample_weight=w, verbose=False)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-end", default="2026-06-30")
    ap.add_argument("--test-start", default="2026-09-01")
    ap.add_argument("--prior-end", default="2019-12-31", help="daily prior trained up to here (before the hourly training days)")
    ap.add_argument("--grid", default="small", choices=["small", "full"])
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "model"), exist_ok=True)

    hourly = pd.read_csv(os.path.join(HOURLY_DIR, "hourly_frame.csv"), index_col=0, parse_dates=[0])
    hourly["date"] = pd.to_datetime(hourly["date"])
    daily = pd.read_csv(os.path.join(HOURLY_DIR, "daily_table.csv"), index_col=0, parse_dates=[0])
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))

    # ---- model 1: daily prior ----
    P0 = F.prior_features(daily)
    rows = P0.index[(P0.index.year >= 2010) & (P0.index <= a.prior_end)]
    ok = P0.loc[rows].notna().all(axis=1) & daily.loc[rows, "max_temp_c"].notna()
    prior = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, random_state=42, n_jobs=4)
    prior.fit(P0.loc[rows][ok], daily.loc[rows, "max_temp_c"][ok])
    prior_mu = pd.Series(prior.predict(P0), index=P0.index)
    chk = pd.concat([prior_mu.rename("p"), daily.max_temp_c], axis=1).loc["2024-01-01":].dropna()
    prior_rmse = float(np.sqrt(np.mean((chk.p - chk.max_temp_c) ** 2)))
    prior_mae = float(np.mean(np.abs(chk.p - chk.max_temp_c)))

    # ---- model 2: hourly bracket classifier ----
    D = F.assemble(hourly, F.yesterday_features(daily), prior_mu).dropna(subset=["temp", "rmax", "final_max"])
    D["d_raw"] = np.floor(D.final_max) - np.floor(D.rmax)
    D = D[D.d_raw >= 0].copy()
    D["y"] = D.d_raw.clip(upper=F.CAP).astype(int)
    tr = D[D.date <= a.train_end]; va = D[(D.date > a.train_end) & (D.date < a.test_start)]
    te = D[D.date >= a.test_start].reset_index(drop=True)

    grid = (itertools.product([2, 3, 4, 6], [0.03, 0.1], [1, 10]) if a.grid == "full"
            else itertools.product([3, 5, 7], [0.1], [10]))
    log = []
    for depth, lr, mcw in grid:
        p = dict(max_depth=depth, learning_rate=lr, min_child_weight=mcw, reg_lambda=1.0)
        m = fit_classifier(tr, p, 1500, eval_df=va)
        log.append({**p, "best_trees": int(m.best_iteration) + 1,
                    "val_logloss": mlogloss(m.predict_proba(va[F.HOURLY_FEATURES]), va.y.to_numpy())})
    log = pd.DataFrame(log).sort_values("val_logloss"); log.to_csv(os.path.join(OUT, "xgb_tuning_log.csv"), index=False)
    best = log.iloc[0]
    params = dict(max_depth=int(best.max_depth), learning_rate=float(best.learning_rate),
                  min_child_weight=float(best.min_child_weight), reg_lambda=1.0)
    trva = pd.concat([tr, va])
    model = fit_classifier(trva, params, int(best.best_trees))

    # ---- test-set scores ----
    P = model.predict_proba(te[F.HOURLY_FEATURES]); y = te.y.to_numpy()
    tab = pd.crosstab(trva.hour, trva.y).reindex(index=range(24), columns=range(K), fill_value=0) + 0.5
    Pc = (tab.T / tab.sum(axis=1)).T.loc[te.hour].to_numpy()  # baseline: average outcome for each hour
    te["pred_final_max"] = np.floor(te.rmax) + (P * np.arange(K)).sum(1) + 0.5
    split = f"test_{a.test_start[:7]}_onward"
    metrics = [(split, "log_loss", "hourly_bracket_d", mlogloss(P, y)), (split, "brier_score", "hourly_bracket_d", mbrier(P, y)),
               (split, "log_loss", "baseline_per_hour", mlogloss(Pc, y)), (split, "brier_score", "baseline_per_hour", mbrier(Pc, y)),
               (split, "temperature_rmse", "hourly_predicted_final_max", float(np.sqrt(np.mean((te.pred_final_max - te.final_max) ** 2)))),
               ("test_2024_onward", "temperature_rmse", "daily_prior", prior_rmse), ("test_2024_onward", "temperature_mae", "daily_prior", prior_mae)]
    for h in range(24):
        mk = te.hour.to_numpy() == h
        if mk.any():
            metrics += [(split, "log_loss", f"hour_{h:02d}", mlogloss(P[mk], y[mk])), (split, "log_loss", f"baseline_hour_{h:02d}", mlogloss(Pc[mk], y[mk]))]
    pd.DataFrame(metrics, columns=["split", "metric", "target", "value"]).to_csv(os.path.join(OUT, "xgb_metrics.csv"), index=False)

    # ---- predictions per bracket (input for xgb_backtest.py) ----
    mk = pd.read_csv(PM_MARKETS, parse_dates=["obs_date"])
    mk = mk[~mk.event_slug.str.startswith("arch-") & ~mk.obs_date.dt.strftime("%Y-%m-%d").isin(EXCLUDED_DATES)].copy()
    mk["lo"] = pd.to_numeric(mk.bracket_low, errors="coerce").fillna(-np.inf)
    mk["hi"] = pd.to_numeric(mk.bracket_high, errors="coerce").fillna(np.inf)
    out = []
    for i, r in te.iterrows():
        br = mk[mk.obs_date == r.date]
        if br.empty: continue
        probs = F.bracket_probs(P[i], r.rmax, br[["bracket", "lo", "hi"]])
        for b, p in probs.items():
            out.append(dict(date=r.date.date(), hour=int(r.hour), time_hkt=r.time, temp_now=r.temp, running_max=r.rmax,
                            bracket=b, model_p=round(float(p), 5), actual_max=r.final_max))
    pd.DataFrame(out).to_csv(os.path.join(OUT, "xgb_predictions.csv"), index=False)
    te[["date", "hour", "temp", "rmax", "prior_mu", "pred_final_max", "final_max"]].round(3).to_csv(
        os.path.join(OUT, "xgb_hourly_temperature.csv"), index=False)

    gain = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(F.HOURLY_FEATURES).fillna(0)
    gain.sort_values(ascending=False).rename("gain").rename_axis("feature").to_csv(os.path.join(OUT, "xgb_feature_importance.csv"))
    prior.save_model(os.path.join(OUT, "model", "prior_xgb.json"))
    model.save_model(os.path.join(OUT, "model", "intraday_xgb.json"))
    json.dump(dict(prior_end=a.prior_end, train_start=str(tr.date.min().date()), train_end=a.train_end, test_start=a.test_start,
                   train_days=int(tr.date.nunique()), val_days=int(va.date.nunique()), test_days=int(te.date.nunique()),
                   params=params, trees=int(best.best_trees), cap=F.CAP, hourly_features=F.HOURLY_FEATURES,
                   prior_features=list(P0.columns)), open(os.path.join(OUT, "model", "model_meta.json"), "w"), indent=1)

    print(f"daily prior RMSE from 2024: {prior_rmse:.3f} C")
    print(f"train {tr.date.nunique()} days | validation {va.date.nunique()} | test {te.date.nunique()}  params {params}, {int(best.best_trees)} trees")
    print(f"test log-loss {mlogloss(P, y):.3f} vs baseline {mlogloss(Pc, y):.3f} | Brier {mbrier(P, y):.3f} vs {mbrier(Pc, y):.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
