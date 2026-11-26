"""Day-ahead models (predict day t's max from data up to t-1): every member's model on one split, default vs tuned.

Split by whole years: train 2010-2022 | validation 2023 (tuning) | test 2024-01-01 .. last day (refit on train+val).
Models:  Random Forest (Chris), Gradient Boosting (Sullivan), KNN (Ada), XGBoost (Ryan), Ridge (linear baseline)
Benchmarks: persistence (yesterday's max), climatology (average max for the date in training years).
Bracket probabilities: Normal(prediction, sigma), sigma = the tuned model's validation residual SD.
Backtest: on Polymarket days, one decision per bracket at 00:10 on the day (10-minute prices).
"""
import itertools, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import suite_common as C

OUT = os.path.join(C.OUT, "daily")
os.makedirs(OUT, exist_ok=True)
Y = "target_max_temp_c"
# Lasso-selected columns in the group's cleaned file (Chris & Sullivan's original inputs)
CLEANED_22 = ["minT_lag1", "meanT_lag1", "minT_lag2", "cos_day", "maxT_ma3", "solar_lag1", "pres_lag2", "maxT_ma7", "evap_lag1",
              "sin_day", "maxT_change_1d", "pres_change_1d", "wind_speed_lag1", "rh_lag2", "wind_dir_cos_lag1", "rain_lag2",
              "rain_lag1", "year", "wind_dir_sin_lag1", "maxT_lag7", "rh_ma7", "pres_lag7"]

df = pd.read_csv(C.TRAINING_SET, parse_dates=["date"]).dropna(subset=[Y]).reset_index(drop=True)
FEATS = [c for c in df.columns if c not in ("date", Y)]
tr, va, te = df[df.date.dt.year <= 2022], df[df.date.dt.year == 2023], df[df.date.dt.year >= 2024]
trva = pd.concat([tr, va])
print(f"train {len(tr)} | val {len(va)} | test {len(te)} ({te.date.min():%Y-%m-%d} .. {te.date.max():%Y-%m-%d}) | {len(FEATS)} features")


def pipe(model, scale=False):
    steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else []) + [model]
    return make_pipeline(*steps)


# ---------------------------------------------------------------- model specs: default + grid
SPECS = {
    "rf": dict(default=(CLEANED_22, lambda: pipe(RandomForestRegressor(100, max_features=0.7, min_samples_leaf=3, random_state=42, n_jobs=-1))),
               default_note="Chris's settings: 100 trees, max_features 0.7, min_samples_leaf 3, 22 Lasso-selected features",
               grid=[dict(max_features=mf, min_samples_leaf=l, max_depth=d) for mf, l, d in itertools.product([0.33, 0.5, 0.7, 1.0], [1, 3, 5, 10, 20], [None, 12])],
               build=lambda p: pipe(RandomForestRegressor(300, random_state=42, n_jobs=-1, **p))),
    "gb": dict(default=(CLEANED_22, lambda: pipe(GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, min_samples_leaf=3, max_features=0.7, subsample=0.8, random_state=42))),
               default_note="Sullivan's settings: 100 trees, learning rate 0.05, depth 2, 22 Lasso-selected features",
               grid=[dict(max_depth=d, learning_rate=lr) for d, lr in itertools.product([2, 3, 4, 5], [0.02, 0.05, 0.1])],
               build=None),  # tuned with staged predictions below
    "knn": dict(default=(FEATS, lambda: pipe(KNeighborsRegressor(1), scale=True)),
                default_note="K = 1 (the single most similar past day), standardised features",
                grid=[dict(n_neighbors=k, weights=w, p=pp) for k, w, pp in itertools.product(
                    [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150], ["uniform", "distance"], [1, 2])],
                build=lambda p: pipe(KNeighborsRegressor(**p), scale=True)),
    "xgb": dict(default=(FEATS, lambda: pipe(XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42, n_jobs=2))),
                default_note="Ryan's daily prior settings: 400 trees, learning rate 0.05, depth 4",
                grid=[dict(max_depth=d, learning_rate=lr, min_child_weight=m) for d, lr, m in itertools.product([2, 3, 4, 6], [0.03, 0.1], [1, 5, 20])],
                build=None),  # tuned with early stopping below
    "ridge": dict(default=(FEATS, lambda: pipe(Ridge(1.0), scale=True)), default_note="alpha = 1",
                  grid=[dict(alpha=a) for a in [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000]],
                  build=lambda p: pipe(Ridge(**p), scale=True)),
}

log, results, preds = [], {}, pd.DataFrame({"date": te.date.values, "actual": te[Y].values})
val_preds = pd.DataFrame({"date": va.date.values, "actual": va[Y].values})
for key, s in SPECS.items():
    t0 = time.time()
    # default
    cols, mk = s["default"]
    m = mk().fit(tr[cols], tr[Y])
    d_tr, d_va = C.rmse(m.predict(tr[cols]), tr[Y]), C.rmse(m.predict(va[cols]), va[Y])
    m_def = mk().fit(trva[cols], trva[Y])
    preds[f"{key}_default"] = m_def.predict(te[cols])
    log.append(dict(model=C.OWNER[key], version="default", params=s["default_note"], train_rmse=d_tr, val_rmse=d_va, chosen=False))
    # tuning on the validation year
    best = None
    for p in s["grid"]:
        if key == "gb":
            g = pipe(GradientBoostingRegressor(n_estimators=1500, min_samples_leaf=3, max_features=0.7, subsample=0.8, random_state=42, **p)).fit(tr[FEATS], tr[Y])
            Xv = g[:-1].transform(va[FEATS])
            curve = [C.rmse(pv, va[Y]) for pv in g[-1].staged_predict(Xv)]
            n = int(np.argmin(curve)) + 1; p = {**p, "n_estimators": n}
            g = pipe(GradientBoostingRegressor(min_samples_leaf=3, max_features=0.7, subsample=0.8, random_state=42, **p)).fit(tr[FEATS], tr[Y])
        elif key == "xgb":
            imp = SimpleImputer(strategy="median").fit(tr[FEATS])
            x = XGBRegressor(n_estimators=3000, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2, early_stopping_rounds=100, **p)
            x.fit(imp.transform(tr[FEATS]), tr[Y], eval_set=[(imp.transform(va[FEATS]), va[Y])], verbose=False)
            p = {**p, "n_estimators": int(x.best_iteration) + 1}
            g = pipe(XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2, **p)).fit(tr[FEATS], tr[Y])
        else:
            g = s["build"](p).fit(tr[FEATS], tr[Y])
        r = dict(model=C.OWNER[key], version="tuning", params=str(p), train_rmse=C.rmse(g.predict(tr[FEATS]), tr[Y]),
                 val_rmse=C.rmse(g.predict(va[FEATS]), va[Y]), chosen=False)
        log.append(r)
        if best is None or r["val_rmse"] < best[0]["val_rmse"]:
            best = (r, p, g)
    r, p, g = best
    log[[i for i, x in enumerate(log) if x is r][0]]["chosen"] = True
    resid = g.predict(va[FEATS]) - va[Y].values
    sigma = float(np.std(resid))
    val_preds[key] = g.predict(va[FEATS])
    if key == "gb":
        final = pipe(GradientBoostingRegressor(min_samples_leaf=3, max_features=0.7, subsample=0.8, random_state=42, **p))
    elif key == "xgb":
        final = pipe(XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2, **p))
    else:
        final = s["build"](p)
    final.fit(trva[FEATS], trva[Y])
    preds[key] = final.predict(te[FEATS])
    pi = permutation_importance(g, va[FEATS], va[Y], scoring="neg_root_mean_squared_error", n_repeats=5, random_state=0, n_jobs=1)
    results[key] = dict(params=p, sigma=sigma, train_rmse=r["train_rmse"], val_rmse=r["val_rmse"], default_val=d_va, default_train=d_tr,
                        importance=pd.Series(pi.importances_mean, index=FEATS))
    print(f"{C.OWNER[key]:30s} default val {d_va:.3f} | tuned val {r['val_rmse']:.3f} (train {r['train_rmse']:.3f}) {p} | sigma {sigma:.3f} | {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- benchmarks
preds["persistence"] = te.maxT_lag1.values
doy = lambda d: d.dt.dayofyear.clip(upper=365)
clim = tr.groupby(doy(tr.date))[Y].mean()
clim = pd.concat([clim.iloc[-7:], clim, clim.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
clim.index = range(1, 366)
preds["climatology"] = clim.reindex(doy(te.date)).values
val_preds["persistence"], val_preds["climatology"] = va.maxT_lag1.values, clim.reindex(doy(va.date)).values
for b in ("persistence", "climatology"):
    results[b] = dict(sigma=float(np.std(val_preds[b] - val_preds.actual)))

# HKO forecast and Polymarket at 00:10 (from the hourly benchmark file), on the days they exist
bench = pd.read_csv(os.path.join(C.PROJECT_ROOT, "hourly_benchmark_code", "out", "hourly_benchmark_forecasts.csv"),
                    parse_dates=["date"])
h0 = bench[bench.hour == 0].set_index("date")
preds["hko"] = h0.hko_fcst.reindex(preds.date).values
preds["market"] = h0.market_ev.reindex(preds.date).values

# ---------------------------------------------------------------- scores
MODELS = ["rf", "gb", "knn", "xgb", "ridge"]
rows = []
for k in MODELS + ["persistence", "climatology"]:
    for ver in (["default", "tuned"] if k in MODELS else ["benchmark"]):
        col = f"{k}_default" if ver == "default" else k
        m = C.point_metrics(preds[col], preds.actual)
        sig = results[k]["sigma"]
        m["bracket_logloss"] = C.gauss_bracket_logloss(preds[col].values, sig, preds.actual.values)
        rows.append(dict(model=C.OWNER[k], version=ver, **m, val_rmse=results[k]["val_rmse"] if ver == "tuned" else
                         (results[k]["default_val"] if ver == "default" else C.rmse(val_preds[k], val_preds.actual)),
                         train_rmse=results[k]["train_rmse"] if ver == "tuned" else (results[k]["default_train"] if ver == "default" else np.nan)))
lb = pd.DataFrame(rows)
# same-days comparison with HKO and Polymarket
sub = preds.dropna(subset=["hko", "market"])
rows = []
for k in MODELS + ["persistence", "climatology", "hko", "market"]:
    rows.append(dict(model=C.OWNER[k], **C.point_metrics(sub[k], sub.actual)))
lb_mkt = pd.DataFrame(rows).sort_values("RMSE")
print(f"\nTest {te.date.min():%Y-%m-%d}..{te.date.max():%Y-%m-%d}:"); print(lb.round(3).to_string(index=False))
print(f"\nSame {len(sub)} days as HKO forecast + Polymarket ({sub.date.min():%Y-%m-%d}..{sub.date.max():%Y-%m-%d}):"); print(lb_mkt.round(3).to_string(index=False))

# ---------------------------------------------------------------- daily backtest at 00:10
mk, prices = C.load_markets(), C.load_prices()
bt_s, bt_first = [], []
pp = preds.set_index("date")
for k in MODELS:
    rows = []
    for d, br in mk[mk.obs_date.isin(pp.index)].groupby("obs_date"):
        p = C.gauss_bracket_probs(pp.loc[d, k], results[k]["sigma"], br.lo.values, br.hi.values)
        p = p / p.sum()
        rows += [dict(date=d, hour=0, time_hkt=d, bracket=b, model_p=float(x)) for b, x in zip(br.bracket, p)]
    _, first, s = C.backtest(pd.DataFrame(rows), prices, mk, 10, C.OWNER[k])
    bt_s.append(s); bt_first.append(first)
bt_s = pd.DataFrame(bt_s)
print("\nDaily backtest (one decision per bracket at 00:10):"); print(bt_s.round(3).to_string(index=False))

# ---------------------------------------------------------------- save
preds.to_csv(os.path.join(OUT, "daily_test_predictions.csv"), index=False)
val_preds.to_csv(os.path.join(OUT, "daily_val_predictions.csv"), index=False)
lb.to_csv(os.path.join(OUT, "daily_leaderboard.csv"), index=False)
lb_mkt.to_csv(os.path.join(OUT, "daily_leaderboard_vs_hko_market.csv"), index=False)
pd.DataFrame(log).to_csv(os.path.join(OUT, "daily_tuning_log.csv"), index=False)
bt_s.to_csv(os.path.join(OUT, "daily_backtest_summary.csv"), index=False)
pd.concat(bt_first).to_csv(os.path.join(OUT, "daily_backtest_trades.csv"), index=False)
pd.DataFrame({C.OWNER[k]: results[k]["importance"] for k in MODELS}).to_csv(os.path.join(OUT, "daily_permutation_importance.csv"))
pd.DataFrame([dict(model=C.OWNER[k], params=str(results[k].get("params", "")), sigma=results[k]["sigma"]) for k in MODELS + ["persistence", "climatology"]]).to_csv(
    os.path.join(OUT, "daily_final_settings.csv"), index=False)
print("saved", OUT)
