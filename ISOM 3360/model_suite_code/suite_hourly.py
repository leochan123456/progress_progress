"""Hourly (within-day) versions of every member's model, on the same dataset and target as the XGBoost hourly model.

At hour h: running max since midnight r is known; predict d = floor(final max) - floor(r), classes 0..8 (8 = '8 or more').
Split by whole days: train 1 Jul 2020 - 30 Jun 2026 | validation Jul-Aug 2026 (tuning) | test 1 Sep - 4 Nov 2026.
Models: Random Forest (Chris), Gradient Boosting (Sullivan; sklearn's histogram GB, the fast version for 50k rows),
        KNN (Ada), XGBoost (Ryan, the saved model). Default settings vs tuned on validation log-loss.
Outputs: per-hour predicted final max (for RMSE/SSE by hour), bracket probabilities, backtest at h:10.
"""
import itertools, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
import suite_common as C
F = C.F

OUT = os.path.join(C.OUT, "hourly")
os.makedirs(OUT, exist_ok=True)
HD = os.path.join(C.REPO, "Storage", "HKO", "hourly")
K = F.CAP + 1
FEATS = F.HOURLY_FEATURES

# ---------------------------------------------------------------- dataset (identical to xgb_models.py)
hourly = pd.read_csv(os.path.join(HD, "hourly_frame.csv"), index_col=0, parse_dates=[0]); hourly["date"] = pd.to_datetime(hourly["date"])
daily = pd.read_csv(os.path.join(HD, "daily_table.csv"), index_col=0, parse_dates=[0]); daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
P0 = F.prior_features(daily)
rows = P0.index[(P0.index.year >= 2010) & (P0.index <= "2019-12-31")]
ok = P0.loc[rows].notna().all(axis=1) & daily.loc[rows, "max_temp_c"].notna()
prior = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42, n_jobs=4)
prior.fit(P0.loc[rows][ok], daily.loc[rows, "max_temp_c"][ok])
prior_mu = pd.Series(prior.predict(P0), index=P0.index)
D = F.assemble(hourly, F.yesterday_features(daily), prior_mu).dropna(subset=["temp", "rmax", "final_max"])
D["d_raw"] = np.floor(D.final_max) - np.floor(D.rmax)
D = D[D.d_raw >= 0].copy(); D["y"] = D.d_raw.clip(upper=F.CAP).astype(int)
tr = D[D.date <= "2026-06-30"]; va = D[(D.date > "2026-06-30") & (D.date < "2026-09-01")]; te = D[D.date >= "2026-09-01"].reset_index(drop=True)
trva = pd.concat([tr, va])
print(f"train {tr.date.nunique()} days / {len(tr)} rows | val {va.date.nunique()} | test {te.date.nunique()} days / {len(te)} rows")


def mlogloss(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1))))


def full_proba(model, X):
    """predict_proba with a column for every class 0..K-1 (classes missing from training get 0)."""
    P = np.zeros((len(X), K)); P[:, model.classes_] = model.predict_proba(X)
    return P


def rf(**p): return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(random_state=42, n_jobs=-1, **p))
def knn(**p): return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(**p))


SPECS = {
    "rf": dict(default=("Chris's settings: 100 trees, max_features 0.7, min_samples_leaf 3, class_weight balanced_subsample",
                        lambda: rf(n_estimators=100, max_features=0.7, min_samples_leaf=3, class_weight="balanced_subsample")),
               grid=[dict(n_estimators=200, max_features=mf, min_samples_leaf=l) for mf, l in itertools.product(["sqrt", 0.5], [5, 20, 60])],
               build=lambda p: rf(**p)),
    "gb": dict(default=("Sullivan's settings: 100 trees, learning rate 0.05, depth 2",
                        lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=2, random_state=42, early_stopping=False)),
               grid=[dict(learning_rate=lr, max_depth=d, min_samples_leaf=l) for lr, d, l in itertools.product([0.05, 0.1], [3, 5], [20, 100])],
               build=None),
    "knn": dict(default=("K = 1, standardised features", lambda: knn(n_neighbors=1)),
                grid=[dict(n_neighbors=k, weights=w) for k, w in itertools.product([1, 5, 10, 25, 50, 100, 200, 400], ["uniform", "distance"])],
                build=lambda p: knn(**p)),
}

log, P_test, info = [], {}, {}
for key, s in SPECS.items():
    t0 = time.time()
    note, mk = s["default"]
    m = mk().fit(tr[FEATS], tr.y)
    d_va = mlogloss(full_proba(m, va[FEATS]), va.y.values)
    P_test[f"{key}_default"] = full_proba(mk().fit(trva[FEATS], trva.y), te[FEATS])
    log.append(dict(model=C.OWNER[key], version="default", params=note, val_logloss=d_va, chosen=False))
    best = None
    for p in s["grid"]:
        if key == "gb":
            g = HistGradientBoostingClassifier(max_iter=800, random_state=42, early_stopping=False, **p).fit(tr[FEATS], tr.y)
            curve = []
            for Pv in g.staged_predict_proba(va[FEATS]):
                Pf = np.zeros((len(va), K)); Pf[:, g.classes_] = Pv; curve.append(mlogloss(Pf, va.y.values))
            p = {**p, "max_iter": int(np.argmin(curve)) + 1}; v = min(curve)
        else:
            g = s["build"](p).fit(tr[FEATS], tr.y); v = mlogloss(full_proba(g, va[FEATS]), va.y.values)
        r = dict(model=C.OWNER[key], version="tuning", params=str(p), val_logloss=v, chosen=False); log.append(r)
        if best is None or v < best[0]["val_logloss"]: best = (r, p)
        print(f"  {key} {p} val {v:.4f}  ({time.time() - t0:.0f}s)")
    r, p = best; r["chosen"] = True
    final = (HistGradientBoostingClassifier(random_state=42, early_stopping=False, **p) if key == "gb" else s["build"](p)).fit(trva[FEATS], trva.y)
    P_test[key] = full_proba(final, te[FEATS])
    info[key] = dict(params=p, val_logloss=r["val_logloss"], default_val=d_va)
    print(f"{C.OWNER[key]:30s} default val {d_va:.4f} | tuned val {r['val_logloss']:.4f} {p} | {time.time() - t0:.0f}s")

# XGBoost: the saved model from xgb_models.py
xm = XGBClassifier(); xm.load_model(os.path.join(C.REPO, "RESULT", "xgboost", "model", "intraday_xgb.json"))
P_test["xgb"] = xm.predict_proba(te[FEATS])
xlog = pd.read_csv(os.path.join(C.REPO, "RESULT", "xgboost", "xgb_tuning_log.csv"))
for _, r in xlog.iterrows():
    log.append(dict(model=C.OWNER["xgb"], version="tuning", params=str({k: r[k] for k in ["max_depth", "learning_rate", "min_child_weight", "best_trees"]}),
                    val_logloss=r.val_logloss, chosen=bool(r.val_logloss == xlog.val_logloss.min())))
info["xgb"] = dict(params="see xgb_tuning_log", val_logloss=float(xlog.val_logloss.min()), default_val=np.nan)

# baseline: average outcome for each hour (training + validation)
tab = pd.crosstab(trva.hour, trva.y).reindex(index=range(24), columns=range(K), fill_value=0) + 0.5
P_test["baseline"] = (tab.T / tab.sum(axis=1)).T.loc[te.hour].to_numpy()

# ---------------------------------------------------------------- scores
y = te.y.values
pt = te[["date", "hour", "time", "temp", "rmax", "final_max"]].copy()
for k, P in P_test.items():
    pt[k] = np.floor(te.rmax) + (P * np.arange(K)).sum(1) + 0.5
order = ["xgb", "rf", "gb", "knn", "rf_default", "gb_default", "knn_default", "baseline"]
name = lambda k: C.OWNER.get(k.replace("_default", ""), "Per-hour average (baseline)") + (" - default" if k.endswith("_default") else "")
rows = []
for k in order:
    m = C.point_metrics(pt[k], pt.final_max)
    rows.append(dict(model=name(k), test_logloss=mlogloss(P_test[k], y), val_logloss=info.get(k, {}).get("val_logloss", np.nan) if not k.endswith("_default") else
                     info[k.replace("_default", "")]["default_val"], **m))
lb = pd.DataFrame(rows)
print(lb.round(3).to_string(index=False))

# ---------------------------------------------------------------- bracket probabilities + backtest at h:10
mk, prices = C.load_markets(), C.load_prices()
bt_s, bt_first, bt_all = [], [], []
for k in ["xgb", "rf", "gb", "knn"]:
    rows = []
    for i, r in te.iterrows():
        br = mk[mk.obs_date == r.date]
        if br.empty: continue
        for b, p in F.bracket_probs(P_test[k][i], r.rmax, br[["bracket", "lo", "hi"]]).items():
            rows.append(dict(date=r.date, hour=int(r.hour), time_hkt=r.time, bracket=b, model_p=float(p)))
    pr = pd.DataFrame(rows); pr.to_csv(os.path.join(OUT, f"hourly_bracket_probs_{k}.csv"), index=False)
    trades, first, s = C.backtest(pr, prices, mk, 10, C.OWNER[k])
    s_late = C.backtest(pr, prices, mk, 70, C.OWNER[k])[2]
    s["first_signal_pnl_1h_late"] = s_late["first_signal_pnl"]
    bt_s.append(s); bt_first.append(first); bt_all.append(trades)
bt_s = pd.DataFrame(bt_s)
print(bt_s.round(3).to_string(index=False))

pt.round(4).to_csv(os.path.join(OUT, "hourly_test_predicted_max.csv"), index=False)
lb.to_csv(os.path.join(OUT, "hourly_leaderboard.csv"), index=False)
pd.DataFrame(log).to_csv(os.path.join(OUT, "hourly_tuning_log.csv"), index=False)
bt_s.to_csv(os.path.join(OUT, "hourly_backtest_summary.csv"), index=False)
pd.concat(bt_first).to_csv(os.path.join(OUT, "hourly_backtest_first_signal_trades.csv"), index=False)
np.savez(os.path.join(OUT, "hourly_test_class_probs.npz"), **P_test, y=y)
print("saved", OUT)
