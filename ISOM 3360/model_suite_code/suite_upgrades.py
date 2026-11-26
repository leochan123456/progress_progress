"""Four day-ahead upgrades, one per member's model, same split as suite_daily.py (train 2010-22, val 2023, test 2024+).

  XGBoost (Ryan)            + HKO's official forecast as an input (latest bulletin issued before midnight for day t).
                            Archive starts Jan 2020; XGBoost's built-in missing-value handling lets it still train on 2010-22.
  Gradient Boosting (Sull.) Quantile regression: 7 percentiles -> full forecast distribution -> bracket odds; calibration.
  KNN (Ada)                 Analog method: compare days by deviation from the date's average (season removed); distance
                            metric Euclidean / Manhattan / cosine / correlation.
  Random Forest (Chris)     Walk-forward: refit at the start of every test month on all data so far; drift by year.
"""
import itertools, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import suite_common as C
BENCH_DIR = os.path.join(C.PROJECT_ROOT, "hourly_benchmark_code")
sys.path.insert(0, BENCH_DIR)
import benchmarks as BM  # HKO bulletin parser

OUT = os.path.join(C.OUT, "upgrades"); os.makedirs(OUT, exist_ok=True)
Y = "target_max_temp_c"
df = pd.read_csv(C.TRAINING_SET, parse_dates=["date"]).dropna(subset=[Y]).reset_index(drop=True)
FEATS = [c for c in df.columns if c not in ("date", Y)]
base = pd.read_csv(os.path.join(C.OUT, "daily", "daily_test_predictions.csv"), parse_dates=["date"])
base_val = pd.read_csv(os.path.join(C.OUT, "daily", "daily_val_predictions.csv"), parse_dates=["date"])
settings = pd.read_csv(os.path.join(C.OUT, "daily", "daily_final_settings.csv")).set_index("model")
yrs = df.date.dt.year
tr, va, te = df[yrs <= 2022].copy(), df[yrs == 2023].copy(), df[yrs >= 2024].copy()
trva = pd.concat([tr, va])
res = {"date": te.date.values, "actual": te[Y].values}
val = {"date": va.date.values, "actual": va[Y].values}
dist = {}   # key -> (test CDF function over bracket bounds)
notes = []

# ===================================================================== 1. XGBoost + HKO forecast
t0 = time.time()
bul = BM.load_hko_forecasts().dropna(subset=["fmax", "target"])
bul = bul[bul.issued < bul.target]                                  # issued before midnight starting day t
hko = bul.sort_values("issued").groupby("target").fmax.last()
print(f"HKO forecasts: {len(hko)} days, {hko.index.min():%Y-%m-%d} .. {hko.index.max():%Y-%m-%d}  ({time.time() - t0:.0f}s parse)")
for d in (df, tr, va, te, trva):
    d["hko_fcst"] = hko.reindex(d.date).values
    d["hko_minus_lag1"] = d.hko_fcst - d.maxT_lag1
FX = FEATS + ["hko_fcst", "hko_minus_lag1"]
cover = pd.DataFrame({"period": ["train 2010-2022", "  of which 2020-2022", "validation 2023", "test 2024+"],
                      "days": [len(tr), int((tr.date.dt.year >= 2020).sum()), len(va), len(te)],
                      "with HKO forecast": [int(tr.hko_fcst.notna().sum()), int(tr[tr.date.dt.year >= 2020].hko_fcst.notna().sum()),
                                            int(va.hko_fcst.notna().sum()), int(te.hko_fcst.notna().sum())]})
print(cover.to_string(index=False))
log = []
for variant, trn in [("2010+ (forecast missing before 2020)", tr), ("2020+ only", tr[tr.date.dt.year >= 2020])]:
    for d_, lr, m in itertools.product([2, 3, 4, 6], [0.03, 0.1], [1, 5, 20]):
        x = XGBRegressor(n_estimators=3000, max_depth=d_, learning_rate=lr, min_child_weight=m, subsample=0.8, colsample_bytree=0.8,
                         random_state=42, n_jobs=2, early_stopping_rounds=100)
        x.fit(trn[FX], trn[Y], eval_set=[(va[FX], va[Y])], verbose=False)
        log.append(dict(variant=variant, max_depth=d_, learning_rate=lr, min_child_weight=m, n_estimators=int(x.best_iteration) + 1,
                        train_rmse=C.rmse(x.predict(trn[FX]), trn[Y]), val_rmse=C.rmse(x.predict(va[FX]), va[Y])))
xlog = pd.DataFrame(log).sort_values("val_rmse"); best = xlog.iloc[0]
xlog["chosen"] = ""; xlog.loc[best.name, "chosen"] = "CHOSEN"
p = dict(max_depth=int(best.max_depth), learning_rate=float(best.learning_rate), min_child_weight=float(best.min_child_weight), n_estimators=int(best.n_estimators))
mk = lambda: XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=2, **p)
trn = tr if best.variant.startswith("2010") else tr[tr.date.dt.year >= 2020]
m_tr = mk().fit(trn[FX], trn[Y]); val["xgb_hko"] = m_tr.predict(va[FX]); sig_x = float(np.std(val["xgb_hko"] - va[Y]))
trvan = trva if best.variant.startswith("2010") else trva[trva.date.dt.year >= 2020]
m_f = mk().fit(trvan[FX], trvan[Y]); res["xgb_hko"] = m_f.predict(te[FX])
gain = pd.Series(m_f.get_booster().get_score(importance_type="gain")).reindex(FX).fillna(0)
gain = (gain / gain.sum()).sort_values(ascending=False)
res["hko_fcst"] = te.hko_fcst.values
notes.append(f"XGBoost + HKO: chosen {best.variant}, {p}, val RMSE {best.val_rmse:.3f}")
print(notes[-1], f"| test {C.rmse(res['xgb_hko'], te[Y]):.3f} | {time.time() - t0:.0f}s")

# ===================================================================== 2. Gradient Boosting quantile regression
t0 = time.time()
QS = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
gp = dict(max_depth=4, learning_rate=0.05, n_estimators=533, min_samples_leaf=3, max_features=0.7, subsample=0.8, random_state=42)
qv, qt = np.zeros((len(va), len(QS))), np.zeros((len(te), len(QS)))
for i, q in enumerate(QS):
    qv[:, i] = make_pipeline(SimpleImputer(strategy="median"), GradientBoostingRegressor(loss="quantile", alpha=q, **gp)).fit(tr[FEATS], tr[Y]).predict(va[FEATS])
    qt[:, i] = make_pipeline(SimpleImputer(strategy="median"), GradientBoostingRegressor(loss="quantile", alpha=q, **gp)).fit(trva[FEATS], trva[Y]).predict(te[FEATS])
    print(f"  quantile {q} done ({time.time() - t0:.0f}s)")
qv.sort(axis=1); qt.sort(axis=1)


def q_cdf(Q, x):
    """CDF at x (array per row) from the 7 quantiles, linear between knots, Normal tails beyond the 5th/95th."""
    s = np.maximum((Q[:, -1] - Q[:, 0]) / (2 * 1.645), 0.3)
    knots_x = np.column_stack([Q[:, 0] - 3 * s, Q, Q[:, -1] + 3 * s]); knots_p = np.r_[0.0013, QS, 0.9987]
    x = np.broadcast_to(np.asarray(x, float), (len(Q),)) if np.ndim(x) == 0 else np.asarray(x, float)
    out = np.array([np.interp(xi, kx, knots_p, left=0, right=1) for xi, kx in zip(x, knots_x)])
    return out


qt_raw, qv_raw = qt.copy(), qv.copy()
# spread calibration: stretch every percentile's distance from the median by s, s chosen on the validation year
s_grid = np.round(np.arange(0.8, 2.01, 0.05), 2)
s_ll = [(s, float(-np.mean(np.log(np.clip(
    (lambda Q: q_cdf(Q, np.floor(va[Y].values) + 1) - q_cdf(Q, np.floor(va[Y].values)))(qv[:, [3]] + s * (qv - qv[:, [3]])), 1e-6, 1))))) for s in s_grid]
S_BEST = min(s_ll, key=lambda x: x[1])[0]
print("spread factor chosen on validation:", S_BEST)
pd.DataFrame(s_ll, columns=["spread_factor", "val_bracket_logloss"]).to_csv(os.path.join(OUT, "gb_spread_calibration.csv"), index=False)
qt = qt[:, [3]] + S_BEST * (qt - qt[:, [3]]); qv = qv[:, [3]] + S_BEST * (qv - qv[:, [3]])
res["gb_quantile"] = qt[:, 3]; val["gb_quantile"] = qv[:, 3]
for i, q in enumerate(QS): res[f"gb_q{int(q * 100):02d}"] = qt[:, i]
sig_g = float(settings.loc[C.OWNER["gb"], "sigma"])


def bracket_ll_quant(Q, actual):
    k = np.floor(actual); p = q_cdf(Q, k + 1) - q_cdf(Q, k)
    return float(-np.mean(np.log(np.clip(p, 1e-6, 1))))


gb_cmp = pd.DataFrame([
    dict(version="Tuned GB + Normal spread (one number, fixed width)", val_bracket_logloss=C.gauss_bracket_logloss(base_val.gb.values, sig_g, va[Y].values),
         test_bracket_logloss=C.gauss_bracket_logloss(base.gb.values, sig_g, te[Y].values),
         cover_80=float(np.mean(np.abs(te[Y].values - base.gb.values) <= 1.2816 * sig_g)), cover_90=float(np.mean(np.abs(te[Y].values - base.gb.values) <= 1.645 * sig_g)),
         avg_80_width=2 * 1.2816 * sig_g),
    dict(version="Quantile GB, raw (7 percentiles, width changes day to day)", val_bracket_logloss=bracket_ll_quant(qv_raw, va[Y].values),
         test_bracket_logloss=bracket_ll_quant(qt_raw, te[Y].values),
         cover_80=float(np.mean((te[Y].values >= qt_raw[:, 1]) & (te[Y].values <= qt_raw[:, 5]))), cover_90=float(np.mean((te[Y].values >= qt_raw[:, 0]) & (te[Y].values <= qt_raw[:, 6]))),
         avg_80_width=float(np.mean(qt_raw[:, 5] - qt_raw[:, 1]))),
    dict(version=f"Quantile GB, spread-calibrated on 2023 (x{S_BEST})", val_bracket_logloss=bracket_ll_quant(qv, va[Y].values),
         test_bracket_logloss=bracket_ll_quant(qt, te[Y].values),
         cover_80=float(np.mean((te[Y].values >= qt[:, 1]) & (te[Y].values <= qt[:, 5]))), cover_90=float(np.mean((te[Y].values >= qt[:, 0]) & (te[Y].values <= qt[:, 6]))),
         avg_80_width=float(np.mean(qt[:, 5] - qt[:, 1])))])
print(gb_cmp.round(3).to_string(index=False))
# reliability: every test day x bracket 10..37 -> predicted P vs outcome
rel = []
for b in range(10, 38):
    hit = (np.floor(te[Y].values) == b).astype(int)
    rel.append(pd.DataFrame({"normal": C.gauss_bracket_probs(base.gb.values, sig_g, b, b + 1), "quantile": q_cdf(qt, b + 1) - q_cdf(qt, b), "hit": hit}))
rel = pd.concat(rel)
bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0001]
rtab = []
for name in ("normal", "quantile"):
    g = rel.groupby(pd.cut(rel[name], bins, right=False), observed=True)
    rtab.append(pd.DataFrame({"model": name, "predicted": g[name].mean(), "actual_freq": g.hit.mean(), "n": g.size()}).reset_index(names="bin"))
rtab = pd.concat(rtab); rtab["bin"] = rtab["bin"].astype(str)
print(f"GB quantile done {time.time() - t0:.0f}s")

# ===================================================================== 3. KNN analog method
t0 = time.time()
SEASON = ["sin_day", "cos_day", "month", "year"]
AF = [c for c in FEATS if c not in SEASON]
doy = lambda d: d.dt.dayofyear.clip(upper=365).values


def clim_table(col):
    c = tr.groupby(doy(tr.date))[col].mean().reindex(range(1, 366)).interpolate(limit_direction="both")
    c = pd.concat([c.iloc[-7:], c, c.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
    return c.values  # index 0 = day 1


CL = {c: clim_table(c) for c in AF + [Y]}
def anomalies(d): return pd.DataFrame({c: d[c].values - CL[c][doy(d.date) - 1] for c in AF}, index=d.index)
A_tr, A_va, A_trva, A_te = anomalies(tr), anomalies(va), anomalies(trva), anomalies(te)
yclim = lambda d: CL[Y][doy(d.date) - 1]
rows = []
for space, (Xa, Xb) in {"raw features (season included)": (tr[FEATS], va[FEATS]), "deviation from date average (analog)": (A_tr, A_va)}.items():
    analog = space.startswith("deviation")
    ytr = tr[Y].values - (yclim(tr) if analog else 0)
    for metric, k, w in itertools.product(["euclidean", "manhattan", "cosine", "correlation"], [5, 10, 15, 20, 30, 50, 75, 100], ["uniform", "distance"]):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor(k, weights=w, metric=metric, algorithm="brute"))
        pv = m.fit(Xa, ytr).predict(Xb) + (yclim(va) if analog else 0)
        rows.append(dict(feature_space=space, metric=metric, k=k, weights=w, val_rmse=C.rmse(pv, va[Y])))
klog = pd.DataFrame(rows)
ktab = klog.loc[klog.groupby(["feature_space", "metric"]).val_rmse.idxmin()].sort_values("val_rmse")
print(ktab.round(3).to_string(index=False))
bk = klog[klog.feature_space.str.startswith("deviation")].sort_values("val_rmse").iloc[0]
klog["chosen"] = ""; klog.loc[bk.name, "chosen"] = "CHOSEN"
mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsRegressor(int(bk.k), weights=bk.weights, metric=bk.metric, algorithm="brute"))
val["knn_analog"] = mk().fit(A_tr, tr[Y].values - yclim(tr)).predict(A_va) + yclim(va)
res["knn_analog"] = mk().fit(A_trva, trva[Y].values - yclim(trva)).predict(A_te) + yclim(te)
sig_k = float(np.std(val["knn_analog"] - va[Y]))
notes.append(f"KNN analog: {bk.metric}, K={int(bk.k)}, {bk.weights}, val RMSE {bk.val_rmse:.3f}")
print(notes[-1], f"| test {C.rmse(res['knn_analog'], te[Y]):.3f} | {time.time() - t0:.0f}s")

# ===================================================================== 4. RF walk-forward
t0 = time.time()
rp = dict(n_estimators=300, max_features=0.5, min_samples_leaf=1, random_state=42, n_jobs=-1)
wf = np.full(len(te), np.nan)
months = pd.period_range(te.date.min(), te.date.max(), freq="M")
for mth in months:
    start = mth.to_timestamp()
    hist = df[df.date < start]; idx = (te.date.dt.to_period("M") == mth).values
    m = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(**rp)).fit(hist[FEATS], hist[Y])
    wf[idx] = m.predict(te.loc[idx, FEATS])
res["rf_walkforward"] = wf
val["rf_walkforward"] = base_val.rf.values   # same model as tuned RF on the validation year (refits only matter in test)
sig_r = float(settings.loc[C.OWNER["rf"], "sigma"])
drift = []
for y_, g in pd.DataFrame(res).assign(rf=base.rf.values).groupby(pd.to_datetime(res["date"]).year):
    drift.append(dict(year=y_, days=len(g), actual_mean=g.actual.mean(), train_years_mean_for_these_dates=float(np.mean(CL[Y][doy(pd.Series(g.date)) - 1])),
                      rf_static_bias=float((g.rf - g.actual).mean()), rf_walkforward_bias=float((g.rf_walkforward - g.actual).mean()),
                      rf_static_rmse=C.rmse(g.rf, g.actual), rf_walkforward_rmse=C.rmse(g.rf_walkforward, g.actual)))
drift = pd.DataFrame(drift)
print(drift.round(3).to_string(index=False)); print(f"RF walk-forward {len(months)} refits, {time.time() - t0:.0f}s")

# ===================================================================== combine, score, backtest
R = pd.DataFrame(res)
for k in ["rf", "gb", "knn", "xgb"]:
    R[f"{k}_tuned"] = base[k].values; R[f"{k}_default"] = base[f"{k}_default"].values
R["persistence"], R["market"] = base.persistence.values, base.market.values
SIG = {"xgb_hko": sig_x, "knn_analog": sig_k, "rf_walkforward": sig_r}
ups = {"xgb": "xgb_hko", "gb": "gb_quantile", "knn": "knn_analog", "rf": "rf_walkforward"}
rows = []
for k, u in ups.items():
    for ver, col in [("original (default)", f"{k}_default"), ("tuned", f"{k}_tuned"), ("upgraded", u)]:
        m = C.point_metrics(R[col], R.actual)
        if col == "gb_quantile": ll = bracket_ll_quant(qt, te[Y].values)
        else:
            s = SIG.get(col, float(settings.loc[C.OWNER[k], "sigma"])); ll = C.gauss_bracket_logloss(R[col].values, s, R.actual.values)
        rows.append(dict(model=C.OWNER[k], version=ver, **m, bracket_logloss=ll))
up_lb = pd.DataFrame(rows)
print(up_lb.round(3).to_string(index=False))

mkts, prices = C.load_markets(), C.load_prices()
bt = []
Ri = R.set_index("date")
for k, u in ups.items():
    rows = []
    for d, br in mkts[mkts.obs_date.isin(Ri.index)].groupby("obs_date"):
        if u == "gb_quantile":
            i = int(np.where(R.date == d)[0][0]); Q = qt[[i]]
            pr = np.array([q_cdf(Q, h)[0] - q_cdf(Q, l)[0] for l, h in zip(br.lo.clip(-60, 60), br.hi.clip(-60, 60))])
        else:
            pr = C.gauss_bracket_probs(Ri.loc[d, u], SIG[u], br.lo.values, br.hi.values)
        pr = pr / pr.sum()
        rows += [dict(date=d, hour=0, time_hkt=d, bracket=b, model_p=float(x)) for b, x in zip(br.bracket, pr)]
    _, first, s = C.backtest(pd.DataFrame(rows), prices, mkts, 10, C.OWNER[k] + " - upgraded")
    bt.append(s)
bt = pd.DataFrame(bt); print(bt.round(3).to_string(index=False))

R.to_csv(os.path.join(OUT, "upgrade_test_predictions.csv"), index=False)
up_lb.to_csv(os.path.join(OUT, "upgrade_leaderboard.csv"), index=False)
xlog.to_csv(os.path.join(OUT, "xgb_hko_tuning_log.csv"), index=False)
cover.to_csv(os.path.join(OUT, "xgb_hko_coverage.csv"), index=False)
gain.rename("gain_share").to_csv(os.path.join(OUT, "xgb_hko_importance.csv"))
gb_cmp.to_csv(os.path.join(OUT, "gb_quantile_vs_normal.csv"), index=False)
rtab.to_csv(os.path.join(OUT, "gb_reliability.csv"), index=False)
klog.to_csv(os.path.join(OUT, "knn_analog_tuning_log.csv"), index=False)
ktab.to_csv(os.path.join(OUT, "knn_similarity_comparison.csv"), index=False)
drift.to_csv(os.path.join(OUT, "rf_walkforward_drift.csv"), index=False)
bt.to_csv(os.path.join(OUT, "upgrade_backtest_summary.csv"), index=False)
open(os.path.join(OUT, "notes.txt"), "w").write("\n".join(notes))
print("saved", OUT)
