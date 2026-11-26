"""Shared pieces for the model suite: paths, metrics, bracket probabilities, Polymarket backtest."""
import os
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
REPO = os.path.join(PROJECT_ROOT, "ISOM3360_XGBoost_for_repo", "ISOM 3360")
sys.path.insert(0, REPO)
import xgb_features as F  # noqa: E402

OUT = os.path.join(HERE, "out")
TRAINING_SET = os.path.join(PROJECT_ROOT, "Storage", "HKO", "processed", "hko_training_set.csv")
PM_MARKETS = os.path.join(PROJECT_ROOT, "Storage", "PM", "cleaned", "pm_markets_cleaned.csv")
PM_PRICES_10MIN = os.path.join(PROJECT_ROOT, "hourly_benchmark_code", "pm_price_10min.csv")
EXCLUDED = {"2026-03-13", "2026-03-14", "2026-05-18", "2026-05-19"}
FEE, MIN_EDGE = 0.05, 0.05
OWNER = {"rf": "Random Forest (Chris)", "gb": "Gradient Boosting (Sullivan)", "knn": "KNN (Ada)", "xgb": "XGBoost (Ryan)",
         "ridge": "Ridge regression (baseline)", "persistence": "Persistence: yesterday's max", "climatology": "Climatology: average for the date",
         "hko": "HKO official forecast", "market": "Polymarket"}


def rmse(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    return float(np.sqrt(np.mean((p - a) ** 2)))


def point_metrics(p, a):
    p, a = np.asarray(p, float), np.asarray(a, float)
    e = p - a
    return dict(n=len(e), SSE=float((e ** 2).sum()), RMSE=float(np.sqrt((e ** 2).mean())), MAE=float(np.abs(e).mean()),
                bias=float(e.mean()), bracket_hit=float((np.floor(p) == np.floor(a)).mean()))


def gauss_bracket_probs(mu, sigma, lo, hi):
    """P(floor(max) in [lo, hi)) for a Normal(mu, sigma) forecast. lo/hi may be +-inf."""
    return np.clip(norm.cdf((hi - mu) / sigma) - norm.cdf((lo - mu) / sigma), 0, 1)


def gauss_bracket_logloss(mu, sigma, actual):
    k = np.floor(actual)
    p = gauss_bracket_probs(mu, sigma, k, k + 1)
    return float(-np.mean(np.log(np.clip(p, 1e-6, 1))))


def load_markets():
    mk = pd.read_csv(PM_MARKETS, parse_dates=["obs_date"])
    mk = mk[~mk.event_slug.str.startswith("arch-") & ~mk.obs_date.dt.strftime("%Y-%m-%d").isin(EXCLUDED)].copy()
    b = mk.bracket.apply(F.bracket_bounds)
    mk["lo"], mk["hi"] = [x[0] for x in b], [x[1] for x in b]
    return mk


def load_prices():
    return pd.read_csv(PM_PRICES_10MIN, parse_dates=["obs_date", "time_hkt"]).rename(columns={"obs_date": "date"})


def backtest(pred, prices, markets, lag_minutes, label, seed=0):
    """pred: date, hour, time_hkt (decision reference time), bracket, model_p.
    Price = last 10-minute price at or before time_hkt + lag. One row per qualifying quote; first-signal per
    date x bracket x side. Returns (trades, first, summary dict)."""
    win = markets[markets.resolved_yes == 1][["obs_date", "bracket"]].rename(columns={"obs_date": "date", "bracket": "winner"})
    p = pred.copy()
    p["cut"] = p.time_hkt + pd.Timedelta(minutes=lag_minutes)
    p = p.sort_values("cut")
    ph = prices.sort_values("time_hkt")
    q = pd.merge_asof(p, ph[["date", "bracket", "time_hkt", "yes_price"]].rename(columns={"time_hkt": "price_time"}),
                      left_on="cut", right_on="price_time", by=["date", "bracket"], direction="backward")
    q = q.merge(win, on="date", how="inner").dropna(subset=["yes_price"])
    q["won"] = (q.bracket == q.winner).astype(int)
    t = q[(q.yes_price >= 0.01) & (q.yes_price <= 0.99)].copy()
    t["fee"] = FEE * t.yes_price * (1 - t.yes_price)
    t["edge_yes"] = t.model_p - t.yes_price - t.fee
    t["edge_no"] = (1 - t.model_p) - (1 - t.yes_price) - t.fee
    yes = t[t.edge_yes > MIN_EDGE].assign(side="YES", price=lambda x: x.yes_price, edge=lambda x: x.edge_yes,
                                          pnl=lambda x: x.won - x.yes_price - x.fee)
    no = t[(t.edge_no > MIN_EDGE) & ~(t.edge_yes > MIN_EDGE)].assign(
        side="NO", price=lambda x: 1 - x.yes_price, edge=lambda x: x.edge_no, pnl=lambda x: (1 - x.won) - (1 - x.yes_price) - x.fee)
    cols = ["date", "hour", "bracket", "side", "price", "model_p", "fee", "edge", "won", "pnl"]
    trades = pd.concat([yes, no])[cols].sort_values(["date", "hour", "bracket"])
    first = trades.sort_values("hour", kind="stable").drop_duplicates(["date", "bracket", "side"]).sort_values(["date", "hour", "bracket"])
    rng = np.random.default_rng(seed)
    per_day = first.groupby("date").pnl.sum().reindex(sorted(q.date.unique()), fill_value=0.0)
    boot = [per_day.iloc[rng.integers(0, len(per_day), len(per_day))].sum() for _ in range(5000)] if len(per_day) else [0]
    s = dict(model=label, days=int(q.date.nunique()), all_quote_trades=len(trades), all_quote_pnl=float(trades.pnl.sum()),
             first_signal_trades=len(first), yes=int((first.side == "YES").sum()), no=int((first.side == "NO").sum()),
             first_signal_pnl=float(first.pnl.sum()), pnl_per_trade=float(first.pnl.mean()) if len(first) else np.nan,
             win_rate=float((first.pnl > 0).mean()) if len(first) else np.nan,
             pnl_95_low=float(np.percentile(boot, 2.5)), pnl_95_high=float(np.percentile(boot, 97.5)),
             expected_edge=float(first.edge.mean()) if len(first) else np.nan)
    return trades.assign(model=label), first.assign(model=label), s
