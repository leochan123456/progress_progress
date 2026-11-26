"""Backtest within-day Yes/No trades against the hourly XGBoost bracket probabilities (the buy points).

For every test date x hour x bracket, compare the model probability (xgb_predictions.csv) with Polymarket's Yes price
just after HKO publishes the reading (~:05), so the market has seen the same information.

    fee           = fee_coefficient * price * (1 - price)
    Buy Yes edge  = model_p - price - fee
    Buy No edge   = (1 - model_p) - (1 - price) - fee
A trade is recorded when the edge exceeds --min-edge.

Inputs : RESULT/xgboost/xgb_predictions.csv (xgb_models.py)
         Storage/PM/cleaned/pm_markets_cleaned.csv, pm_price_history_hourly_cleaned.csv (clean_pm_data.py)
Outputs: RESULT/xgboost/backtest/xgb_within_day_trades.csv      every qualifying hourly quote (same as polymarket_backtest.py)
         RESULT/xgboost/backtest/xgb_first_signal_trades.csv    each date x bracket x side once, at its first signal hour
         RESULT/xgboost/backtest/xgb_within_day_summary.csv
         RESULT/xgboost/backtest/xgb_model_vs_market_by_hour.csv
         RESULT/xgboost/backtest/xgb_model_vs_market.png
"""
import argparse
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
PM_CLEANED = os.path.join(PROJECT_ROOT, "Storage", "PM", "cleaned")
PRED = os.path.join(SCRIPT_DIR, "RESULT", "xgboost", "xgb_predictions.csv")
OUT = os.path.join(SCRIPT_DIR, "RESULT", "xgboost", "backtest")
DEFAULT_FEE_COEFFICIENT = 0.05
DEFAULT_MIN_EDGE = 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    ap.add_argument("--fee-coefficient", type=float, default=DEFAULT_FEE_COEFFICIENT)
    ap.add_argument("--prices", default=os.path.join(PM_CLEANED, "pm_price_history_hourly_cleaned.csv"),
                    help="price history; a 10-minute file (fetch_pm_10min.py) allows --lag-minutes 10")
    ap.add_argument("--lag-minutes", type=float, default=70,
                    help="use the last price at or before reading time + this many minutes. HKO publishes the h:00 "
                         "reading by ~h:05; hourly snapshots are stamped h:00:0x, so 70 = the next hourly snapshot")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pred = pd.read_csv(PRED, parse_dates=["date", "time_hkt"])
    mk = pd.read_csv(os.path.join(PM_CLEANED, "pm_markets_cleaned.csv"), parse_dates=["obs_date"])
    win = mk[mk.resolved_yes == 1][["obs_date", "bracket"]].rename(columns={"obs_date": "date", "bracket": "winner"})
    ph = pd.read_csv(a.prices, parse_dates=["obs_date", "time_hkt"])
    ph = ph.rename(columns={"obs_date": "date", "time_hkt": "price_time"}).sort_values("price_time")

    # market price = last price at or before reading time + lag (after HKO publishes the reading, ~:05)
    pred["price_cutoff"] = pred.time_hkt + pd.Timedelta(minutes=a.lag_minutes)
    pred = pred.sort_values("price_cutoff")
    q = pd.merge_asof(pred, ph[["date", "bracket", "price_time", "yes_price"]], left_on="price_cutoff", right_on="price_time",
                      by=["date", "bracket"], direction="backward")
    q = q.merge(win, on="date", how="inner")
    q["won"] = (q.bracket == q.winner).astype(int)
    q = q.dropna(subset=["yes_price"])

    # model vs market: normalise each hour's prices to sum to 1, look at the winning bracket
    q["mkt_p"] = q.yes_price / q.groupby(["date", "hour"]).yes_price.transform("sum")
    w = q[q.won == 1]
    byh = w.groupby("hour").agg(days=("date", "nunique"), model_p_win=("model_p", "mean"), market_p_win=("mkt_p", "mean"))
    byh.to_csv(os.path.join(OUT, "xgb_model_vs_market_by_hour.csv"))

    # trades
    t = q[(q.yes_price >= 0.01) & (q.yes_price <= 0.99)].copy()
    t["fee"] = a.fee_coefficient * t.yes_price * (1 - t.yes_price)
    t["edge_yes"] = t.model_p - t.yes_price - t.fee
    t["edge_no"] = (1 - t.model_p) - (1 - t.yes_price) - t.fee
    yes = t[t.edge_yes > a.min_edge].assign(side="YES", price=lambda x: x.yes_price, edge=lambda x: x.edge_yes,
                                            pnl=lambda x: x.won - x.yes_price - x.fee)
    no = t[(t.edge_no > a.min_edge) & ~(t.edge_yes > a.min_edge)].assign(
        side="NO", price=lambda x: 1 - x.yes_price, edge=lambda x: x.edge_no, pnl=lambda x: (1 - x.won) - (1 - x.yes_price) - x.fee)
    cols = ["date", "hour", "bracket", "side", "price", "model_p", "fee", "edge", "won", "actual_max", "pnl"]
    trades = pd.concat([yes, no])[cols].sort_values(["date", "hour", "bracket"])
    trades.to_csv(os.path.join(OUT, "xgb_within_day_trades.csv"), index=False)
    first = trades.sort_values("hour").drop_duplicates(["date", "bracket", "side"]).sort_values(["date", "hour", "bracket"])
    first.to_csv(os.path.join(OUT, "xgb_first_signal_trades.csv"), index=False)

    rng = np.random.default_rng(0); per_day = first.groupby("date").pnl.sum()
    boot = [per_day.reindex(rng.choice(per_day.index, len(per_day))).sum() for _ in range(5000)]
    summary = pd.DataFrame([dict(
        lag_minutes=a.lag_minutes, min_edge=a.min_edge, fee_coefficient=a.fee_coefficient, quote_rows_same_day=len(t), trade_rows=len(trades),
        yes_trades=int((trades.side == "YES").sum()), no_trades=int((trades.side == "NO").sum()),
        total_pnl_per_share=trades.pnl.sum(), mean_pnl_per_trade=trades.pnl.mean(), winning_trades=int((trades.pnl > 0).sum()),
        first_signal_trades=len(first), first_signal_total_pnl=first.pnl.sum(), first_signal_mean_pnl=first.pnl.mean(),
        first_signal_pnl_95pct_low=np.percentile(boot, 2.5), first_signal_pnl_95pct_high=np.percentile(boot, 97.5),
        mean_expected_edge=first.edge.mean(), test_days=q.date.nunique())])
    summary.to_csv(os.path.join(OUT, "xgb_within_day_summary.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    ax.plot(byh.index, byh.market_p_win, color="#eb6834", lw=2, marker="o", ms=4, label="Polymarket (next hourly price)")
    ax.plot(byh.index, byh.model_p_win, color="#2a78d6", lw=2, marker="o", ms=4, label="XGBoost model")
    ax.set_ylim(0, 1.02); ax.set_xticks(range(0, 24, 2)); ax.grid(axis="y", color="#e6e5e0")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Hour of the day (HKT)"); ax.set_ylabel("Avg. probability on the bracket that won")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"XGBoost vs Polymarket by hour ({q.date.nunique()} test days)", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "xgb_model_vs_market.png"))

    s = summary.iloc[0]
    print(f"test days {int(s.test_days)} | all qualifying quotes: {int(s.trade_rows)} trades, total {s.total_pnl_per_share:+.2f}")
    print(f"first signal only: {int(s.first_signal_trades)} trades, total {s.first_signal_total_pnl:+.2f} "
          f"(95% range {s.first_signal_pnl_95pct_low:+.2f} to {s.first_signal_pnl_95pct_high:+.2f}); "
          f"expected edge {s.mean_expected_edge * 100:.1f}c vs realised {s.first_signal_mean_pnl * 100:.1f}c")
    print(f"avg probability on winning bracket: model {w.model_p.mean():.1%} vs market {w.mkt_p.mean():.1%}")


if __name__ == "__main__":
    main()
