"""Backtest within-day Yes/No trades against Random Forest probabilities."""
import argparse
import os
import re

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PM_CLEANED = os.path.join(SCRIPT_DIR, "Storage", "PM", "cleaned")
MODEL_RESULTS = os.path.join(SCRIPT_DIR, "RESULT")
BACKTEST_OUTPUT = os.path.join(SCRIPT_DIR, "RESULT", "backtest")

DEFAULT_FEE_COEFFICIENT = 0.05
DEFAULT_MIN_EDGE = 0.03
EXCLUDED_DATES = {"2026-03-13", "2026-03-14", "2026-05-18", "2026-05-19"}


def bracket_probability(row, label):
    match = re.search(r"(-?\d+)", str(label))
    if match is None:
        return np.nan
    cutoff = int(match.group(1))
    if "or below" in label:
        next_column = f"p_ge_{cutoff + 1}c"
        return 1.0 - row[next_column] if next_column in row else 1.0
    if "or higher" in label or "or above" in label:
        column = f"p_ge_{cutoff}c"
        return row[column] if column in row else 0.0
    column = f"p_bracket_{cutoff}c"
    return row[column] if column in row else np.nan


def settlement_winner(actual_max, label):
    if pd.isna(actual_max):
        return np.nan
    actual_bracket = int(np.floor(actual_max))
    match = re.search(r"(-?\d+)", str(label))
    if match is None:
        return np.nan
    cutoff = int(match.group(1))
    if "or below" in label:
        return int(actual_bracket <= cutoff)
    if "or higher" in label or "or above" in label:
        return int(actual_bracket >= cutoff)
    return int(actual_bracket == cutoff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", default=os.path.join(PM_CLEANED, "pm_price_history_hourly_cleaned.csv"))
    parser.add_argument("--predictions", default=os.path.join(MODEL_RESULTS, "rf_predictions.csv"))
    parser.add_argument("--settlements", default=os.path.join(PM_CLEANED, "hko_daily_extract_cleaned.csv"))
    parser.add_argument("--output-dir", default=BACKTEST_OUTPUT)
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE)
    parser.add_argument("--fee-coefficient", type=float, default=DEFAULT_FEE_COEFFICIENT)
    parser.add_argument("--include-excluded-dates", action="store_true")
    args = parser.parse_args()

    prices = pd.read_csv(args.prices)
    predictions = pd.read_csv(args.predictions)
    settlements = pd.read_csv(args.settlements)

    prices["obs_date"] = pd.to_datetime(prices["obs_date"], errors="raise").dt.strftime("%Y-%m-%d")
    prices["time_hkt"] = pd.to_datetime(prices["time_hkt"], errors="raise")
    prices["hour"] = prices["time_hkt"].dt.strftime("%H:%M")
    prices = prices[prices["time_hkt"].dt.strftime("%Y-%m-%d") == prices["obs_date"]].copy()
    predictions["date"] = pd.to_datetime(predictions["date"], errors="raise").dt.strftime("%Y-%m-%d")
    settlements["date"] = pd.to_datetime(settlements["date"], errors="raise").dt.strftime("%Y-%m-%d")

    prices = prices.merge(predictions, left_on="obs_date", right_on="date", how="inner")
    prices = prices.merge(
        settlements[["date", "abs_daily_max_c"]],
        left_on="obs_date",
        right_on="date",
        how="left",
        suffixes=("", "_settlement"),
    )
    prices = prices.dropna(subset=["abs_daily_max_c"])
    if not args.include_excluded_dates:
        prices = prices[~prices["obs_date"].isin(EXCLUDED_DATES)]

    prices["model_p"] = prices.apply(lambda row: bracket_probability(row, row["bracket"]), axis=1)
    prices["settlement_winner"] = prices.apply(
        lambda row: settlement_winner(row["abs_daily_max_c"], row["bracket"]), axis=1
    )
    prices["fee"] = args.fee_coefficient * prices["yes_price"] * (1 - prices["yes_price"])
    yes_edge = prices["model_p"] - prices["yes_price"] - prices["fee"]
    no_edge = (1 - prices["model_p"]) - (1 - prices["yes_price"]) - prices["fee"]
    yes_signal = yes_edge > args.min_edge
    no_signal = no_edge > args.min_edge

    yes_trades = prices.loc[yes_signal].copy()
    yes_trades["side"] = "buy_yes"
    yes_trades["edge"] = yes_edge.loc[yes_signal]
    yes_trades["pnl"] = np.where(
        yes_trades["settlement_winner"] == 1,
        1 - yes_trades["yes_price"] - yes_trades["fee"],
        -yes_trades["yes_price"] - yes_trades["fee"],
    )

    no_trades = prices.loc[no_signal].copy()
    no_trades["side"] = "buy_no"
    no_trades["edge"] = no_edge.loc[no_signal]
    no_trades["pnl"] = np.where(
        no_trades["settlement_winner"] == 0,
        no_trades["yes_price"] - no_trades["fee"],
        -(1 - no_trades["yes_price"]) - no_trades["fee"],
    )

    trades = pd.concat([yes_trades, no_trades], ignore_index=True)
    trades = trades.drop(columns=["date", "date_settlement"], errors="ignore")
    trades = trades.rename(columns={"obs_date": "date", "yes_price": "price"})
    output_columns = [
        "date", "hour", "bracket", "side", "price", "model_p", "edge", "pnl",
        "fee", "settlement_winner", "abs_daily_max_c", "time_hkt",
    ]
    trades = trades[output_columns].sort_values(["date", "hour", "bracket", "side"])
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "polymarket_within_day_trades.csv")
    trades.to_csv(output_path, index=False)

    summary = pd.DataFrame([{
        "min_edge": args.min_edge,
        "fee_coefficient": args.fee_coefficient,
        "quote_rows_same_day": len(prices),
        "trade_rows": len(trades),
        "yes_trades": int((trades["side"] == "buy_yes").sum()),
        "no_trades": int((trades["side"] == "buy_no").sum()),
        "total_pnl_per_share": trades["pnl"].sum(),
        "mean_pnl_per_trade": trades["pnl"].mean() if len(trades) else np.nan,
        "winning_trades": int((trades["pnl"] > 0).sum()),
    }])
    summary_path = os.path.join(args.output_dir, "polymarket_within_day_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print("saved:", output_path)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
