"""Backtest the daily and, when available, hourly Random Forest probabilities."""
import os

import numpy as np
import pandas as pd

import rf_common as C


def build_daily_probabilities(predictions, markets, column, sigma):
    rows = []
    forecasts = predictions.set_index("date")[column]
    for date, brackets in markets[markets.obs_date.isin(forecasts.index)].groupby("obs_date"):
        probabilities = C.SUITE.gauss_bracket_probs(
            float(forecasts.loc[date]), sigma, brackets.lo.to_numpy(), brackets.hi.to_numpy()
        )
        total = probabilities.sum()
        if total <= 0:
            raise ValueError(f"RF bracket probabilities sum to zero on {date:%Y-%m-%d}.")
        probabilities /= total
        rows.extend(
            {
                "date": date,
                "hour": 0,
                "time_hkt": date,
                "bracket": bracket,
                "model_p": float(probability),
            }
            for bracket, probability in zip(brackets.bracket, probabilities)
        )
    return pd.DataFrame(rows)


def main():
    predictions_path = os.path.join(C.DAILY_OUT, "daily_test_predictions.csv")
    walkforward_path = os.path.join(C.OUT, "rf_walkforward_test_predictions.csv")
    settings_path = os.path.join(C.DAILY_OUT, "daily_final_settings.csv")
    markets_path = C.SUITE.PM_MARKETS
    C.ensure_files(predictions_path, walkforward_path, settings_path, markets_path)

    predictions = pd.read_csv(predictions_path, parse_dates=["date"])
    walkforward = pd.read_csv(walkforward_path, parse_dates=["date"])
    settings = pd.read_csv(settings_path).set_index("version")
    markets = C.SUITE.load_markets()
    prices, lag_minutes, price_source = C.load_prices()
    summaries, first_signals, all_trades = [], [], []

    for frame, column, version, label in (
        (predictions, "rf_default", "default", "Chris RF daily (default)"),
        (predictions, "rf_tuned", "tuned", "Chris RF daily (tuned)"),
        (walkforward, "rf_walkforward", "tuned", "Chris RF monthly walk-forward"),
    ):
        probability_rows = build_daily_probabilities(
            frame, markets, column, float(settings.loc[version, "sigma"])
        )
        trades, first, summary = C.SUITE.backtest(
            probability_rows, prices, markets, lag_minutes, label,
        )
        summary.update({"price_source": price_source, "lag_minutes": lag_minutes})
        summaries.append(summary)
        first_signals.append(first)
        all_trades.append(trades)

    hourly_path = os.path.join(C.HOURLY_OUT, "hourly_bracket_probs_rf.csv")
    if os.path.isfile(hourly_path):
        hourly = pd.read_csv(hourly_path, parse_dates=["date", "time_hkt"])
        trades, first, summary = C.SUITE.backtest(
            hourly, prices, markets, lag_minutes, "Chris RF hourly (tuned)"
        )
        summary.update({"price_source": price_source, "lag_minutes": lag_minutes})
        summaries.append(summary)
        first_signals.append(first)
        all_trades.append(trades)

    os.makedirs(C.BACKTEST_OUT, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(os.path.join(C.BACKTEST_OUT, "rf_backtest_summary.csv"), index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(
        os.path.join(C.BACKTEST_OUT, "rf_backtest_trades.csv"), index=False
    )
    pd.concat(first_signals, ignore_index=True).to_csv(
        os.path.join(C.BACKTEST_OUT, "rf_backtest_first_signal_trades.csv"), index=False
    )

    print(f"Price source: {price_source}; cutoff lag: {lag_minutes} minutes")
    print(summary_frame.round(4).to_string(index=False))
    print("saved:", C.BACKTEST_OUT)


if __name__ == "__main__":
    main()
