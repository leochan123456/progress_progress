import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import rf_backtest
import rf_common


class BacktestTests(unittest.TestCase):
    def test_walkforward_probabilities_use_walkforward_forecast(self):
        date = pd.Timestamp("2026-03-20")
        forecasts = pd.DataFrame({
            "date": [date], "rf_tuned": [20.0], "rf_walkforward": [24.0],
        })
        markets = pd.DataFrame({
            "obs_date": [date, date], "bracket": ["low", "high"],
            "lo": [float("-inf"), 22.0], "hi": [22.0, float("inf")],
        })
        probabilities = rf_backtest.build_daily_probabilities(
            forecasts, markets, "rf_walkforward", 1.0
        )
        self.assertGreater(probabilities.loc[1, "model_p"], 0.97)
        self.assertAlmostEqual(probabilities.model_p.sum(), 1.0)
        self.assertEqual(probabilities.loc[0, "time_hkt"], date)

    def test_canonical_ten_minute_file_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "prices.csv")
            pd.DataFrame({
                "obs_date": ["2026-03-20"], "time_hkt": ["2026-03-20 00:10:00"],
                "bracket": ["high"], "yes_price": [0.4],
            }).to_csv(path, index=False)
            with patch.object(rf_common, "PM_PRICE_10MIN", path):
                prices, lag, source = rf_common.load_prices()
        self.assertEqual((lag, source), (10, "10-minute"))
        self.assertEqual(prices.loc[0, "date"], pd.Timestamp("2026-03-20"))

    def test_daily_backtest_uses_latest_price_at_or_before_0010(self):
        date = pd.Timestamp("2026-03-20")
        forecasts = pd.DataFrame({"date": [date], "rf_walkforward": [24.0]})
        markets = pd.DataFrame({
            "obs_date": [date, date], "bracket": ["low", "high"],
            "lo": [float("-inf"), 22.0], "hi": [22.0, float("inf")],
            "resolved_yes": [0, 1],
        })
        prices = pd.DataFrame({
            "date": [date, date, date], "bracket": ["high"] * 3,
            "time_hkt": [date, date + pd.Timedelta(minutes=10), date + pd.Timedelta(minutes=20)],
            "yes_price": [0.2, 0.4, 0.8],
        })
        probabilities = rf_backtest.build_daily_probabilities(
            forecasts, markets, "rf_walkforward", 1.0
        )
        trades, _, summary = rf_common.SUITE.backtest(
            probabilities, prices, markets, 10, "walk-forward"
        )
        self.assertEqual(summary["days"], 1)
        self.assertEqual(trades.iloc[0]["price"], 0.4)


if __name__ == "__main__":
    unittest.main()
