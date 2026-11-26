"""Score the hourly forecasts: SSE, RMSE, MAE, bias and bracket hit-rate, by hour of day."""
import os
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
NAMES = {"xgb": "XGBoost hourly (Ryan)", "market_ev": "Polymarket expected value", "market_mode": "Polymarket favourite bracket",
         "hko_fcst_obs": "HKO forecast + observed max", "hko_fcst": "HKO forecast only", "running_max": "Observed running max"}


def score(B, cols):
    rows = []
    for h, g in list(B.groupby("hour")) + [("all", B)]:
        for c in cols:
            e = g[c] - g.actual
            rows.append(dict(hour=h, forecaster=NAMES[c], n=len(e), SSE=float((e ** 2).sum()), RMSE=float(np.sqrt((e ** 2).mean())),
                             MAE=float(e.abs().mean()), bias=float(e.mean()),
                             bracket_hit=float((np.floor(g[c]) == np.floor(g.actual)).mean())))
    return pd.DataFrame(rows)


def main():
    B = pd.read_csv(os.path.join(OUT, "hourly_benchmark_forecasts.csv"), parse_dates=["date"])
    bench = ["market_ev", "market_mode", "hko_fcst_obs", "hko_fcst", "running_max"]
    full = B.dropna(subset=bench)
    test = B[B.date >= "2026-09-01"].dropna(subset=bench + ["xgb"])
    s_full, s_test = score(full, bench), score(test, ["xgb"] + bench)
    s_full.to_csv(os.path.join(OUT, "errors_by_hour_full_period.csv"), index=False)
    s_test.to_csv(os.path.join(OUT, "errors_by_hour_test.csv"), index=False)
    for name, s, d in [("TEST (with XGBoost)", s_test, test), ("FULL market period", s_full, full)]:
        print(f"\n== {name}: {d.date.nunique()} days {d.date.min():%Y-%m-%d} -> {d.date.max():%Y-%m-%d}, {len(d)} date-hours")
        print(s[s.hour == "all"].drop(columns="hour").round(3).to_string(index=False))
        piv = s[s.hour != "all"].pivot(index="hour", columns="forecaster", values="RMSE")
        print(piv.loc[[0, 3, 6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 23]].round(2).to_string())


if __name__ == "__main__":
    main()
