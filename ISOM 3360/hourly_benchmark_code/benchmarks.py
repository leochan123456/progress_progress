"""Hourly benchmarks for the HKO daily max temperature, scored with SSE / RMSE / MAE by hour of day.

For each date D and hour h (HKT), every forecaster gives a point forecast of D's final max, using only what is
known shortly after HKO's h:00 reading is published:
  market_ev      Polymarket: prices normalised to sum to 1, expected value of the bracket (bracket b -> b + 0.5;
                 'X or below' -> X - 0.5, 'X or higher' -> X + 1.0). Price = the h:10 snapshot (10-minute price
                 history), just after HKO publishes the h:00 reading.
  market_mode    Polymarket favourite bracket -> b + 0.5
  hko_fcst       HKO Local Weather Forecast: latest max temperature stated for day D in a bulletin issued by
                 h:10 (includes the previous evening's 'tomorrow' forecast). HKO stops stating today's max
                 around 11:00, so after that the last stated value carries forward.
  hko_fcst_obs   max(hko_fcst, running max observed so far)  -- HKO forecast updated hourly with observations
  running_max    observed max since midnight at h:00 (lower bound)
  xgb            XGBoost hourly model: floor(running max) + E[d] + 0.5   (test days only)
Truth = HKO settlement max (Daily Extract, as used by Polymarket).
"""
import glob, os, re, sys
import numpy as np
import pandas as pd

BENCH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BENCH)
REPO = os.path.join(PROJECT_ROOT, "ISOM3360_XGBoost_for_repo", "ISOM 3360")
PM = os.path.join(PROJECT_ROOT, "Storage", "PM", "cleaned")
OUT = os.path.join(BENCH, "out")
HOURLY_FRAME = os.path.join(REPO, "Storage", "HKO", "hourly", "hourly_frame.csv")
EXCLUDED = {"2026-03-13", "2026-03-14", "2026-05-18", "2026-05-19"}
MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


# ---------------- HKO forecast bulletins ----------------
def parse_bulletin(xml):
    t = re.search(r"Bulletin updated at (\d\d):(\d\d) HKT (\d\d)/(\d\d)/(\d{4})", xml)
    if not t:
        return None
    issued = pd.Timestamp(int(t[5]), int(t[4]), int(t[3]), int(t[1]), int(t[2]))
    body = re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " | ", re.search(r"CDATA\[(.*?)\]\]", xml, re.S)[1]))
    sec = re.search(r"(?:Weather )?[Ff]orecast for ([^:|]+?):\s*\|(.*?)(?:\| \||Outlook|$)", body)
    if not sec:
        return dict(issued=issued, target=None, fmax=np.nan, head="", text=body[:300])
    head, text = sec[1].strip(), sec[2]
    d = re.search(r"\((?:\w+), (\d+) (\w{3}) (\d{4})\)", head)
    if d:
        target = pd.Timestamp(int(d[3]), MON[d[2]], int(d[1]))
    elif "tomorrow" in head.lower():
        target = issued.normalize() + pd.Timedelta(days=1)
    else:  # today / this morning / this afternoon / this afternoon and tonight / tonight
        target = issued.normalize()
    v = np.nan
    m = re.search(r"maximum temperatures?\D{0,45}?(\d+)(?: to (\d+))? degrees", text)
    if m:
        v = (int(m[1]) + int(m[2])) / 2 if m[2] else int(m[1])
    else:
        m = re.search(r"[Tt]emperatures? (?:will )?rang(?:e|ing) (?:between|from) (\d+) (?:and|to) (\d+) degrees", text)
        if m:
            v = int(m[2])
        else:
            m = re.search(r"[Tt]emperatures? will (?:linger|stay|hover|remain) (?:at )?(?:around |about |near )?(\d+) degrees (?:during the day|at first)", text)
            if m:
                v = int(m[1])
    # 'tonight and tomorrow' etc: the max refers to tomorrow; 'this afternoon and tonight' never states today's max
    return dict(issued=issued, target=target, fmax=v, head=head, text=text[:300])


def load_hko_forecasts():
    rows = [parse_bulletin(open(f, encoding="utf-8", errors="replace").read()) for f in sorted(glob.glob(os.path.join(BENCH, "flw_raw", "*.xml")))]
    b = pd.DataFrame([r for r in rows if r]).drop_duplicates("issued").sort_values("issued")
    return b


# ---------------- main ----------------
def main():
    os.makedirs(OUT, exist_ok=True)
    bul = load_hko_forecasts()
    bul.to_csv(os.path.join(OUT, "hko_forecast_bulletins.csv"), index=False)
    fc = bul.dropna(subset=["fmax"])

    mk = pd.read_csv(os.path.join(PM, "pm_markets_cleaned.csv"), parse_dates=["obs_date"])
    mk = mk[~mk.obs_date.dt.strftime("%Y-%m-%d").isin(EXCLUDED)]
    truth = mk[mk.resolved_yes == 1].set_index("obs_date").hko_abs_max_c
    lo = pd.to_numeric(mk.bracket_low, errors="coerce"); hi = pd.to_numeric(mk.bracket_high, errors="coerce")
    b = mk.bracket.str.extract(r"(-?\d+)")[0].astype(float)
    mk["value"] = np.where(mk.bracket.str.contains("below"), b - 0.5, np.where(mk.bracket.str.contains("higher|above"), b + 1.0, b + 0.5))
    val = mk.set_index(["obs_date", "bracket"]).value

    ph = pd.read_csv(os.path.join(BENCH, "pm_price_10min.csv"), parse_dates=["obs_date", "time_hkt"])
    ph = ph[ph.obs_date.isin(truth.index)]

    hourly = pd.read_csv(HOURLY_FRAME, parse_dates=["time", "date"])
    hourly = hourly[hourly.date.isin(truth.index)][["date", "hour", "time", "rmax"]]

    rows = []
    for d, g in ph.groupby("obs_date"):
        g = g.sort_values("time_hkt")
        br = mk[mk.obs_date == d].bracket
        for h in range(24):
            cut = d + pd.Timedelta(hours=h, minutes=12)   # the h:10 snapshot: HKO's h:00 reading is public by ~h:05
            snap = g[g.time_hkt <= cut].groupby("bracket").yes_price.last().reindex(br)
            if snap.notna().sum() < 0.8 * len(br) or snap.sum() <= 0:
                continue
            p = snap.fillna(0) / snap.fillna(0).sum()
            v = val.loc[d].reindex(p.index)
            f = fc[(fc.target == d) & (fc.issued <= cut)]
            rows.append(dict(date=d, hour=h, market_ev=float((p * v).sum()), market_mode=float(v[p.idxmax()]),
                             market_p_max=float(p.max()), hko_fcst=f.fmax.iloc[-1] if len(f) else np.nan,
                             hko_issued=f.issued.iloc[-1] if len(f) else pd.NaT, actual=truth[d]))
    B = pd.DataFrame(rows).merge(hourly[["date", "hour", "rmax"]], on=["date", "hour"], how="left").rename(columns={"rmax": "running_max"})
    B["hko_fcst_obs"] = np.fmax(B.hko_fcst, B.running_max)

    x = pd.read_csv(os.path.join(REPO, "RESULT", "xgboost", "xgb_hourly_temperature.csv"), parse_dates=["date"])
    B = B.merge(x[["date", "hour", "pred_final_max"]].rename(columns={"pred_final_max": "xgb"}), on=["date", "hour"], how="left")
    B.to_csv(os.path.join(OUT, "hourly_benchmark_forecasts.csv"), index=False)
    print(B.describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
