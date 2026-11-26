"""
Shared feature code for the HK max-temperature model.

Used by BOTH the weekly report pipeline (build_dataset -> train_model -> evaluate_report) and the hourly
live_signals job, so training and live predictions compute every feature the same way.
"""
import io, re
import numpy as np
import pandas as pd

ELEMENTS = ["temperature", "since_midnight_maxmin", "humidity", "pressure", "solar", "wind"]
LIVE_FILES = {
    "temperature": "latest_1min_temperature.csv",
    "since_midnight_maxmin": "latest_since_midnight_maxmin.csv",
    "humidity": "latest_1min_humidity.csv",
    "pressure": "latest_1min_pressure.csv",
    "solar": "latest_1min_solar.csv",
    "wind": "latest_10min_wind.csv",
}
HKO_LIVE_BASE = "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/"

CAP = 8  # classes 0..CAP ; CAP = "CAP or more brackets above the running max"

HOURLY_FEATURES = [
    "hour", "temp", "rmax", "rmin", "rmax_frac", "gap_to_rmax", "temp_chg_1h", "temp_chg_3h", "rh", "rh_chg_3h",
    "pres", "pres_chg_3h", "pres_chg_24h", "kp_temp", "kp_minus_hko", "solar", "solar_cum_today", "wspd", "gust",
    "wdir_sin", "wdir_cos", "calm", "stations_rmax_median", "stations_rmax_p90", "stations_med_minus_hko",
    "y_max", "y_min", "y_mean", "y_rh", "y_pres", "y_cloud", "y_rain_log", "y2_max", "temp_minus_ymax", "sin_day", "cos_day",
    "prior_mu", "prior_minus_rmax", "prior_minus_temp"]

DEG = {"North": 0, "Northeast": 45, "East": 90, "Southeast": 135, "South": 180, "Southwest": 225, "West": 270, "Northwest": 315}


# ----------------------------------------------------------------------------- parsing HKO CSV snapshots
def parse_hko_snapshot(txt):
    """One HKO regional-weather CSV snapshot (archive or live) -> long DataFrame (obs_time_hkt, station, values)."""
    # calm wind is written 'station,Calm,Calm,<gust>,' -> 'station,Calm,0,<gust>'
    txt = re.sub(r",Calm,Calm,([^,\n]*),[ \t]*$", r",Calm,0,\1", txt.lstrip("﻿"), flags=re.M)
    df = pd.read_csv(io.StringIO(txt), dtype=str, index_col=False)
    df = df.rename(columns={df.columns[0]: "obs_time", df.columns[1]: "station"})
    df["obs_time_hkt"] = pd.to_datetime(df.obs_time, format="%Y%m%d%H%M", errors="coerce")
    for c in df.columns[2:]:
        if c in ("obs_time_hkt",): continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() > 0.5: df[c] = num
    return df.drop(columns=["obs_time"])


def normalise_long(d, floor_to_hour=False):
    """Merge header spellings that changed over the years ('MaximumAir' vs 'Maximum Air'); keep on-the-hour rows."""
    groups = {}
    for c in d.columns: groups.setdefault(c.replace(" ", ""), []).append(c)
    for _, cols in groups.items():
        if len(cols) > 1:
            d[cols[-1]] = d[cols].bfill(axis=1).iloc[:, 0]
            d = d.drop(columns=cols[:-1])
    d = d.copy()
    d["obs_time_hkt"] = pd.to_datetime(d.obs_time_hkt)
    if floor_to_hour:   # live job: a :10 reading stands in for :00 if the run is late
        d["obs_time_hkt"] = d.obs_time_hkt.dt.floor("h")
        d = d.drop_duplicates(["obs_time_hkt", "station"], keep="first")
    else:
        d = d[d.obs_time_hkt.dt.minute == 0]
    return d


def _col(d, key):
    return [c for c in d.columns if key in c][0]


# ----------------------------------------------------------------------------- hourly frame
def build_hourly_frame(long):
    """long: dict element -> long DataFrame. Returns an hourly grid (index = time) with HKO HQ + King's Park features."""
    t, mm, rh, pr, so, wi = (long[e] for e in ELEMENTS)

    def st(d, station, mapping):
        x = d[d.station == station].drop_duplicates("obs_time_hkt").set_index("obs_time_hkt")
        return x[list(mapping)].rename(columns=mapping)

    H = pd.concat([
        st(t, "HK Observatory", {_col(t, "Air Temp"): "temp"}),
        st(mm, "HK Observatory", {_col(mm, "Maximum"): "rmax", _col(mm, "Minimum"): "rmin"}),
        st(rh, "HK Observatory", {_col(rh, "Humidity"): "rh"}),
        st(pr, "HK Observatory", {_col(pr, "Pressure"): "pres"}),
        st(t, "King's Park", {_col(t, "Air Temp"): "kp_temp"}),
        st(so, "King's Park", {_col(so, "Global"): "solar"}),
        st(wi, "King's Park", {_col(wi, "Direction"): "wdir", _col(wi, "Mean Speed"): "wspd", _col(wi, "Gust"): "gust"}),
    ], axis=1, sort=True)
    allmax = mm[mm.obs_time_hkt.dt.hour != 0].pivot_table(index="obs_time_hkt", columns="station", values=_col(mm, "Maximum"))
    H["stations_rmax_median"] = allmax.median(axis=1)
    H["stations_rmax_p90"] = allmax.quantile(0.9, axis=1)
    H = H.reindex(pd.date_range(H.index.min().floor("D"), H.index.max().ceil("h"), freq="h"))
    H.index.name = "time"
    h0 = H.index.hour == 0
    # at 00:00 'since midnight' still shows the PREVIOUS day's figures: keep them as prev_day_max, reset the running max
    H["prev_day_max"] = np.where(h0, H.rmax, np.nan)
    H.loc[h0, "rmax"] = H.loc[h0, "temp"]; H.loc[h0, "rmin"] = H.loc[h0, "temp"]
    H["temp_chg_1h"] = H.temp - H.temp.shift(1); H["temp_chg_3h"] = H.temp - H.temp.shift(3)
    H["rh_chg_3h"] = H.rh - H.rh.shift(3)
    H["pres_chg_3h"] = H.pres - H.pres.shift(3); H["pres_chg_24h"] = H.pres - H.pres.shift(24)
    H["date"] = H.index.normalize(); H["hour"] = H.index.hour
    H["solar_cum_today"] = H.solar.fillna(0).groupby(H.date).cumsum()
    H["gap_to_rmax"] = H.temp - H.rmax
    H["rmax_frac"] = H.rmax - np.floor(H.rmax)
    H["kp_minus_hko"] = H.kp_temp - H.temp
    H["stations_med_minus_hko"] = H.stations_rmax_median - H.rmax
    wd = H.wdir.map(DEG)
    H["wdir_sin"], H["wdir_cos"] = np.sin(np.deg2rad(wd)), np.cos(np.deg2rad(wd))
    H["calm"] = (H.wdir == "Calm").astype(int)
    return H.drop(columns=["wdir"])


# ----------------------------------------------------------------------------- daily table
DAILY_COLS = ["max_temp_c", "min_temp_c", "mean_temp_c", "mean_rh_pct", "mean_pressure_hpa", "rainfall_mm", "mean_cloud_pct"]


def build_daily_table(daily_obs_csv, extract_frames=(), hourly=None):
    """Daily table indexed by date. Priority: climate CSV > HKO Daily Extract > values derived from hourly readings."""
    dd = pd.read_csv(daily_obs_csv, parse_dates=["date"])[["date"] + DAILY_COLS]
    parts = [dd]
    for de in extract_frames:
        de = de.rename(columns={"abs_daily_max_c": "max_temp_c", "abs_daily_min_c": "min_temp_c", "rh_pct": "mean_rh_pct",
                                "pressure_hpa": "mean_pressure_hpa", "rain_mm": "rainfall_mm", "cloud_pct": "mean_cloud_pct"})
        de["date"] = pd.to_datetime(de.date)
        parts.append(de[["date"] + DAILY_COLS])
    daily = pd.concat(parts).drop_duplicates("date", keep="first").sort_values("date").set_index("date")
    if hourly is not None and len(hourly):
        # fill days not yet published (e.g. yesterday at the weekend) from the hourly readings
        g = hourly.groupby("date")
        hd = pd.DataFrame({"min_temp_c": g.rmin.min(), "mean_temp_c": g.temp.mean(), "mean_rh_pct": g.rh.mean(),
                           "mean_pressure_hpa": g.pres.mean()})
        nxt = hourly[hourly.hour == 0].set_index("date").prev_day_max      # 00:00 of day d+1 = final max of day d
        nxt.index = nxt.index - pd.Timedelta(days=1)
        hd["max_temp_c"] = nxt.reindex(hd.index).fillna(g.rmax.max())
        complete = g.size() >= 20
        hd = hd[complete.reindex(hd.index).fillna(False)]
        missing = hd.index.difference(daily.index)
        if len(missing): daily = pd.concat([daily, hd.loc[missing]]).sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    return daily


def prior_features(daily):
    """Inputs for the daily prior (day t's max from data up to t-1). Columns the Daily Extract also has."""
    P = pd.DataFrame(index=daily.index)
    for c, s in [("max_temp_c", "mx"), ("min_temp_c", "mn"), ("mean_temp_c", "me"), ("mean_rh_pct", "rh"),
                 ("mean_pressure_hpa", "pr"), ("mean_cloud_pct", "cl")]:
        for k in (1, 2, 7): P[f"{s}_l{k}"] = daily[c].shift(k)
    P["rain_l1"] = np.log1p(daily.rainfall_mm.shift(1)); P["rain_l2"] = np.log1p(daily.rainfall_mm.shift(2))
    P["mx_ma3"] = daily.max_temp_c.shift(1).rolling(3).mean(); P["mx_ma7"] = daily.max_temp_c.shift(1).rolling(7).mean()
    P["pr_chg"] = P.pr_l1 - P.pr_l2
    doy = P.index.dayofyear
    P["sin_day"] = np.sin(2 * np.pi * doy / 365.25); P["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    return P


def yesterday_features(daily):
    Y = pd.DataFrame(index=daily.index)
    Y["final_max"] = daily.max_temp_c
    for c, s in [("max_temp_c", "y_max"), ("min_temp_c", "y_min"), ("mean_temp_c", "y_mean"), ("mean_rh_pct", "y_rh"),
                 ("mean_pressure_hpa", "y_pres"), ("mean_cloud_pct", "y_cloud")]:
        Y[s] = daily[c].shift(1)
    Y["y_rain_log"] = np.log1p(daily.rainfall_mm.shift(1))
    Y["y2_max"] = daily.max_temp_c.shift(2)
    doy = Y.index.dayofyear
    Y["sin_day"] = np.sin(2 * np.pi * doy / 365.25); Y["cos_day"] = np.cos(2 * np.pi * doy / 365.25)
    return Y


def assemble(H, Y, prior_mu):
    """Hourly rows with every model feature. prior_mu: Series indexed by date."""
    Y = Y.copy(); Y["prior_mu"] = prior_mu.reindex(Y.index)
    D = H.reset_index().merge(Y, left_on="date", right_index=True, how="left")
    D["temp_minus_ymax"] = D.temp - D.y_max
    D["prior_minus_rmax"] = D.prior_mu - D.rmax
    D["prior_minus_temp"] = D.prior_mu - D.temp
    return D


def bracket_probs(p_d, rmax, brackets):
    """Map class probabilities P(d) onto market brackets.
    brackets: DataFrame with columns bracket, lo, hi (hi exclusive; +-inf for open ends). Returns Series bracket -> prob."""
    k0 = int(np.floor(rmax)); out = pd.Series(0.0, index=brackets.bracket.values)
    for d, p in enumerate(p_d):
        k = k0 + d
        hit = brackets[brackets.hi > k] if d == len(p_d) - 1 else brackets[(brackets.lo <= k) & (brackets.hi > k)]
        if len(hit): out[hit.bracket.iloc[0]] += p
    return out / out.sum() if out.sum() > 0 else out


def bracket_bounds(label):
    """'26°C or below' -> (-inf, 27); '31°C' -> (31, 32); '36°C or higher' -> (36, inf). Upper bound exclusive."""
    n = int(re.search(r"(-?\d+)", label).group(1))
    if "below" in label: return float("-inf"), n + 1
    if "higher" in label or "above" in label: return n, float("inf")
    return n, n + 1


def taker_fee(price, rate=0.05):
    return rate * price * (1 - price)
