"""
ISOM 3360 - HK max temperature model: data pull + feature engineering.

Downloads HKO daily climate CSVs (open data), merges them into one row per day,
and builds a training set where every attribute is known by the end of day t-1.

Usage:  python hko_build_training_set.py [--start 2010-01-01] [--end 2026-09-30] [--raw Storage/HKO/raw_hko]
Output: Storage/HKO/processed/hko_daily_observations.csv, Storage/HKO/processed/hko_training_set.csv
"""
import argparse, io, math, os, time, urllib.request
import numpy as np
import pandas as pd

BASE = "https://data.weather.gov.hk/cis/csvfile"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HKO_STORAGE = os.path.join(SCRIPT_DIR, "Storage", "HKO")
HKO_PROCESSED_STORAGE = os.path.join(HKO_STORAGE, "processed")
# (station, element) -> column name.  HKO = Observatory HQ (Tsim Sha Tsui), KP = King's Park
SERIES = {
    ("HKO", "MAXT"): "max_temp_c",
    ("HKO", "MINT"): "min_temp_c",
    ("HKO", "TEMP"): "mean_temp_c",
    ("HKO", "RH"):   "mean_rh_pct",
    ("HKO", "DEW"):  "mean_dew_point_c",
    ("HKO", "MSLP"): "mean_pressure_hpa",
    ("HKO", "RF"):   "rainfall_mm",
    ("HKO", "CLD"):  "mean_cloud_pct",
    ("KP", "SUN"):   "sunshine_hrs",
    ("KP", "GSR"):   "solar_radiation_mj_m2",
    ("KP", "EVAP"):  "evaporation_mm",
    ("KP", "PDIR"):  "wind_dir_deg",
    ("KP", "WSPD"):  "wind_speed_kmh",
}


def download(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    for (stn, el) in SERIES:
        path = os.path.join(raw_dir, f"daily_{stn}_{el}_ALL.csv")
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            continue  # cached
        url = f"{BASE}/{stn}/ALL/daily_{stn}_{el}_ALL.csv"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=180) as r:
                    open(path, "wb").write(r.read())
                break
            except Exception as e:
                print(f"retry {url}: {e}"); time.sleep(3)


def load_series(path, name, start, end):
    lines = [l for l in open(path, encoding="utf-8-sig", errors="replace").read().splitlines() if l[:4].isdigit()]
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=["y", "m", "d", "v", "flag"],
                     dtype=str, keep_default_na=False)
    df = df[df.y.astype(int) >= int(start[:4]) - 1]
    df["date"] = pd.to_datetime(dict(year=df.y.astype(int), month=df.m.astype(int), day=df.d.astype(int)), errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df.date >= start) & (df.date <= end)].copy()
    v = df.v.str.strip()
    out = df[["date"]].copy()
    if name == "rainfall_mm":  # 'Trace' = < 0.05 mm
        out["rain_trace"] = v.str.contains("Trace|微量").astype(int)
        v = v.replace({"Trace": "0", "微量": "0"})
    out[name] = pd.to_numeric(v, errors="coerce")  # '***' (unavailable) -> NaN
    return out


def build(raw_dir, start, end):
    # pull 30 extra days before `start` so lag/rolling features exist on day one
    pull_start = (pd.Timestamp(start) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    obs = pd.DataFrame({"date": pd.date_range(pull_start, end, freq="D")})
    for (stn, el), name in SERIES.items():
        obs = obs.merge(load_series(os.path.join(raw_dir, f"daily_{stn}_{el}_ALL.csv"), name, pull_start, end),
                        on="date", how="left")

    # Missing values: carry the last observed value forward (uses only the past -> no leakage)
    f = obs.set_index("date").ffill()

    X = pd.DataFrame(index=f.index)
    # 1. Past weather on t-1, t-2, t-7
    for col, short in [("max_temp_c", "maxT"), ("min_temp_c", "minT"), ("mean_rh_pct", "rh"),
                       ("mean_pressure_hpa", "pres"), ("rainfall_mm", "rain")]:
        for k in (1, 2, 7):
            X[f"{short}_lag{k}"] = f[col].shift(k)
    # extra same-day-before observations (t-1 only)
    for col, short in [("mean_temp_c", "meanT"), ("mean_dew_point_c", "dew"), ("mean_cloud_pct", "cloud"),
                       ("evaporation_mm", "evap")]:
        X[f"{short}_lag1"] = f[col].shift(1)
    # 2. Engineered: moving averages over t-1..t-3 and t-1..t-7 (shift first so day t is excluded)
    for col, short in [("max_temp_c", "maxT"), ("mean_rh_pct", "rh")]:
        X[f"{short}_ma3"] = f[col].shift(1).rolling(3).mean()
        X[f"{short}_ma7"] = f[col].shift(1).rolling(7).mean()
    X["pres_change_1d"] = f["mean_pressure_hpa"].shift(1) - f["mean_pressure_hpa"].shift(2)
    X["maxT_change_1d"] = f["max_temp_c"].shift(1) - f["max_temp_c"].shift(2)
    # 3. Sun and wind (t-1). Wind direction is circular -> sin/cos so 350 deg and 10 deg are close
    X["sun_lag1"] = f["sunshine_hrs"].shift(1)
    X["solar_lag1"] = f["solar_radiation_mj_m2"].shift(1)
    X["wind_speed_lag1"] = f["wind_speed_kmh"].shift(1)
    rad = np.deg2rad(f["wind_dir_deg"].shift(1))
    X["wind_dir_sin_lag1"] = np.sin(rad).round(4)
    X["wind_dir_cos_lag1"] = np.cos(rad).round(4)
    # 4. Season: day of year as a smooth cycle (known in advance, no leakage)
    doy = f.index.dayofyear
    days_in_year = np.where(f.index.is_leap_year, 366, 365)
    X["sin_day"] = np.sin(2 * math.pi * doy / days_in_year).round(4)
    X["cos_day"] = np.cos(2 * math.pi * doy / days_in_year).round(4)
    X["month"] = f.index.month
    X["year"] = f.index.year
    # Target (Y): actual max temp on day t (from the un-imputed observations)
    X["target_max_temp_c"] = obs.set_index("date")["max_temp_c"]

    X = X.loc[start:end].round(2).reset_index()
    obs = obs[obs.date >= start].reset_index(drop=True)
    return obs, X


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--raw", default=os.path.join(HKO_STORAGE, "raw_hko"))
    a = ap.parse_args()
    download(a.raw)
    obs, train = build(a.raw, a.start, a.end)
    os.makedirs(HKO_PROCESSED_STORAGE, exist_ok=True)
    observations_path = os.path.join(HKO_PROCESSED_STORAGE, "hko_daily_observations.csv")
    training_path = os.path.join(HKO_PROCESSED_STORAGE, "hko_training_set.csv")
    obs.to_csv(observations_path, index=False)
    train.to_csv(training_path, index=False)
    print("saved:", observations_path)
    print("saved:", training_path)
    print("observations:", obs.shape, " training set:", train.shape)
    print("missing in observations:\n", obs.isna().sum()[obs.isna().sum() > 0])
    print("missing in training set:", int(train.isna().sum().sum()))
