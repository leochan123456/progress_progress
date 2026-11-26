"""
ISOM 3360 - HK max temperature model: HKO data pull + feature engineering (v2).

What it does
  1. Downloads HKO's daily climate CSVs (13 variables). HKO updates these MONTHLY with checked data, so they lag
     by 3-8 weeks. Cached copies are re-downloaded when older than --refresh-hours, and every download is checked
     (a cut-off file is retried, never cached).
  2. Fills the gap from the last checked day up to yesterday with HKO's Daily Extract (preliminary daily data, the
     source Polymarket settles on): max/min/mean temperature, humidity, dew point, pressure, rainfall, cloud.
     King's Park sunshine, solar radiation, evaporation and wind are not in the Daily Extract, so they stay blank
     for those days (not forward-filled) until HKO's monthly update.
  3. Builds the training set: one row per day t, every attribute known by the end of day t-1.

Usage
  python hko_build_training_set.py                      # 2010-01-01 to the latest available day
  python hko_build_training_set.py --end 2026-10-31     # stop at a date
  python hko_build_training_set.py --no-extract         # checked monthly data only
Outputs (defaults under Storage/HKO): hko_daily_observations.csv, hko_training_set.csv
"""
import argparse, io, json, math, os, time, urllib.request
import numpy as np
import pandas as pd

BASE = "https://data.weather.gov.hk/cis/csvfile"
EXTRACT = "https://www.weather.gov.hk/cis/dailyExtract/dailyExtract_{ym}.xml"
HEADERS = {"User-Agent": "Mozilla/5.0 (isom3360-research)"}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HKO_STORAGE = os.path.join(SCRIPT_DIR, "Storage", "HKO")
DEFAULT_RAW = os.path.join(HKO_STORAGE, "raw_hko")
DEFAULT_OUT = os.path.join(HKO_STORAGE, "processed")
# (station, element) -> column.  HKO = Observatory HQ (Tsim Sha Tsui), KP = King's Park
SERIES = {
    ("HKO", "MAXT"): "max_temp_c", ("HKO", "MINT"): "min_temp_c", ("HKO", "TEMP"): "mean_temp_c",
    ("HKO", "RH"): "mean_rh_pct", ("HKO", "DEW"): "mean_dew_point_c", ("HKO", "MSLP"): "mean_pressure_hpa",
    ("HKO", "RF"): "rainfall_mm", ("HKO", "CLD"): "mean_cloud_pct",
    ("KP", "SUN"): "sunshine_hrs", ("KP", "GSR"): "solar_radiation_mj_m2", ("KP", "EVAP"): "evaporation_mm",
    ("KP", "PDIR"): "wind_dir_deg", ("KP", "WSPD"): "wind_speed_kmh",
}
KP_COLS = ["sunshine_hrs", "solar_radiation_mj_m2", "evaporation_mm", "wind_dir_deg", "wind_speed_kmh"]


def fetch(url, timeout=180, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
                return r.read()
        except Exception as e:
            print(f"  retry {i + 1}/{tries}: {url} ({e})"); time.sleep(3 * (i + 1))
    raise RuntimeError(f"could not download {url}")


def data_lines(text):
    return [l for l in text.splitlines() if l[:4].isdigit()]


def download(raw_dir, refresh_hours):
    """Download the ALL-years file for each series; re-download when older than refresh_hours; validate."""
    os.makedirs(raw_dir, exist_ok=True)
    for (stn, el) in SERIES:
        path = os.path.join(raw_dir, f"daily_{stn}_{el}_ALL.csv")
        if os.path.exists(path) and time.time() - os.path.getmtime(path) < refresh_hours * 3600:
            continue
        url = f"{BASE}/{stn}/ALL/daily_{stn}_{el}_ALL.csv"
        for attempt in range(3):
            text = fetch(url).decode("utf-8-sig", errors="replace")
            lines = data_lines(text)
            # a complete file has thousands of rows and a well-formed last row (year,month,day,value,flag)
            if len(lines) > 1000 and len(lines[-1].split(",")) >= 5:
                open(path, "w", encoding="utf-8").write(text)
                print(f"  {stn}_{el}: {len(lines):,} days, last {lines[-1].split(',')[:3]}")
                break
            print(f"  {stn}_{el}: incomplete download ({len(lines)} rows), retrying")
        else:
            raise RuntimeError(f"{url} kept coming back incomplete")


def load_series(path, name, start, end):
    lines = data_lines(open(path, encoding="utf-8-sig", errors="replace").read())
    df = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=["y", "m", "d", "v", "flag"],
                     dtype=str, keep_default_na=False)
    df = df[df.y.astype(int) >= int(start[:4]) - 1]
    df["date"] = pd.to_datetime(dict(year=df.y.astype(int), month=df.m.astype(int), day=df.d.astype(int)), errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    df = df[(df.date >= start) & (df.date <= end)].copy()
    v = df.v.str.strip()
    out = df[["date"]].copy()
    if name == "rainfall_mm":  # 'Trace' = < 0.05 mm
        out["rain_trace"] = v.str.contains("Trace|微量").astype(int)
        v = v.replace({"Trace": "0", "微量": "0"})
    out[name] = pd.to_numeric(v, errors="coerce")  # '***' (unavailable) -> NaN
    return out


def daily_extract(first_day, last_day):
    """HKO Daily Extract rows between first_day and last_day (preliminary data, updated each working day)."""
    rows, m = [], first_day.replace(day=1)
    while m <= last_day:
        d = json.loads(fetch(EXTRACT.format(ym=f"{m:%Y%m}"), timeout=60))
        num = lambda s: 0.0 if "Trace" in s else (float(s) if s.strip() not in ("", "***") else np.nan)
        for blk in d["stn"]["data"]:
            for r in blk["dayData"]:
                if not r[0].strip().isdigit(): continue
                day = pd.Timestamp(m.year, m.month, int(r[0]))
                if first_day <= day <= last_day:
                    rows.append(dict(date=day, mean_pressure_hpa=num(r[1]), max_temp_c=num(r[2]), mean_temp_c=num(r[3]),
                                     min_temp_c=num(r[4]), mean_dew_point_c=num(r[5]), mean_rh_pct=num(r[6]),
                                     mean_cloud_pct=num(r[7]), rainfall_mm=num(r[8]), rain_trace=int("Trace" in r[8])))
        m = (m + pd.offsets.MonthBegin(1)).normalize()
    return pd.DataFrame(rows).dropna(subset=["max_temp_c"]) if rows else pd.DataFrame()


def build(raw_dir, start, end, use_extract):
    yesterday = pd.Timestamp.now(tz="Asia/Hong_Kong").tz_localize(None).normalize() - pd.Timedelta(days=1)
    end = min(pd.Timestamp(end), yesterday) if end else yesterday
    pull_start = (pd.Timestamp(start) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    series = {name: load_series(os.path.join(raw_dir, f"daily_{stn}_{el}_ALL.csv"), name, pull_start, end)
              for (stn, el), name in SERIES.items()}
    checked_end = min(s.dropna(subset=[s.columns[-1]]).date.max() for s in series.values())
    obs = pd.DataFrame({"date": pd.date_range(pull_start, end, freq="D")})
    for s in series.values():
        obs = obs.merge(s, on="date", how="left")
    obs = obs.set_index("date")

    n_ext = 0
    if use_extract and checked_end < end:
        ext = daily_extract(checked_end + pd.Timedelta(days=1), end)
        if len(ext):
            ext = ext.set_index("date"); idx = ext.index.intersection(obs.index)
            for c in ext.columns: obs.loc[idx, c] = ext.loc[idx, c]
            n_ext = len(idx)
            end = min(end, ext.index.max())
    if n_ext == 0:
        end = min(end, checked_end)   # nothing newer than the checked monthly data
    obs = obs.loc[:end]
    print(f"checked monthly data to {checked_end:%Y-%m-%d}; Daily Extract days: {n_ext}; dataset ends {end:%Y-%m-%d}")

    # Missing values: carry the last observed value forward (past only -> no leakage).
    # King's Park columns are NOT filled past the last checked day (they are simply not published yet).
    f = obs.copy()
    f.loc[:checked_end] = f.loc[:checked_end].ffill()
    hq = [c for c in f.columns if c not in KP_COLS]
    f[hq] = f[hq].ffill()

    X = pd.DataFrame(index=f.index)
    # 1. Past weather on t-1, t-2, t-7
    for col, short in [("max_temp_c", "maxT"), ("min_temp_c", "minT"), ("mean_rh_pct", "rh"),
                       ("mean_pressure_hpa", "pres"), ("rainfall_mm", "rain")]:
        for k in (1, 2, 7):
            X[f"{short}_lag{k}"] = f[col].shift(k)
    for col, short in [("mean_temp_c", "meanT"), ("mean_dew_point_c", "dew"), ("mean_cloud_pct", "cloud"), ("evaporation_mm", "evap")]:
        X[f"{short}_lag1"] = f[col].shift(1)
    # 2. Moving averages over t-1..t-3 and t-1..t-7 (shift first so day t is excluded), day-to-day changes
    for col, short in [("max_temp_c", "maxT"), ("mean_rh_pct", "rh")]:
        X[f"{short}_ma3"] = f[col].shift(1).rolling(3).mean()
        X[f"{short}_ma7"] = f[col].shift(1).rolling(7).mean()
    X["pres_change_1d"] = f["mean_pressure_hpa"].shift(1) - f["mean_pressure_hpa"].shift(2)
    X["maxT_change_1d"] = f["max_temp_c"].shift(1) - f["max_temp_c"].shift(2)
    # 3. Sun and wind on t-1; wind direction is circular -> sin/cos
    X["sun_lag1"] = f["sunshine_hrs"].shift(1)
    X["solar_lag1"] = f["solar_radiation_mj_m2"].shift(1)
    X["wind_speed_lag1"] = f["wind_speed_kmh"].shift(1)
    rad = np.deg2rad(f["wind_dir_deg"].shift(1))
    X["wind_dir_sin_lag1"] = np.sin(rad).round(4)
    X["wind_dir_cos_lag1"] = np.cos(rad).round(4)
    # 4. Season
    doy = f.index.dayofyear
    days_in_year = np.where(f.index.is_leap_year, 366, 365)
    X["sin_day"] = np.sin(2 * math.pi * doy / days_in_year).round(4)
    X["cos_day"] = np.cos(2 * math.pi * doy / days_in_year).round(4)
    X["month"] = f.index.month
    X["year"] = f.index.year
    # Target (Y): actual max temp on day t (un-filled)
    X["target_max_temp_c"] = obs["max_temp_c"]

    X = X.loc[start:end].round(2).reset_index()
    obs = obs.loc[start:end].reset_index()
    return obs, X


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None, help="default: latest available day (yesterday, HKT)")
    ap.add_argument("--raw", default=DEFAULT_RAW, help="folder for the downloaded HKO files")
    ap.add_argument("--out", default=DEFAULT_OUT, help="folder for the two output CSVs")
    ap.add_argument("--refresh-hours", type=float, default=12, help="re-download cached HKO files older than this")
    ap.add_argument("--no-extract", action="store_true", help="checked monthly data only (no Daily Extract days)")
    a = ap.parse_args()
    download(a.raw, a.refresh_hours)
    obs, train = build(a.raw, a.start, a.end, not a.no_extract)
    os.makedirs(a.out, exist_ok=True)
    obs.to_csv(os.path.join(a.out, "hko_daily_observations.csv"), index=False)
    train.to_csv(os.path.join(a.out, "hko_training_set.csv"), index=False)
    print("observations:", obs.shape, " training set:", train.shape)
    blanks = train.isna().sum()
    print("blank cells:", blanks[blanks > 0].to_dict() or "none")
