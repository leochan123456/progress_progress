"""
Hourly HKO observations from the data.gov.hk historical archive (snapshots of HKO's
'regional weather' CSVs, archived every ~10 min since mid-2020). Keeps the first snapshot
in each clock hour, for every station.

Usage: python hko_hourly_scraper.py 2026-03-01 2026-11-04 [outdir]
Output: <outdir>/hko_hourly_<element>.csv (long format: obs_time_hkt, station, values...)
"""
import sys, os, re, json, time, urllib.request, urllib.parse, concurrent.futures as cf, io
import pandas as pd

START, END = sys.argv[1], sys.argv[2]
# the archive can only list fully archived days -> clamp to yesterday (HKT)
END = min(pd.Timestamp(END), pd.Timestamp.now(tz="Asia/Hong_Kong").tz_localize(None).normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
OUT = sys.argv[3] if len(sys.argv) > 3 else "."
H = {"User-Agent": "Mozilla/5.0 (isom3360-research)"}
BASE = "https://data.weather.gov.hk/weatherAPI/hko_data/regional-weather/"
FILES = {
    "temperature": "latest_1min_temperature.csv",          # 1-min mean air temp, ~27 stations
    "since_midnight_maxmin": "latest_since_midnight_maxmin.csv",  # running max/min since 00:00
    "humidity": "latest_1min_humidity.csv",
    "pressure": "latest_1min_pressure.csv",
    "solar": "latest_1min_solar.csv",
    "wind": "latest_10min_wind.csv",
}


def get(u, tries=4):
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60).read()
        except Exception:
            if i == tries - 1: raise
            time.sleep(1.5 * (i + 1))


def versions(url):
    enc = urllib.parse.quote(url, safe="")
    ts, cur = [], pd.Timestamp(START)
    while cur <= pd.Timestamp(END):
        nxt = min(cur + pd.DateOffset(months=1) - pd.Timedelta(days=1), pd.Timestamp(END))
        d = json.loads(get(f"https://api.data.gov.hk/v1/historical-archive/list-file-versions?url={enc}"
                           f"&start={cur:%Y%m%d}&end={nxt:%Y%m%d}"))
        ts += d.get("timestamps", [])
        cur = nxt + pd.Timedelta(days=1)
    # The HH:00 reading is published ~HH:08; snapshots taken at HH:00-HH:04 still hold the HH-1:50 reading.
    # So take the first snapshot at or after HH:05 in each hour (fall back to the hour's first snapshot).
    first, fallback = {}, {}
    for t in sorted(ts):
        fallback.setdefault(t[:11], t)
        if int(t[11:13]) >= 5: first.setdefault(t[:11], t)
    return enc, sorted({**fallback, **first}.values())


def fetch(args):
    enc, t, cache = args
    p = os.path.join(cache, t + ".csv")
    if os.path.exists(p) and os.path.getsize(p) > 50:
        return t, open(p, encoding="utf-8").read()
    url = (f"https://s3-ap-southeast-1.amazonaws.com/historical-resource-archive/{t[:4]}/{t[4:6]}/{t[6:8]}/"
           f"{urllib.parse.quote(enc, safe='')}/{t[9:]}")
    try:
        txt = get(url).decode("utf-8-sig", "replace")
    except Exception:
        return t, None
    open(p, "w", encoding="utf-8").write(txt)
    return t, txt


ONLY = os.environ.get("ONLY")  # e.g. ONLY=wind to redo one element
for name, f in FILES.items():
    if ONLY and name not in ONLY.split(","): continue
    t0 = time.time()
    cache = os.path.join(OUT, "raw", name); os.makedirs(cache, exist_ok=True)
    enc, snaps = versions(BASE + f)
    frames, failed = [], 0
    with cf.ThreadPoolExecutor(24) as ex:
        for t, txt in ex.map(fetch, [(enc, t, cache) for t in snaps]):
            if not txt: failed += 1; continue
            # calm wind is written 'station,Calm,Calm,<gust>,' (one extra field) -> 'station,Calm,0,<gust>'
            txt = re.sub(r",Calm,Calm,([^,\n]*),[ \t]*$", r",Calm,0,\1", txt, flags=re.M)
            df = pd.read_csv(io.StringIO(txt), dtype=str, index_col=False)
            df.insert(0, "snapshot", t)
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={df.columns[1]: "obs_time", df.columns[2]: "station"})
    df["obs_time_hkt"] = pd.to_datetime(df.obs_time, format="%Y%m%d%H%M", errors="coerce")
    for c in df.columns[3:]:
        if c != "obs_time_hkt":
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().mean() > 0.5: df[c] = num
    df = df.drop(columns=["obs_time"]).drop_duplicates(["obs_time_hkt", "station"])
    cols = ["obs_time_hkt", "station"] + [c for c in df.columns if c not in ("obs_time_hkt", "station", "snapshot")] + ["snapshot"]
    df[cols].sort_values(["obs_time_hkt", "station"]).to_csv(os.path.join(OUT, f"hko_hourly_{name}.csv"), index=False)
    print(f"{name}: {len(snaps)} hourly snapshots, {failed} failed, {len(df)} rows, {time.time() - t0:.0f}s", flush=True)
