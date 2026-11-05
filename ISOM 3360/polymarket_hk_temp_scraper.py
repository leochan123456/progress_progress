"""
Scrape Polymarket "Highest temperature in Hong Kong on <date>" markets (series 11312):
market metadata + hourly price history for every bracket, plus the HKO Daily Extract
"Absolute Daily Max" each market settles on.

Output: Storage/PM/pm_markets.csv, Storage/PM/pm_price_history_hourly.csv, Storage/PM/hko_daily_extract.csv
"""
import json, os, re, time, urllib.request, concurrent.futures as cf
from datetime import datetime, timezone, timedelta
import pandas as pd

H = {"User-Agent": "Mozilla/5.0 (isom3360-research)"}
HKT = timezone(timedelta(hours=8))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PM_STORAGE = os.path.join(SCRIPT_DIR, "Storage", "PM")
os.makedirs(PM_STORAGE, exist_ok=True)


def get(url, tries=4):
    for i in range(tries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=60).read())
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(2 * (i + 1))


# ---------- 1. events + markets ----------
events, off = [], 0
while True:
    batch = get(f"https://gamma-api.polymarket.com/events?series_id=11312&limit=100&offset={off}&order=startDate&ascending=true")
    events += batch; off += 100
    if len(batch) < 100: break

def bracket_bounds(label):
    """'26°C or below' -> (-inf, 27) ; '31°C' -> (31, 32) ; '36°C or higher' -> (36, inf). Upper bound exclusive."""
    n = int(re.search(r"(-?\d+)", label).group(1))
    if "below" in label: return float("-inf"), n + 1
    if "higher" in label or "above" in label: return n, float("inf")
    return n, n + 1

rows = []
for e in events:
    m_date = re.search(r"on-(\w+-\d+-\d{4})$", e["slug"]).group(1)
    obs_date = datetime.strptime(m_date, "%B-%d-%Y").date()
    for m in e.get("markets", []):
        toks = json.loads(m.get("clobTokenIds") or "[]")
        if not toks: continue
        prices = json.loads(m.get("outcomePrices") or "[null,null]")
        lo, hi = bracket_bounds(m["groupItemTitle"])
        fs = m.get("feeSchedule") or {}
        rows.append(dict(
            obs_date=obs_date, event_slug=e["slug"], bracket=m["groupItemTitle"], bracket_low=lo, bracket_high=hi,
            yes_token=toks[0], market_open_utc=m.get("startDate") or e.get("startDate"), market_close_utc=m.get("endDate") or e.get("endDate"),
            closed=m.get("closed"), volume_usd=round(float(m.get("volume") or 0), 2),
            final_yes_price=float(prices[0]) if prices[0] is not None else None,
            resolved_yes=(1 if m.get("closed") and prices[0] == "1" else (0 if m.get("closed") and prices[0] == "0" else None)),
            fees_enabled=m.get("feesEnabled"), fee_rate=fs.get("rate"), fee_taker_only=fs.get("takerOnly"),
            tick_size=m.get("orderPriceMinTickSize"), min_order=m.get("orderMinSize")))
mk = pd.DataFrame(rows)
print("events:", len(events), " bracket markets:", len(mk))

# ---------- 2. hourly price history (Yes token; No = 1 - Yes) ----------
def history(row):
    t0 = int(pd.Timestamp(row.market_open_utc).timestamp()) - 3600
    t1 = int(pd.Timestamp(row.market_close_utc).timestamp()) + 3 * 86400
    try:
        h = get(f"https://clob.polymarket.com/prices-history?market={row.yes_token}&startTs={t0}&endTs={t1}&fidelity=60").get("history", [])
    except Exception as ex:
        print("fail", row.event_slug, row.bracket, ex); h = []
    return [(row.obs_date, row.bracket, p["t"], p["p"]) for p in h]

with cf.ThreadPoolExecutor(6) as ex:
    out = [r for res in ex.map(history, mk.itertuples(index=False)) for r in res]
ph = pd.DataFrame(out, columns=["obs_date", "bracket", "ts", "yes_price"])
ph["time_utc"] = pd.to_datetime(ph.ts, unit="s", utc=True)
ph["time_hkt"] = ph.time_utc.dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
# hours until 23:59 HKT on the observation day (the day is over -> max is known)
day_end = pd.to_datetime(ph.obs_date.astype(str)) + pd.Timedelta(hours=23, minutes=59)
ph["hours_to_day_end"] = ((day_end - ph.time_hkt).dt.total_seconds() / 3600).round(1)
ph["time_utc"] = ph.time_utc.dt.tz_localize(None)
ph = ph.drop(columns="ts").sort_values(["obs_date", "bracket", "time_hkt"])

# ---------- 3. HKO Daily Extract (the settlement source) ----------
months = sorted({(d.year, d.month) for d in mk.obs_date})
de = []
for y, m in months:
    try:
        d = get(f"https://www.weather.gov.hk/cis/dailyExtract/dailyExtract_{y}{m:02d}.xml")
    except Exception:
        continue
    for blk in d["stn"]["data"]:
        for r in blk["dayData"]:
            if not r[0].strip().isdigit(): continue
            try: day = datetime(y, m, int(r[0])).date()
            except ValueError: continue
            num = lambda s: (0.0 if "Trace" in s else (float(s) if s.strip() not in ("", "***") else None))
            de.append(dict(date=day, pressure_hpa=num(r[1]), abs_daily_max_c=num(r[2]), mean_temp_c=num(r[3]),
                           abs_daily_min_c=num(r[4]), dew_point_c=num(r[5]), rh_pct=num(r[6]), cloud_pct=num(r[7]), rain_mm=num(r[8])))
de = pd.DataFrame(de)

mk = mk.merge(de[["date", "abs_daily_max_c"]].rename(columns={"date": "obs_date", "abs_daily_max_c": "hko_abs_max_c"}), on="obs_date", how="left")
mk.to_csv(os.path.join(PM_STORAGE, "pm_markets.csv"), index=False)
ph.to_csv(os.path.join(PM_STORAGE, "pm_price_history_hourly.csv"), index=False)
de.to_csv(os.path.join(PM_STORAGE, "hko_daily_extract.csv"), index=False)
print("price points:", len(ph), " daily extract days:", len(de))
print("saved directory:", PM_STORAGE)
