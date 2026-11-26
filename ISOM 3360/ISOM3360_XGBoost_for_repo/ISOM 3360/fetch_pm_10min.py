"""Polymarket HK temperature: 10-minute Yes price history for every bracket (CLOB prices-history, fidelity=10).
Usage: python fetch_pm_10min.py MARKETS_CSV OUT_CSV"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def get(u, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            time.sleep(2 * (i + 1))
    return {"history": []}


def one(r):
    d = pd.Timestamp(r.obs_date).tz_localize("Asia/Hong_Kong")
    t0, t1 = int((d - pd.Timedelta(days=3)).timestamp()), int((d + pd.Timedelta(days=1, hours=2)).timestamp())
    out = []
    for a in range(t0, t1, 86400):   # one day per request keeps responses small
        h = get(f"https://clob.polymarket.com/prices-history?market={r.yes_token}&startTs={a}&endTs={min(a + 86400, t1)}&fidelity=10")
        out += [(r.obs_date, r.bracket, x["p"], x["t"]) for x in h.get("history", [])]
    return out


m = pd.read_csv(sys.argv[1])
with ThreadPoolExecutor(12) as ex:
    rows = [x for res in ex.map(one, m.itertuples()) for x in res]
p = pd.DataFrame(rows, columns=["obs_date", "bracket", "yes_price", "t"]).drop_duplicates(["obs_date", "bracket", "t"])
p["time_hkt"] = pd.to_datetime(p.t, unit="s", utc=True).dt.tz_convert("Asia/Hong_Kong").dt.tz_localize(None)
p.drop(columns="t").sort_values(["obs_date", "bracket", "time_hkt"]).to_csv(sys.argv[2], index=False)
print(len(p), "rows", p.obs_date.nunique(), "days")
