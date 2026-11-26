"""Download HKO Local Weather Forecast bulletins (hourly snapshots) from the data.gov.hk historical archive.
Usage: python fetch_flw.py START END OUTDIR   (dates YYYYMMDD)"""
import json, os, sys, time, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

URL = "https://rss.weather.gov.hk/rss/LocalWeatherForecast.xml"
API = "https://api.data.gov.hk/v1/historical-archive"
S3 = "https://s3-ap-southeast-1.amazonaws.com/historical-resource-archive/{y}/{m}/{d}/{enc}/{hm}"
ENC = urllib.parse.quote(urllib.parse.quote(URL, safe=""), safe="")


def get(u, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
                return r.read()
        except Exception as e:
            time.sleep(2 * (i + 1))
    raise RuntimeError(u)


def main(start, end, out):
    os.makedirs(out, exist_ok=True)
    stamps = []
    for m0 in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        a = max(m0, pd.Timestamp(start)); b = min(m0 + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        q = urllib.parse.urlencode(dict(url=URL, start=f"{a:%Y%m%d}", end=f"{b:%Y%m%d}"))
        stamps += json.loads(get(f"{API}/list-file-versions?{q}"))["timestamps"]
    print(len(stamps), "versions")

    def one(ts):
        p = os.path.join(out, ts + ".xml")
        if os.path.exists(p) and os.path.getsize(p) > 500:
            return
        d, hm = ts.split("-")
        open(p, "wb").write(get(S3.format(y=d[:4], m=d[4:6], d=d[6:], enc=ENC, hm=hm)))

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(one, stamps))
    print("done", len(os.listdir(out)))


if __name__ == "__main__":
    main(*sys.argv[1:4])
