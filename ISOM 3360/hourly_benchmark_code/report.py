"""Chart + Excel for the hourly benchmark comparison."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
test = pd.read_csv(os.path.join(OUT, "errors_by_hour_test.csv"))
full = pd.read_csv(os.path.join(OUT, "errors_by_hour_full_period.csv"))
B = pd.read_csv(os.path.join(OUT, "hourly_benchmark_forecasts.csv"), parse_dates=["date", "hko_issued"])

SERIES = [("XGBoost hourly (Ryan)", "#2a78d6"), ("Polymarket expected value", "#eb6834"),
          ("HKO forecast + observed max", "#1baf7a"), ("HKO forecast only", "#eda100")]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

t = test[test.hour != "all"].copy(); t["hour"] = t.hour.astype(int)
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200, facecolor=SURF)
ax.set_facecolor(SURF)
ends = {}
for name, col in SERIES:
    s = t[t.forecaster == name].sort_values("hour")
    ax.plot(s.hour, s.RMSE, color=col, lw=2, marker="o", ms=4, label=name, solid_capstyle="round")
    ends[name] = (s.hour.iloc[-1], s.RMSE.iloc[-1])
# direct labels at the right edge, nudged apart
ys = sorted([(v[1], n) for n, v in ends.items()])
placed = []
for y, n in ys:
    y2 = max([y] + [p + 0.09 for p in placed])
    placed.append(y2)
    ax.annotate(n, (23, y), xytext=(23.5, y2), textcoords="data", va="center", fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8))
ax.hlines(0.289, -0.5, 23, color=INK2, lw=0.8, ls=(0, (3, 3)))
ax.text(0.2, 0.31, "0.29 floor for any forecast that only names the bracket (uses its midpoint)", fontsize=8, color=INK2)
ax.set_xlim(-0.5, 29.5); ax.set_ylim(0, 1.35)
ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=8, color=INK2)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f °C")); ax.tick_params(axis="y", labelsize=8, colors=INK2)
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_xlabel("Hour of the day (HKT): forecast made just after HKO's reading for that hour", fontsize=9, color=INK2)
ax.set_ylabel("RMSE of the predicted daily max", fontsize=9, color=INK2)
nd = B[(B.date >= "2026-09-01") & B.xgb.notna()].date.nunique()
fig.suptitle("Error of the day's max-temperature forecast, hour by hour", x=0.06, ha="left", fontsize=13, color=INK, y=0.97)
ax.set_title(f"Test period 1 Sep – 4 Nov 2026 ({nd} days). Lower is better.", loc="left", fontsize=9, color=INK2)
leg = ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(0.78, 1.0))
for txt in leg.get_texts(): txt.set_color(INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "rmse_by_hour.png"), facecolor=SURF)

# ---------- Excel ----------
def wide(s, metric):
    w = s[s.hour != "all"].pivot(index="hour", columns="forecaster", values=metric)
    w.index = w.index.astype(int); w = w.sort_index()
    w.loc["All hours"] = s[s.hour == "all"].set_index("forecaster")[metric]
    return w.round(3)

order_t = [n for n, _ in SERIES] + ["Polymarket favourite bracket", "Observed running max"]
order_f = order_t[1:]
summ_t = test[test.hour == "all"].drop(columns="hour").set_index("forecaster").loc[order_t].round(3)
summ_f = full[full.hour == "all"].drop(columns="hour").set_index("forecaster").loc[order_f].round(3)
notes = pd.DataFrame({"Notes": [
    "What: every hour of the day, each forecaster predicts that day's HKO max temperature (the Polymarket settlement value).",
    "Timing: forecasts use only what is known at h:10 HKT, just after HKO publishes the h:00 reading (~h:05).",
    "XGBoost hourly (Ryan): floor(running max) + expected extra degrees + 0.5. Out-of-sample test days only (1 Sep - 4 Nov 2026).",
    "Polymarket expected value: 10-minute prices at h:10, normalised to sum to 1; bracket b counts as b+0.5, 'X or below' as X-0.5, 'X or higher' as X+1.",
    "Polymarket favourite bracket: the most expensive bracket, as b+0.5.",
    "HKO forecast only: max temperature stated in HKO's latest Local Weather Forecast bulletin issued by h:10 (archived hourly on data.gov.hk). "
    "HKO states today's max until ~11:00; after that the last value carries forward. Early hours use the previous evening's 'tomorrow' forecast.",
    "HKO forecast + observed max: max(HKO forecast, running max observed so far) - the HKO forecast updated every hour with observations.",
    "Observed running max: highest reading since midnight (a lower bound; shown for reference).",
    "SSE = sum of squared errors over all date-hours; RMSE = sqrt(mean squared error); MAE = mean absolute error; bias = mean(forecast - actual).",
    "bracket_hit = share of forecasts landing in the winning 1°C bracket (floor(forecast) == floor(actual)).",
    "Floor: a forecast that only names the bracket (market, XGBoost) can't beat RMSE ~0.29 even when certain, because it reports the bracket midpoint.",
    "Excluded dates (bad markets): 13-14 Mar and 18-19 May 2026. Full market period sheet has no XGBoost because earlier days were in its training data.",
]})
xl = os.path.join(OUT, "HK_Temp_Hourly_Benchmarks.xlsx")
with pd.ExcelWriter(xl, engine="openpyxl") as w:
    notes.to_excel(w, "Read me", index=False)
    summ_t.to_excel(w, "Summary test (Sep-Nov)")
    wide(test, "RMSE")[order_t].to_excel(w, "RMSE by hour (test)")
    wide(test, "SSE")[order_t].to_excel(w, "SSE by hour (test)")
    wide(test, "MAE")[order_t].to_excel(w, "MAE by hour (test)")
    summ_f.to_excel(w, "Summary full (Mar-Nov)")
    wide(full, "RMSE")[order_f].to_excel(w, "RMSE by hour (full)")
    wide(full, "SSE")[order_f].to_excel(w, "SSE by hour (full)")
    B.round({c: 3 for c in B.select_dtypes("number").columns}).to_excel(w, "Forecasts (date x hour)", index=False)
    pd.read_csv(os.path.join(OUT, "hko_forecast_bulletins.csv")).to_excel(w, "HKO bulletins", index=False)

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
wb = load_workbook(xl)
for ws in wb.worksheets:
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="2A5D9F")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.freeze_panes = "B2"
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 16
    ws.column_dimensions["A"].width = 14
wb["Read me"].column_dimensions["A"].width = 160
wb["Read me"].freeze_panes = None
for name in ["HKO bulletins"]:
    wb[name].column_dimensions["F"].width = 80
wb.save(xl)
print("wrote", xl)
