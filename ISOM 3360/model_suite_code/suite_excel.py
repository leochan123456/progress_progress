"""Build the verifiable model-suite workbook: every score is an Excel formula over the prediction sheets."""
import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L
import suite_common as C

D, H = os.path.join(C.OUT, "daily"), os.path.join(C.OUT, "hourly")
XL = os.path.join(C.OUT, "HK_Temp_Model_Suite.xlsx")
FONT = "Arial"
HEAD = PatternFill("solid", fgColor="1F4E79"); SUB = PatternFill("solid", fgColor="DDEBF7"); BEST = PatternFill("solid", fgColor="E2EFDA")
thin = Side(style="thin", color="BFBFBF")

wb = Workbook(); wb.remove(wb.active)


def sheet(name, df=None, widths=None, note=None, start_row=1, index=False):
    ws = wb.create_sheet(name)
    r0 = start_row
    if note:
        ws.cell(1, 1, note).font = Font(name=FONT, italic=True, color="595959", size=9)
        r0 = 3
    if df is not None:
        d = df.reset_index() if index else df
        for j, c in enumerate(d.columns, 1):
            cell = ws.cell(r0, j, str(c)); cell.font = Font(name=FONT, bold=True, color="FFFFFF"); cell.fill = HEAD
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for i, row in enumerate(d.itertuples(index=False), r0 + 1):
            for j, v in enumerate(row, 1):
                if isinstance(v, (pd.Timestamp,)): v = v.to_pydatetime()
                if isinstance(v, float) and np.isnan(v): v = None
                if isinstance(v, (np.integer,)): v = int(v)
                if isinstance(v, (np.floating,)): v = float(v)
                c = ws.cell(i, j, v); c.font = Font(name=FONT)
                if hasattr(v, "year"): c.number_format = "yyyy-mm-dd"
        ws.freeze_panes = ws.cell(r0 + 1, 2)
    for j in range(1, (ws.max_column or 1) + 1):
        ws.column_dimensions[L(j)].width = (widths or {}).get(j, 14)
    return ws, r0


def style_header(ws, r, c1, c2):
    for j in range(c1, c2 + 1):
        c = ws.cell(r, j); c.font = Font(name=FONT, bold=True, color="FFFFFF"); c.fill = HEAD
        c.alignment = Alignment(wrap_text=True, vertical="center")


def fmt(ws, rng, f):
    for row in ws[rng]:
        for c in row: c.number_format = f; c.font = Font(name=FONT)


# ======================================================================= data
dp = pd.read_csv(os.path.join(D, "daily_test_predictions.csv"), parse_dates=["date"])
dlb = pd.read_csv(os.path.join(D, "daily_leaderboard.csv"))
dtune = pd.read_csv(os.path.join(D, "daily_tuning_log.csv"))
dbt = pd.read_csv(os.path.join(D, "daily_backtest_summary.csv"))
dtr = pd.read_csv(os.path.join(D, "daily_backtest_trades.csv"), parse_dates=["date"])
dimp = pd.read_csv(os.path.join(D, "daily_permutation_importance.csv"), index_col=0)
dset = pd.read_csv(os.path.join(D, "daily_final_settings.csv"))
hp = pd.read_csv(os.path.join(H, "hourly_test_predicted_max.csv"), parse_dates=["date"])
hlb = pd.read_csv(os.path.join(H, "hourly_leaderboard.csv"))
htune = pd.read_csv(os.path.join(H, "hourly_tuning_log.csv"))
hbt = pd.read_csv(os.path.join(H, "hourly_backtest_summary.csv"))
htr = pd.read_csv(os.path.join(H, "hourly_backtest_first_signal_trades.csv"), parse_dates=["date"])
bench = pd.read_csv(os.path.join(C.PROJECT_ROOT, "hourly_benchmark_code", "out", "hourly_benchmark_forecasts.csv"),
                    parse_dates=["date"])

MODELS = ["rf", "gb", "knn", "xgb", "ridge"]
DCOLS = [("actual", "Actual max (HKO)")] + [(k, C.OWNER[k]) for k in MODELS] + [(f"{k}_default", C.OWNER[k] + " - default") for k in MODELS] + \
        [("persistence", C.OWNER["persistence"]), ("climatology", C.OWNER["climatology"])]

# ======================================================================= Read me
ws = wb.create_sheet("Read me")
lines = [
    ("HK max temperature - model suite (ISOM 3360)", True),
    ("Every member's model, rebuilt on one split and scored the same way. All scores on the leaderboard sheets are live Excel formulas over the prediction sheets, so you can check any number.", False),
    ("", False),
    ("PART A - Day-ahead models (predict day t's max from data up to day t-1)", True),
    ("Data: hko_training_set.csv (HKO daily climate data + Daily Extract), 2010-01-01 to 2026-11-23, 34 features.", False),
    ("Split by whole years: train 2010-2022 | validation 2023 (used to choose settings) | test 2024-01-01 to 2026-11-23 (model refit on train + validation).", False),
    ("Default = each member's original settings. Tuned = best validation RMSE over the grid in 'A3 Tuning log'.", False),
    ("Bracket log-loss: each model's forecast turned into 1°C bracket probabilities with a Normal curve (spread = its validation error SD). Lower is better.", False),
    ("A5 compares against HKO's official forecast and Polymarket at 00:10 on the days both exist.", False),
    ("", False),
    ("PART B - Hourly models (every hour, predict the day's final max from what HKO has measured so far)", True),
    ("Same dataset and target as the XGBoost hourly model: at hour h, predict d = floor(final max) - floor(running max), classes 0..8.", False),
    ("Split by whole days: train 1 Jul 2020 - 30 Jun 2026 | validation Jul-Aug 2026 | test 1 Sep - 4 Nov 2026 (58 days).", False),
    ("Predicted max = floor(running max) + expected extra degrees + 0.5. Polymarket and HKO are the hourly benchmarks (forecasts as of h:10).", False),
    ("Gradient Boosting uses sklearn's HistGradientBoostingClassifier (same algorithm, fast on 50k rows).", False),
    ("", False),
    ("Backtests (A6, B5): real 10-minute Polymarket prices, fee = 0.05 x price x (1 - price), trade when edge > 5c, each date x bracket x side counted once (first signal).", False),
    ("Daily models decide once at 00:10; hourly models every hour at h:10. 95% range = bootstrap over days. Prices are last-trade history, not executable bid/ask.", False),
    ("", False),
    ("How to verify: RMSE = SQRT(SUMXMY2(forecast, actual) / COUNT(actual)); SSE = SUMXMY2(forecast, actual); MAE = mean |forecast - actual|;", False),
    ("bias = mean(forecast - actual); bracket hit = share where INT(forecast) = INT(actual). Validation and log-loss columns (blue) come from Python and are labelled.", False),
    ("", False),
    ("PART C - One upgrade per member's model (day-ahead, same split as Part A): sheets C1-C8", True),
    ("Random Forest (Chris): walk-forward, refit every month on all data so far. Gradient Boosting (Sullivan): quantile regression -> full forecast distribution.", False),
    ("KNN (Ada): analog method, days compared by deviation from the date's average, Euclidean/Manhattan/cosine/correlation. XGBoost (Ryan): HKO's official forecast as an input.", False),
    ("", False),
    ("Code: suite_daily.py, suite_hourly.py, suite_upgrades.py, suite_common.py, suite_excel.py (own re-implementations; no member code was executed).", False),
]
for i, (t, b) in enumerate(lines, 1):
    c = ws.cell(i, 1, t); c.font = Font(name=FONT, bold=b, size=12 if i == 1 else 10)
ws.column_dimensions["A"].width = 150

# ======================================================================= A1 predictions
dcols = [c for c, _ in DCOLS]
a1 = dp[["date"] + dcols].rename(columns=dict(DCOLS))
ws, r0 = sheet("A1 Daily predictions", a1.round(3), note="Test days 2024-01-01 .. 2026-11-23. One row per day; values in °C.")
n1 = len(a1); first, last = r0 + 1, r0 + n1
fmt(ws, f"B{first}:{L(len(a1.columns))}{last}", "0.00")
colof = {c: L(j + 2) for j, c in enumerate(dcols)}

# ======================================================================= A2 leaderboard (formulas)
ws, _ = sheet("A2 Daily leaderboard", widths={1: 42, 2: 10})
ws.cell(1, 1, "Day-ahead models, test 2024-01-01 .. 2026-11-23. Black = Excel formulas over 'A1 Daily predictions'. Blue = from Python (validation / log-loss).").font = Font(name=FONT, italic=True, size=9)
hdr = ["Model", "Version", "n (days)", "SSE", "RMSE", "MAE", "Bias", "Bracket hit", "Val RMSE (2023)", "Train RMSE", "Bracket log-loss"]
for j, h in enumerate(hdr, 1): ws.cell(3, j, h)
style_header(ws, 3, 1, len(hdr))
A = f"'A1 Daily predictions'!${colof['actual']}${first}:${colof['actual']}${last}"
r = 4
for _, row in dlb.iterrows():
    k = [kk for kk in MODELS + ["persistence", "climatology"] if C.OWNER[kk] == row.model][0]
    col = f"{k}_default" if row.version == "default" else k
    P = f"'A1 Daily predictions'!${colof[col]}${first}:${colof[col]}${last}"
    vals = [row.model, row.version, f"=COUNT({A})", f"=SUMXMY2({P},{A})", f"=SQRT(D{r}/C{r})",
            f"=SUMPRODUCT(ABS({P}-{A}))/C{r}", f"=(SUM({P})-SUM({A}))/C{r}", f"=SUMPRODUCT(--(INT({P})=INT({A})))/C{r}",
            row.val_rmse, row.train_rmse, row.bracket_logloss]
    for j, v in enumerate(vals, 1):
        c = ws.cell(r, j, None if isinstance(v, float) and np.isnan(v) else v); c.font = Font(name=FONT, color="0000FF" if j >= 9 else "000000")
    r += 1
fmt(ws, f"C4:C{r - 1}", "0"); fmt(ws, f"D4:D{r - 1}", "#,##0.0"); fmt(ws, f"E4:G{r - 1}", "0.000"); fmt(ws, f"H4:H{r - 1}", "0.0%"); fmt(ws, f"I4:K{r - 1}", "0.000")
for rr in range(4, r):
    for j in (9, 10, 11): ws.cell(rr, j).font = Font(name=FONT, color="0000FF")
ws.cell(r + 1, 1, "Overfitting check: compare Train RMSE with Val RMSE. A big gap (Random Forest) = memorising; both high and close (Gradient Boosting default) = too simple.").font = Font(name=FONT, italic=True, size=9)
ws.cell(r + 2, 1, "Persistence/climatology Val RMSE is their own 2023 error; their bracket log-loss uses the 2023 error SD.").font = Font(name=FONT, italic=True, size=9)
ws.freeze_panes = "C4"

# ======================================================================= A3 tuning log
t = dtune.copy(); t["chosen"] = t.chosen.map({True: "CHOSEN", False: ""})
ws, r0 = sheet("A3 Daily tuning log", t.round(4), widths={1: 30, 2: 10, 3: 80}, note="Every setting tried, scored on the 2023 validation year. CHOSEN = lowest validation RMSE.")
for i in range(len(t)):
    if t.chosen.iloc[i] == "CHOSEN":
        for j in range(1, 7): ws.cell(r0 + 1 + i, j).fill = BEST

# ======================================================================= A4 KNN K curve
kk = dtune[(dtune.model == C.OWNER["knn"]) & (dtune.version == "tuning")].copy()
kk["k"] = kk.params.str.extract(r"'n_neighbors': (\d+)")[0].astype(int)
kk["weights"] = kk.params.str.extract(r"'weights': '(\w+)'")[0]; kk["p"] = kk.params.str.extract(r"'p': (\d)")[0].astype(int)
curve = kk.pivot_table(index="k", columns=["weights", "p"], values="val_rmse")
curve.columns = [f"{w}, {'Manhattan' if p == 1 else 'Euclidean'} distance" for w, p in curve.columns]
curve = curve.reset_index()
ws, r0 = sheet("A4 KNN K curve", curve.round(4), widths={1: 8, 2: 26, 3: 26, 4: 26, 5: 26},
               note="Ada's KNN: validation RMSE (2023) for each K. K = 1 copies the single most similar past day and is far worse; the curve bottoms out around K = 20-50.")
ch = LineChart(); ch.title = "KNN: validation RMSE by K"; ch.y_axis.title = "RMSE (°C)"; ch.x_axis.title = "K"
ch.add_data(Reference(ws, min_col=2, max_col=1 + len(curve.columns) - 1, min_row=r0, max_row=r0 + len(curve)), titles_from_data=True)
ch.set_categories(Reference(ws, min_col=1, min_row=r0 + 1, max_row=r0 + len(curve)))
ch.roundedCorners = False; ch.legend.overlay = False; ch.title.overlay = False; ch.x_axis.delete = False; ch.y_axis.delete = False
ch.height, ch.width = 9, 18; ws.add_chart(ch, "G3")

# ======================================================================= A5 vs HKO and market
cmp_cols = ["actual"] + MODELS + ["persistence", "climatology", "hko", "market"]
names = {**dict(DCOLS), "hko": C.OWNER["hko"] + " (latest at 00:10)", "market": C.OWNER["market"] + " expected value at 00:10"}
sub = dp.dropna(subset=["hko", "market"])[["date"] + cmp_cols].rename(columns=names)
ws, r0 = sheet("A5 Daily vs HKO & market", None, widths={1: 44})
ws.cell(1, 1, f"Same {len(sub)} days (Polymarket market days with an HKO forecast, {sub.date.min():%Y-%m-%d} .. {sub.date.max():%Y-%m-%d}). Formulas over the table below.").font = Font(name=FONT, italic=True, size=9)
for j, h in enumerate(["Forecaster", "n", "SSE", "RMSE", "MAE", "Bias", "Bracket hit"], 1): ws.cell(3, j, h)
style_header(ws, 3, 1, 7)
dstart = 3 + len(cmp_cols) + 3
for j, cname in enumerate(sub.columns, 1): ws.cell(dstart, j, cname)
style_header(ws, dstart, 1, len(sub.columns))
for i, row in enumerate(sub.itertuples(index=False), dstart + 1):
    for j, v in enumerate(row, 1):
        c = ws.cell(i, j, v.to_pydatetime() if hasattr(v, "to_pydatetime") else round(float(v), 3)); c.font = Font(name=FONT)
        c.number_format = "yyyy-mm-dd" if j == 1 else "0.00"
d1, d2 = dstart + 1, dstart + len(sub)
Aa = f"$B${d1}:$B${d2}"
for i, cname in enumerate(cmp_cols[1:], 4):
    col = L(cmp_cols.index(cname) + 2); P = f"${col}${d1}:${col}${d2}"
    for j, v in enumerate([names[cname], f"=COUNT({Aa})", f"=SUMXMY2({P},{Aa})", f"=SQRT(C{i}/B{i})", f"=SUMPRODUCT(ABS({P}-{Aa}))/B{i}",
                           f"=(SUM({P})-SUM({Aa}))/B{i}", f"=SUMPRODUCT(--(INT({P})=INT({Aa})))/B{i}"], 1):
        ws.cell(i, j, v).font = Font(name=FONT)
    fmt(ws, f"C{i}:C{i}", "#,##0.0"); fmt(ws, f"D{i}:F{i}", "0.000"); fmt(ws, f"G{i}:G{i}", "0.0%")
for j in range(2, len(sub.columns) + 1): ws.column_dimensions[L(j)].width = 16

# ======================================================================= A6 daily backtest
ws, r0 = sheet("A6 Daily backtest", dbt.round(4), widths={1: 32}, note="One decision per bracket at 00:10 on the day, real 10-minute prices. pnl per 1-share bet ($1 payout).")
tr0 = r0 + len(dbt) + 4
ws.cell(tr0 - 1, 1, "Trades (first signal per date x bracket x side). Check: total P&L per model =SUMIF over column K.").font = Font(name=FONT, italic=True, size=9)
td = dtr.copy()
for j, h in enumerate(td.columns, 1): ws.cell(tr0, j, h)
style_header(ws, tr0, 1, len(td.columns))
for i, row in enumerate(td.itertuples(index=False), tr0 + 1):
    for j, v in enumerate(row, 1):
        v = v.to_pydatetime() if hasattr(v, "to_pydatetime") else (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
        c = ws.cell(i, j, v); c.font = Font(name=FONT)
        if j == 1: c.number_format = "yyyy-mm-dd"
mcol = L(list(td.columns).index("model") + 1); pcol = L(list(td.columns).index("pnl") + 1)
chk = len(dbt.columns) + 2
ws.cell(r0, chk, "check: SUMIF trades").font = Font(name=FONT, bold=True)
for i in range(len(dbt)):
    ws.cell(r0 + 1 + i, chk, f"=SUMIF(${mcol}${tr0 + 1}:${mcol}${tr0 + len(td)},A{r0 + 1 + i},${pcol}${tr0 + 1}:${pcol}${tr0 + len(td)})").number_format = "0.000"

# ======================================================================= A7 importance + settings
imp = dimp.copy(); imp.index.name = "feature"
imp = imp.loc[imp.mean(axis=1).sort_values(ascending=False).index]
ws, r0 = sheet("A7 Daily importance", imp.round(4), index=True, widths={1: 22, 2: 22, 3: 26, 4: 18, 5: 18, 6: 24},
               note="Permutation importance on the 2023 validation year: how much RMSE rises (°C) when that feature is shuffled. Comparable across models.")
s0 = r0 + len(imp) + 3
for j, h in enumerate(dset.columns, 1): ws.cell(s0, j, h)
style_header(ws, s0, 1, len(dset.columns))
for i, row in enumerate(dset.itertuples(index=False), s0 + 1):
    for j, v in enumerate(row, 1): ws.cell(i, j, round(v, 4) if isinstance(v, float) else v).font = Font(name=FONT)

# ======================================================================= B1 hourly predictions
hb = hp.merge(bench[["date", "hour", "market_ev", "hko_fcst_obs", "hko_fcst"]], on=["date", "hour"], how="inner").dropna(subset=["market_ev", "hko_fcst"])
HCOLS = [("final_max", "Actual max (HKO)"), ("xgb", C.OWNER["xgb"]), ("rf", C.OWNER["rf"]), ("gb", C.OWNER["gb"]), ("knn", C.OWNER["knn"]),
         ("market_ev", "Polymarket expected value"), ("hko_fcst_obs", "HKO forecast + observed max"), ("hko_fcst", "HKO forecast only"),
         ("rf_default", C.OWNER["rf"] + " - default"), ("gb_default", C.OWNER["gb"] + " - default"), ("knn_default", C.OWNER["knn"] + " - default"),
         ("baseline", "Per-hour average (baseline)")]
b1 = hb[["date", "hour", "rmax"] + [c for c, _ in HCOLS]].rename(columns={"rmax": "Running max at h:00", **dict(HCOLS)}).sort_values(["date", "hour"])
ws, r0 = sheet("B1 Hourly predictions", b1.round(3), note="Test 1 Sep - 4 Nov 2026. One row per date x hour; forecasts of the day's final max made at h:10.")
hf, hl = r0 + 1, r0 + len(b1)
fmt(ws, f"C{hf}:{L(len(b1.columns))}{hl}", "0.00")
hcol = {c: L(j + 4) for j, (c, _) in enumerate(HCOLS)}
HR = f"'B1 Hourly predictions'!$B${hf}:$B${hl}"
HA = f"'B1 Hourly predictions'!${hcol['final_max']}${hf}:${hcol['final_max']}${hl}"

# ======================================================================= B2 hourly leaderboard
ws, _ = sheet("B2 Hourly leaderboard", widths={1: 40})
ws.cell(1, 1, "Hourly models + benchmarks, all hours pooled (formulas over 'B1 Hourly predictions'). Blue = from Python: log-loss of the class probabilities (lower is better).").font = Font(name=FONT, italic=True, size=9)
hdr = ["Forecaster", "n (date-hours)", "SSE", "RMSE", "MAE", "Bias", "Bracket hit", "Test log-loss", "Val log-loss"]
for j, h in enumerate(hdr, 1): ws.cell(3, j, h)
style_header(ws, 3, 1, len(hdr))
ll = hlb.set_index("model")
for i, (c, nm) in enumerate(HCOLS[1:], 4):
    P = f"'B1 Hourly predictions'!${hcol[c]}${hf}:${hcol[c]}${hl}"
    lrow = ll.loc[nm] if nm in ll.index else None
    vals = [nm, f"=COUNT({HA})", f"=SUMXMY2({P},{HA})", f"=SQRT(C{i}/B{i})", f"=SUMPRODUCT(ABS({P}-{HA}))/B{i}", f"=(SUM({P})-SUM({HA}))/B{i}",
            f"=SUMPRODUCT(--(INT({P})=INT({HA})))/B{i}", None if lrow is None else float(lrow.test_logloss),
            None if lrow is None or pd.isna(lrow.val_logloss) else float(lrow.val_logloss)]
    for j, v in enumerate(vals, 1): ws.cell(i, j, v).font = Font(name=FONT, color="0000FF" if j >= 8 else "000000")
    fmt(ws, f"C{i}:C{i}", "#,##0.0"); fmt(ws, f"D{i}:F{i}", "0.000"); fmt(ws, f"G{i}:G{i}", "0.0%"); fmt(ws, f"H{i}:I{i}", "0.000")
    for j in (8, 9): ws.cell(i, j).font = Font(name=FONT, color="0000FF")
ws.cell(4 + len(HCOLS), 1, "Bracket-only forecasts (models, Polymarket) report the bracket midpoint, so their RMSE can't go below ~0.29 even when certain.").font = Font(name=FONT, italic=True, size=9)

# ======================================================================= B3 RMSE & SSE by hour
ws, _ = sheet("B3 Error by hour", widths={1: 8})
ws.cell(1, 1, "RMSE and SSE of the predicted daily max at each hour (formulas over 'B1 Hourly predictions'). Lower is better.").font = Font(name=FONT, italic=True, size=9)
show = HCOLS[1:8]
ws.cell(3, 1, "RMSE"); ws.cell(3, 1).font = Font(name=FONT, bold=True)
for j, (c, nm) in enumerate(show, 2): ws.cell(4, j, nm)
ws.cell(4, 1, "Hour"); style_header(ws, 4, 1, len(show) + 1)
s_row = 4 + 24 + 3
ws.cell(s_row - 1, 1, "SSE").font = Font(name=FONT, bold=True)
for j, (c, nm) in enumerate(show, 2): ws.cell(s_row, j, nm)
ws.cell(s_row, 1, "Hour"); style_header(ws, s_row, 1, len(show) + 1)
for h in range(24):
    ws.cell(5 + h, 1, h); ws.cell(s_row + 1 + h, 1, h)
    for j, (c, nm) in enumerate(show, 2):
        P = f"'B1 Hourly predictions'!${hcol[c]}${hf}:${hcol[c]}${hl}"
        sse = f"SUMPRODUCT(({HR}=$A{{r}})*({P}-{HA})^2)"
        ws.cell(5 + h, j, "=SQRT(" + sse.format(r=5 + h) + f"/COUNTIF({HR},$A{5 + h}))").number_format = "0.000"
        ws.cell(s_row + 1 + h, j, "=" + sse.format(r=s_row + 1 + h)).number_format = "#,##0.0"
for j in range(2, len(show) + 2): ws.column_dimensions[L(j)].width = 18
ch = LineChart(); ch.title = "RMSE of the predicted daily max by hour"; ch.y_axis.title = "RMSE (°C)"; ch.x_axis.title = "Hour (HKT)"
ch.add_data(Reference(ws, min_col=2, max_col=len(show) + 1, min_row=4, max_row=28), titles_from_data=True)
ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=28))
ch.roundedCorners = False; ch.legend.overlay = False; ch.title.overlay = False; ch.x_axis.delete = False; ch.y_axis.delete = False
ch.height, ch.width = 11, 24; ws.add_chart(ch, "K3")

# ======================================================================= B4 tuning
t = htune.copy(); t["chosen"] = t.chosen.map({True: "CHOSEN", False: ""})
ws, r0 = sheet("B4 Hourly tuning log", t.round(4), widths={1: 30, 2: 10, 3: 90}, note="Settings tried for the hourly models, scored by log-loss on the Jul-Aug 2026 validation days. CHOSEN = lowest.")
for i in range(len(t)):
    if t.chosen.iloc[i] == "CHOSEN":
        for j in range(1, 6): ws.cell(r0 + 1 + i, j).fill = BEST

# ======================================================================= B5 hourly backtest
ws, r0 = sheet("B5 Hourly backtest", hbt.round(4), widths={1: 32},
               note="Hourly models trade at h:10 (just after HKO's reading). first_signal_pnl_1h_late = same signals priced an hour later.")
tr0 = r0 + len(hbt) + 4
ws.cell(tr0 - 1, 1, "First-signal trades (all models).").font = Font(name=FONT, italic=True, size=9)
for j, h in enumerate(htr.columns, 1): ws.cell(tr0, j, h)
style_header(ws, tr0, 1, len(htr.columns))
for i, row in enumerate(htr.itertuples(index=False), tr0 + 1):
    for j, v in enumerate(row, 1):
        v = v.to_pydatetime() if hasattr(v, "to_pydatetime") else (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
        c = ws.cell(i, j, v); c.font = Font(name=FONT)
        if j == 1: c.number_format = "yyyy-mm-dd"
mcol = L(list(htr.columns).index("model") + 1); pcol = L(list(htr.columns).index("pnl") + 1)
chk = len(hbt.columns) + 2
ws.cell(r0, chk, "check: SUMIF trades").font = Font(name=FONT, bold=True)
for i in range(len(hbt)):
    ws.cell(r0 + 1 + i, chk, f"=SUMIF(${mcol}${tr0 + 1}:${mcol}${tr0 + len(htr)},A{r0 + 1 + i},${pcol}${tr0 + 1}:${pcol}${tr0 + len(htr)})").number_format = "0.000"

# ======================================================================= PART C: upgrades
U = os.path.join(C.OUT, "upgrades")
if os.path.exists(os.path.join(U, "upgrade_leaderboard.csv")):
    R = pd.read_csv(os.path.join(U, "upgrade_test_predictions.csv"), parse_dates=["date"])
    ulb = pd.read_csv(os.path.join(U, "upgrade_leaderboard.csv"))
    UPS = [("rf", "rf_walkforward", "walk-forward monthly refit"), ("gb", "gb_quantile", "quantile regression (median shown)"),
           ("knn", "knn_analog", "analog: deviation from date average"), ("xgb", "xgb_hko", "+ HKO official forecast input")]
    ccols = [("actual", "Actual max (HKO)")]
    for k, u, lab in UPS:
        ccols += [(f"{k}_default", f"{C.OWNER[k]} - original"), (f"{k}_tuned", f"{C.OWNER[k]} - tuned"), (u, f"{C.OWNER[k]} - upgraded ({lab})")]
    ccols += [("hko_fcst", "HKO forecast (evening before)"), ("market", "Polymarket at 00:10")]
    c1 = R[["date"] + [c for c, _ in ccols]].rename(columns=dict(ccols))
    ws, r0 = sheet("C1 Upgrade predictions", c1.round(3), note="Day-ahead test 2024-01-01 .. 2026-11-23: each model's original, tuned and upgraded forecast. HKO/Polymarket blank where not available.")
    cf, cl = r0 + 1, r0 + len(c1)
    fmt(ws, f"B{cf}:{L(len(c1.columns))}{cl}", "0.00")
    ccol = {c: L(j + 2) for j, (c, _) in enumerate(ccols)}
    CA = f"'C1 Upgrade predictions'!${ccol['actual']}${cf}:${ccol['actual']}${cl}"

    ws, _ = sheet("C2 Upgrade leaderboard", widths={1: 30, 2: 44})
    ws.cell(1, 1, "Original vs tuned vs upgraded, test 2024-01-01 .. 2026-11-23. Black = Excel formulas over 'C1 Upgrade predictions'; blue = bracket log-loss from Python.").font = Font(name=FONT, italic=True, size=9)
    hdr = ["Model", "Version", "n", "SSE", "RMSE", "MAE", "Bias", "Bracket hit", "RMSE change vs original", "Bracket log-loss"]
    for j, h in enumerate(hdr, 1): ws.cell(3, j, h)
    style_header(ws, 3, 1, len(hdr))
    r = 4
    for k, u, lab in UPS:
        r_orig = r
        for ver, col in [("original (default)", f"{k}_default"), ("tuned", f"{k}_tuned"), (f"upgraded: {lab}", u)]:
            P = f"'C1 Upgrade predictions'!${ccol[col]}${cf}:${ccol[col]}${cl}"
            ll = ulb[(ulb.model == C.OWNER[k]) & (ulb.version.str.startswith(ver.split(":")[0]))].bracket_logloss
            vals = [C.OWNER[k], ver, f"=COUNT({CA})", f"=SUMXMY2({P},{CA})", f"=SQRT(D{r}/C{r})", f"=SUMPRODUCT(ABS({P}-{CA}))/C{r}",
                    f"=(SUM({P})-SUM({CA}))/C{r}", f"=SUMPRODUCT(--(INT({P})=INT({CA})))/C{r}", f"=E{r}-E${r_orig}", float(ll.iloc[0]) if len(ll) else None]
            for j, v in enumerate(vals, 1): ws.cell(r, j, v).font = Font(name=FONT, color="0000FF" if j == 10 else "000000")
            if ver.startswith("upgraded"):
                for j in range(1, 11): ws.cell(r, j).fill = BEST
            fmt(ws, f"D{r}:D{r}", "#,##0.0"); fmt(ws, f"E{r}:G{r}", "0.000"); fmt(ws, f"H{r}:H{r}", "0.0%"); fmt(ws, f"I{r}:J{r}", "0.000")
            ws.cell(r, 10).font = Font(name=FONT, color="0000FF")
            r += 1
        r += 1
    # benchmarks on all test days
    for nm, col in [("Persistence: yesterday's max", None)]:
        pass
    ws.cell(r + 1, 1, "Bracket log-loss: quantile GB uses its own 7-percentile distribution; the others a Normal curve with their validation error SD.").font = Font(name=FONT, italic=True, size=9)

    # same days as HKO + market
    sub = R.dropna(subset=["hko_fcst", "market"])
    scols = ["actual"] + [u for _, u, _ in UPS] + ["xgb_tuned", "hko_fcst", "market"]
    snames = {**dict(ccols)}
    ws, _ = sheet("C3 Upgrades vs HKO & market", widths={1: 52})
    ws.cell(1, 1, f"Same {len(sub)} days with an HKO forecast and a Polymarket price ({sub.date.min():%Y-%m-%d} .. {sub.date.max():%Y-%m-%d}). Formulas over the table below.").font = Font(name=FONT, italic=True, size=9)
    for j, h in enumerate(["Forecaster", "n", "SSE", "RMSE", "MAE", "Bias", "Bracket hit"], 1): ws.cell(3, j, h)
    style_header(ws, 3, 1, 7)
    dstart = 3 + len(scols) + 3
    tab = sub[["date"] + scols].rename(columns=snames)
    for j, h in enumerate(tab.columns, 1): ws.cell(dstart, j, h)
    style_header(ws, dstart, 1, len(tab.columns))
    for i, row in enumerate(tab.itertuples(index=False), dstart + 1):
        for j, v in enumerate(row, 1):
            c = ws.cell(i, j, v.to_pydatetime() if hasattr(v, "to_pydatetime") else round(float(v), 3)); c.font = Font(name=FONT)
            c.number_format = "yyyy-mm-dd" if j == 1 else "0.00"
    d1, d2 = dstart + 1, dstart + len(tab); Aa = f"$B${d1}:$B${d2}"
    for i, cname in enumerate(scols[1:], 4):
        col = L(scols.index(cname) + 2); P = f"${col}${d1}:${col}${d2}"
        for j, v in enumerate([snames[cname], f"=COUNT({Aa})", f"=SUMXMY2({P},{Aa})", f"=SQRT(C{i}/B{i})", f"=SUMPRODUCT(ABS({P}-{Aa}))/B{i}",
                               f"=(SUM({P})-SUM({Aa}))/B{i}", f"=SUMPRODUCT(--(INT({P})=INT({Aa})))/B{i}"], 1):
            ws.cell(i, j, v).font = Font(name=FONT)
        fmt(ws, f"C{i}:C{i}", "#,##0.0"); fmt(ws, f"D{i}:F{i}", "0.000"); fmt(ws, f"G{i}:G{i}", "0.0%")
    for j in range(2, len(tab.columns) + 1): ws.column_dimensions[L(j)].width = 16

    # XGB + HKO details
    cov = pd.read_csv(os.path.join(U, "xgb_hko_coverage.csv")); xl_ = pd.read_csv(os.path.join(U, "xgb_hko_tuning_log.csv"))
    gi = pd.read_csv(os.path.join(U, "xgb_hko_importance.csv")).head(15)
    ws, r0 = sheet("C4 XGB + HKO forecast", cov, widths={1: 36, 2: 14, 3: 18},
                   note="Ryan's upgrade: HKO's official forecast (latest bulletin before midnight) as an input. Coverage, top features (share of gain), tuning log.")
    g0 = r0 + len(cov) + 3
    for j, h in enumerate(["feature", "share of gain"], 1): ws.cell(g0, j, h)
    style_header(ws, g0, 1, 2)
    for i, row in enumerate(gi.itertuples(index=False), g0 + 1):
        ws.cell(i, 1, row[0]).font = Font(name=FONT); c = ws.cell(i, 2, float(row[1])); c.number_format = "0.0%"; c.font = Font(name=FONT)
    t0_ = g0 + len(gi) + 3
    for j, h in enumerate(xl_.columns, 1): ws.cell(t0_, j, h)
    style_header(ws, t0_, 1, len(xl_.columns))
    for i, row in enumerate(xl_.round(4).itertuples(index=False), t0_ + 1):
        for j, v in enumerate(row, 1):
            ws.cell(i, j, v).font = Font(name=FONT)
            if row[-1] == "CHOSEN": ws.cell(i, j).fill = BEST

    # GB quantile
    gq = pd.read_csv(os.path.join(U, "gb_quantile_vs_normal.csv")); rel = pd.read_csv(os.path.join(U, "gb_reliability.csv"))
    ws, r0 = sheet("C5 GB quantile & calibration", gq.round(4), widths={1: 52},
                   note="Sullivan's upgrade: predict 7 percentiles instead of one number. cover_80 should be 80% (share of days inside the 10th-90th percentile band).")
    q0 = r0 + len(gq) + 3
    piv = rel.pivot_table(index="bin", columns="model", values=["predicted", "actual_freq", "n"], sort=False)
    piv.columns = [f"{b} - {a}" for a, b in piv.columns]
    piv = piv.reset_index()
    ws.cell(q0 - 1, 1, "Reliability: every test day x 1°C bracket, grouped by predicted probability. Well calibrated = actual frequency close to predicted.").font = Font(name=FONT, italic=True, size=9)
    for j, h in enumerate(piv.columns, 1): ws.cell(q0, j, h)
    style_header(ws, q0, 1, len(piv.columns))
    for i, row in enumerate(piv.itertuples(index=False), q0 + 1):
        for j, v in enumerate(row, 1):
            c = ws.cell(i, j, v if isinstance(v, str) else float(v)); c.font = Font(name=FONT)
            if not isinstance(v, str): c.number_format = "0" if "- n" in piv.columns[j - 1] else "0.000"
    ch = LineChart(); ch.title = "Calibration: actual frequency by predicted probability"; ch.y_axis.title = "Actual frequency"; ch.x_axis.title = "Predicted probability bin"
    cols_af = [j for j, c in enumerate(piv.columns, 1) if "actual_freq" in c or "predicted" in c]
    for j in cols_af:
        ch.add_data(Reference(ws, min_col=j, min_row=q0, max_row=q0 + len(piv)), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=q0 + 1, max_row=q0 + len(piv)))
    ch.roundedCorners = False; ch.legend.overlay = False; ch.title.overlay = False; ch.x_axis.delete = False; ch.y_axis.delete = False
    ch.height, ch.width = 9, 18; ws.add_chart(ch, f"A{q0 + len(piv) + 3}")

    # KNN analog
    kt = pd.read_csv(os.path.join(U, "knn_similarity_comparison.csv")); kl = pd.read_csv(os.path.join(U, "knn_analog_tuning_log.csv"))
    ws, r0 = sheet("C6 KNN similarity", kt.round(4), widths={1: 40, 2: 14},
                   note="Ada's upgrade: best K for each feature space x similarity measure (validation 2023). Analog = compare days by deviation from the date's average.")
    k0 = r0 + len(kt) + 3
    for j, h in enumerate(kl.columns, 1): ws.cell(k0, j, h)
    style_header(ws, k0, 1, len(kl.columns))
    for i, row in enumerate(kl.round(4).itertuples(index=False), k0 + 1):
        for j, v in enumerate(row, 1):
            ws.cell(i, j, None if isinstance(v, float) and np.isnan(v) else v).font = Font(name=FONT)

    # RF walk-forward drift
    dr = pd.read_csv(os.path.join(U, "rf_walkforward_drift.csv"))
    ws, r0 = sheet("C7 RF walk-forward drift", dr.round(3), widths={1: 8},
                   note="Chris's upgrade: refit every month on all data so far. Bias = mean(forecast - actual); negative = forecasting too cold.")
    ch = LineChart(); ch.title = "Forecast bias by year (°C)"; ch.y_axis.title = "Bias"; ch.x_axis.title = "Year"
    for nm in ["rf_static_bias", "rf_walkforward_bias"]:
        j = list(dr.columns).index(nm) + 1; ch.add_data(Reference(ws, min_col=j, min_row=r0, max_row=r0 + len(dr)), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=r0 + 1, max_row=r0 + len(dr)))
    ch.roundedCorners = False; ch.legend.overlay = False; ch.title.overlay = False; ch.x_axis.delete = False; ch.y_axis.delete = False
    ch.height, ch.width = 8, 16; ws.add_chart(ch, f"A{r0 + len(dr) + 3}")

    ub = pd.read_csv(os.path.join(U, "upgrade_backtest_summary.csv"))
    sheet("C8 Upgrade backtest", ub.round(4), widths={1: 40}, note="Upgraded day-ahead models, one decision per bracket at 00:10 (same rules as A6).")

wb.save(XL)
print("wrote", XL)
