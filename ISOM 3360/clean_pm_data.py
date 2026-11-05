"""Clean Polymarket and HKO settlement CSV exports."""
import argparse
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PM_STORAGE = os.path.join(SCRIPT_DIR, "Storage", "PM")
PM_CLEANED_STORAGE = os.path.join(PM_STORAGE, "cleaned")


def clean_markets(path):
    markets = pd.read_csv(path)
    markets["obs_date"] = pd.to_datetime(markets["obs_date"], errors="coerce").dt.date
    for column in [
        "bracket_low", "bracket_high", "volume_usd", "final_yes_price",
        "resolved_yes", "fee_rate", "tick_size", "min_order", "hko_abs_max_c",
    ]:
        markets[column] = pd.to_numeric(markets[column], errors="coerce")
    for column in ["market_open_utc", "market_close_utc"]:
        markets[column] = pd.to_datetime(markets[column], errors="coerce", utc=True)
    for column in ["closed", "fees_enabled", "fee_taker_only"]:
        markets[column] = markets[column].astype("boolean")

    markets["is_archived_event"] = markets["event_slug"].str.startswith("arch-")
    markets = markets.sort_values(
        ["obs_date", "bracket", "is_archived_event", "market_open_utc", "volume_usd"],
        ascending=[True, True, True, False, False],
    )
    markets = markets.drop_duplicates(["obs_date", "bracket"], keep="first")
    markets = markets.drop(columns="is_archived_event")
    markets = markets.sort_values(["obs_date", "bracket"]).reset_index(drop=True)
    return markets


def clean_price_history(path):
    prices = pd.read_csv(path)
    prices["obs_date"] = pd.to_datetime(prices["obs_date"], errors="coerce").dt.date
    prices["yes_price"] = pd.to_numeric(prices["yes_price"], errors="coerce")
    prices["time_utc"] = pd.to_datetime(prices["time_utc"], errors="coerce", utc=True)
    prices["time_hkt"] = pd.to_datetime(prices["time_hkt"], errors="coerce")
    prices["hours_to_day_end"] = pd.to_numeric(prices["hours_to_day_end"], errors="coerce")
    prices = prices[prices["yes_price"].between(0, 1)]
    prices = prices.dropna(subset=["obs_date", "bracket", "time_utc"])
    prices = prices.drop_duplicates(["obs_date", "bracket", "time_utc"])
    return prices.sort_values(["obs_date", "bracket", "time_utc"]).reset_index(drop=True)


def clean_daily_extract(path):
    extract = pd.read_csv(path)
    extract["date"] = pd.to_datetime(extract["date"], errors="coerce").dt.date
    numeric_columns = [column for column in extract.columns if column != "date"]
    extract[numeric_columns] = extract[numeric_columns].apply(pd.to_numeric, errors="coerce")
    extract = extract.dropna(subset=["date"])
    extract = extract.drop_duplicates("date", keep="last")
    return extract.sort_values("date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=PM_STORAGE)
    parser.add_argument("--output-dir", default=PM_CLEANED_STORAGE)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    outputs = {
        "pm_markets_cleaned.csv": clean_markets(os.path.join(args.input_dir, "pm_markets.csv")),
        "pm_price_history_hourly_cleaned.csv": clean_price_history(
            os.path.join(args.input_dir, "pm_price_history_hourly.csv")
        ),
        "hko_daily_extract_cleaned.csv": clean_daily_extract(
            os.path.join(args.input_dir, "hko_daily_extract.csv")
        ),
    }
    for filename, frame in outputs.items():
        output_path = os.path.join(args.output_dir, filename)
        frame.to_csv(output_path, index=False)
        print(f"saved: {output_path} ({frame.shape[0]} rows, {frame.shape[1]} columns)")


if __name__ == "__main__":
    main()
