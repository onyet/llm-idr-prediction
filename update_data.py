#!/usr/bin/env python3
import json
import re
import yfinance as yf
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configuration
SYMBOLS = ["GC=F", "SAR=X", "USDIDR=X"]
PERIOD = "5y"
DATA_DIR = Path(__file__).resolve().parent / "data"

def update_data():
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    print(f"[{datetime.now()}] Starting data update job...")

    for symbol in SYMBOLS:
        try:
            print(f"Fetching data for {symbol} (period={PERIOD})...")
            ticker = yf.Ticker(symbol)
            
            # Fetch history
            data = ticker.history(period=PERIOD)
            
            if data.empty:
                print(f"Warning: No data found for {symbol}")
                continue

            # Prepare DataFrame of fetched data
            new_df = data.reset_index()
            # Ensure Date column is parsed as datetime
            if 'Date' in new_df.columns:
                new_df['Date'] = pd.to_datetime(new_df['Date'], errors='coerce', utc=True)
                try:
                    new_df['Date'] = new_df['Date'].dt.tz_convert(None)
                except Exception:
                    try:
                        new_df['Date'] = new_df['Date'].dt.tz_localize(None)
                    except Exception:
                        pass
            else:
                print(f"Warning: fetched data for {symbol} missing Date column, skipping")
                continue

            # Sanitize filename: replace non-alphanumeric with _
            # GC=F -> GC_F.json, USDIDR=X -> USDIDR_X.json
            safe_kode = re.sub(r'[^a-zA-Z0-9]', '_', symbol)
            file_path = DATA_DIR / f"{safe_kode}.json"

            # If file exists, merge new dates only
            if file_path.exists():
                with open(file_path, 'r') as f:
                    existing = json.load(f)
                existing_records = existing.get('data', [])
                if existing_records:
                    exist_df = pd.DataFrame(existing_records)
                    if 'Date' in exist_df.columns:
                        exist_df['Date'] = pd.to_datetime(exist_df['Date'], errors='coerce', utc=True)
                        try:
                            exist_df['Date'] = exist_df['Date'].dt.tz_convert(None)
                        except Exception:
                            try:
                                exist_df['Date'] = exist_df['Date'].dt.tz_localize(None)
                            except Exception:
                                pass
                    # compute date-only
                    exist_df = exist_df.dropna(subset=['Date']).reset_index(drop=True)
                    if not exist_df.empty:
                        last_exist_date = exist_df['Date'].dt.date.max()
                        # select new rows with date > last_exist_date
                        new_rows = new_df[new_df['Date'].dt.date > last_exist_date]
                        if not new_rows.empty:
                            # append only new rows
                            added = new_rows.to_dict(orient='records')
                            existing['data'].extend(added)
                            existing['last_updated'] = datetime.utcnow().isoformat()
                            with open(file_path, 'w') as f:
                                json.dump(existing, f, indent=2, default=str)
                            print(f"Updated {symbol} - added {len(added)} new rows to {file_path}")
                        else:
                            print(f"No new data for {symbol} (latest {last_exist_date}), skipping write")
                    else:
                        # existing file but no valid dates -> overwrite
                        result = {
                            "symbol": symbol,
                            "period": PERIOD,
                            "data": new_df.to_dict(orient='records'),
                            "last_updated": datetime.utcnow().isoformat()
                        }
                        with open(file_path, 'w') as f:
                            json.dump(result, f, indent=2, default=str)
                        print(t('no_valid_date_entries') + f": rewrote {symbol} data to {file_path}")
                else:
                    # existing file empty data -> overwrite
                    result = {
                        "symbol": symbol,
                        "period": PERIOD,
                        "data": new_df.to_dict(orient='records'),
                        "last_updated": datetime.utcnow().isoformat()
                    }
                    with open(file_path, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    print(f"Saved {symbol} to {file_path} (was empty)")
            else:
                # Save new file
                result = {
                    "symbol": symbol,
                    "period": PERIOD,
                    "data": new_df.to_dict(orient='records'),
                    "last_updated": datetime.utcnow().isoformat()
                }
                with open(file_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Successfully saved {symbol} to {file_path}")

        except Exception as e:
            print(f"Error updating {symbol}: {str(e)}")

    print(f"[{datetime.now()}] Data update job finished.")

if __name__ == "__main__":
    update_data()
