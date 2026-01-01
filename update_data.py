#!/usr/bin/env python3
import json
import re
import yfinance as yf
from pathlib import Path
from datetime import datetime

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

            # Prepare result structure matching the API output format
            result = {
                "symbol": symbol,
                "period": PERIOD,
                "data": data.reset_index().to_dict(orient="records"),
                "last_updated": datetime.utcnow().isoformat()
            }

            # Sanitize filename: replace non-alphanumeric with _
            # GC=F -> GC_F.json, USDIDR=X -> USDIDR_X.json
            safe_kode = re.sub(r'[^a-zA-Z0-9]', '_', symbol)
            file_path = DATA_DIR / f"{safe_kode}.json"

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"Successfully saved {symbol} to {file_path}")

        except Exception as e:
            print(f"Error updating {symbol}: {str(e)}")

    print(f"[{datetime.now()}] Data update job finished.")

if __name__ == "__main__":
    update_data()
