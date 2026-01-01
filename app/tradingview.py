import json
import re
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
import yfinance as yf

router = APIRouter(prefix="/tradingview", tags=["tradingview"])

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)  # ensure data dir exists

# simple cache for data
_cache: Dict[str, Any] = {}


@router.get("/get/{kode}")
async def get_tradingview_data(kode: str, range: str = "1y", save: int = 0):
    """
    Get historical data for a symbol using yfinance (similar to TradingView).
    - kode: symbol like USDIDR (will be converted to USDIDR=X for forex)
    - range: '1y' or '5y' (default 1y)
    - save: 1 to save data to data/{kode}.json, 0 otherwise (default 0)
    """
    if range not in ["1y", "5y"]:
        raise HTTPException(status_code=400, detail="range must be '1y' or '5y'")
    if save not in [0, 1]:
        raise HTTPException(status_code=400, detail="save must be 0 or 1")

    symbol = kode

    cache_key = f"{symbol}_{range}"
    if cache_key in _cache:
        result = _cache[cache_key]
    else:
        try:
            period = "5y" if range == "5y" else "1y"
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            if data.empty:
                # Check if symbol exists
                try:
                    info = ticker.info
                    if not info or 'symbol' not in info:
                        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found. Try a valid forex pair like 'USDIDR=X'.")
                except:
                    pass  # info might fail, continue
                raise HTTPException(status_code=404, detail=f"No historical data available for '{symbol}' in the last {period}.")
            # Convert to dict for JSON response
            result = {
                "symbol": symbol,
                "period": period,
                "data": data.reset_index().to_dict(orient="records")
            }
            _cache[cache_key] = result
        except HTTPException:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "no data" in error_str or "not found" in error_str or "invalid" in error_str:
                raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found or invalid. Check the symbol (e.g., 'USDIDR=X' for USD/IDR).")
            else:
                raise HTTPException(status_code=500, detail=f"Error fetching data for '{symbol}': {str(e)}")

    # If save=1, save to file
    if save == 1:
        # Sanitize filename: replace non-alphanumeric with _
        safe_kode = re.sub(r'[^a-zA-Z0-9]', '_', kode)
        file_path = DATA_DIR / f"{safe_kode}.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)  # default=str for datetime
            result["saved_to"] = str(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")

    return result


@router.get("/mock/{kode}")
async def get_mock_tradingview_data(kode: str, period: str = "1y", save: int = 0):
    """
    Generate mock historical data for testing (simulates TradingView data format).
    - kode: symbol like XAUIDRG
    - period: '1y' or '5y' (default 1y)
    - save: 1 to save data to data/{kode}_mock.json, 0 otherwise (default 0)
    """
    if period not in ["1y", "5y"]:
        raise HTTPException(status_code=400, detail="period must be '1y' or '5y'")
    if save not in [0, 1]:
        raise HTTPException(status_code=400, detail="save must be 0 or 1")

    import pandas as pd
    from datetime import datetime, timedelta
    import random

    end_date = datetime.utcnow()
    if period == "5y":
        start_date = end_date - timedelta(days=365*5)
        periods_count = 365*5
    else:
        start_date = end_date - timedelta(days=365)
        periods_count = 365

    dates = pd.date_range(start=start_date, end=end_date, freq='B')[:periods_count//2]

    base_price = random.uniform(100, 1000)
    prices = []
    current_price = base_price

    for i in range(len(dates)):
        change = random.gauss(0, 0.02)
        current_price *= (1 + change)
        current_price = max(current_price, base_price * 0.5)

        volatility = abs(change) * current_price * 0.5
        open_price = current_price
        high_price = current_price + random.uniform(0, volatility)
        low_price = current_price - random.uniform(0, volatility)
        close_price = current_price + random.gauss(0, volatility * 0.1)
        volume = random.randint(1000, 100000)

        prices.append({
            'Date': dates[i].isoformat(),
            'Open': round(open_price, 2),
            'High': round(high_price, 2),
            'Low': round(low_price, 2),
            'Close': round(close_price, 2),
            'Volume': volume
        })

    df = pd.DataFrame(prices)
    df.set_index('Date', inplace=True)

    result = {
        "symbol": kode,
        "period": period,
        "source": "mock_tradingview",
        "note": "This is mock data for testing purposes. Real TradingView data requires premium API access.",
        "data": df.reset_index().to_dict(orient="records")
    }

    if save == 1:
        safe_kode = re.sub(r'[^a-zA-Z0-9]', '_', kode)
        file_path = DATA_DIR / f"{safe_kode}_mock.json"
        try:
            with open(file_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            result["saved_to"] = str(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")

    return result
