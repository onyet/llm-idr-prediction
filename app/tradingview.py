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

    # For forex, add =X suffix if not present
    if not kode.endswith('=X'):
        symbol = f"{kode}=X"
    else:
        symbol = kode

    cache_key = f"{symbol}_{range}"
    if cache_key in _cache:
        result = _cache[cache_key]
    else:
        try:
            period = "5y" if range == "5y" else "1y"
            data = yf.Ticker(symbol).history(period=period)
            if data.empty:
                raise HTTPException(status_code=404, detail="No data found for symbol")
            # Convert to dict for JSON response
            result = {
                "symbol": symbol,
                "period": period,
                "data": data.reset_index().to_dict(orient="records")
            }
            _cache[cache_key] = result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")

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