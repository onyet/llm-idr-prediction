import json
import re
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Request
import yfinance as yf

from .i18n import get_lang_from_request, t

router = APIRouter(prefix="/tradingview", tags=["tradingview"])

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)  # pastikan direktori data ada

# cache sederhana untuk data
_cache: Dict[str, Any] = {}


def _get_symbol_unit(symbol: str) -> Dict[str, str]:
    """
    Tentukan unit dan deskripsi untuk sebuah simbol.
    Mengembalikan dict dengan kunci 'unit', 'description', dan 'type'.
    """
    symbol_upper = symbol.upper()
    
    # Forex pairs (XXX/YYY format or XXX=X, XXXYYY=X)
    if "=" in symbol_upper or "USD" in symbol_upper or "IDR" in symbol_upper or "SAR" in symbol_upper:
        if "USDIDR" in symbol_upper:
            return {
                "type": "forex",
                "unit": "IDR/USD",
                "description": "Indonesian Rupiah per 1 US Dollar",
                "note": "Nilai menunjukkan berapa Rupiah yang dibutuhkan untuk membeli 1 Dollar AS"
            }
        elif "USDSAR" in symbol_upper or "SAR" in symbol_upper and "USD" in symbol_upper:
            return {
                "type": "forex",
                "unit": "SAR/USD",
                "description": "Saudi Riyal per 1 US Dollar",
                "note": "Nilai menunjukkan berapa Riyal yang dibutuhkan untuk membeli 1 Dollar AS"
            }
        elif "EUR" in symbol_upper and "USD" in symbol_upper:
            return {
                "type": "forex",
                "unit": "USD/EUR",
                "description": "US Dollar per 1 Euro",
                "note": "Nilai menunjukkan berapa Dollar AS yang dibutuhkan untuk membeli 1 Euro"
            }
        else:
            # Generic forex
            return {
                "type": "forex",
                "unit": "Currency Rate",
                "description": f"Exchange rate for {symbol}",
                "note": "Nilai menunjukkan nilai tukar mata uang"
            }

    # Gold futures
    elif "GC" in symbol_upper or "GOLD" in symbol_upper:
        return {
            "type": "commodity",
            "unit": "USD/troy ounce",
            "description": "Gold price in US Dollars per troy ounce",
            "note": "Nilai menunjukkan harga emas per troy ounce (1 troy oz = 31.1035 gram)"
        }
    
    # Silver futures
    elif "SI" in symbol_upper and "_F" in symbol_upper:
        return {
            "type": "commodity",
            "unit": "USD/troy ounce",
            "description": "Silver price in US Dollars per troy ounce",
            "note": "Nilai menunjukkan harga perak per troy ounce"
        }
    
    # Oil futures
    elif "CL" in symbol_upper or "CRUDE" in symbol_upper or "WTI" in symbol_upper:
        return {
            "type": "commodity",
            "unit": "USD/barrel",
            "description": "Crude Oil price in US Dollars per barrel",
            "note": "Nilai menunjukkan harga minyak mentah per barrel"
        }
    
    # Indonesian stocks (.JK suffix)
    elif ".JK" in symbol_upper:
        return {
            "type": "stock",
            "unit": "IDR/share",
            "description": f"Stock price of {symbol} in Indonesian Rupiah",
            "note": "Nilai menunjukkan harga saham dalam Rupiah per lembar"
        }
    
    # US stocks (no special suffix, or known exchanges)
    elif any(suffix in symbol_upper for suffix in [".US", "^"]) or (len(symbol_upper) <= 5 and symbol_upper.isalpha()):
        return {
            "type": "stock",
            "unit": "USD/share",
            "description": f"Stock price of {symbol} in US Dollars",
            "note": "Nilai menunjukkan harga saham dalam Dollar AS per lembar"
        }
    
    # Other exchanges
    elif ".HK" in symbol_upper:
        return {"type": "stock", "unit": "HKD/share", "description": f"Stock price in Hong Kong Dollars", "note": "Harga saham dalam Dollar Hong Kong"}
    elif ".SI" in symbol_upper:
        return {"type": "stock", "unit": "SGD/share", "description": f"Stock price in Singapore Dollars", "note": "Harga saham dalam Dollar Singapura"}
    elif ".L" in symbol_upper:
        return {"type": "stock", "unit": "GBP/share", "description": f"Stock price in British Pounds", "note": "Harga saham dalam Pound Sterling"}
    elif ".TO" in symbol_upper:
        return {"type": "stock", "unit": "CAD/share", "description": f"Stock price in Canadian Dollars", "note": "Harga saham dalam Dollar Kanada"}
    elif ".AX" in symbol_upper:
        return {"type": "stock", "unit": "AUD/share", "description": f"Stock price in Australian Dollars", "note": "Harga saham dalam Dollar Australia"}
    
    # Default/unknown
    else:
        return {
            "type": "unknown",
            "unit": "Price",
            "description": f"Price data for {symbol}",
            "note": "Unit satuan tidak terdeteksi secara otomatis"
        }



@router.get("/get/{kode}")
async def get_tradingview_data(request: Request, kode: str, range: str = "1y", save: int = 0):
    """
    Ambil data historis untuk sebuah simbol menggunakan yfinance (mirip TradingView).
    - kode: simbol seperti USDIDR (akan dipetakan sesuai yfinance bila perlu)
    - range: '1y' atau '5y' (default 1y)
    - save: 1 untuk menyimpan data ke data/{kode}.json, 0 tidak menyimpan (default 0)
    """
    lang = get_lang_from_request(request)
    
    if range not in ["1y", "5y"]:
        raise HTTPException(status_code=400, detail=t("range_must_be", lang))
    if save not in [0, 1]:
        raise HTTPException(status_code=400, detail=t("save_must_be", lang))

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
                # Cek apakah simbol ada
                try:
                    info = ticker.info
                    if not info or 'symbol' not in info:
                        raise HTTPException(status_code=404, detail=t("symbol_not_found", lang, symbol=symbol))
                except:
                    pass  # properti info kadang gagal, lanjutkan
                raise HTTPException(status_code=404, detail=t("no_historical_data", lang, symbol=symbol, period=period))
            # Konversi ke dict untuk respons JSON dan sertakan informasi unit
            unit_info = _get_symbol_unit(symbol)
            result = {
                "symbol": symbol,
                "period": period,
                "unit": unit_info["unit"],
                "type": unit_info["type"],
                "description": unit_info["description"],
                "note": unit_info["note"],
                "data": data.reset_index().to_dict(orient="records")
            }
            _cache[cache_key] = result
        except HTTPException:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "no data" in error_str or "not found" in error_str or "invalid" in error_str:
                raise HTTPException(status_code=404, detail=t("symbol_invalid", lang, symbol=symbol))
            else:
                raise HTTPException(status_code=500, detail=t("error_fetching_data", lang, symbol=symbol, error=str(e)))

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
            raise HTTPException(status_code=500, detail=t("failed_to_save", lang, error=str(e)))

    return result


@router.get("/mock/{kode}")
async def get_mock_tradingview_data(request: Request, kode: str, period: str = "1y", save: int = 0):
    """
    Hasilkan data historis mock untuk pengujian (mensimulasikan format TradingView).
    - kode: simbol seperti XAUIDRG
    - period: '1y' atau '5y' (default 1y)
    - save: 1 untuk menyimpan data ke data/{kode}_mock.json, 0 tidak menyimpan (default 0)
    """
    lang = get_lang_from_request(request)
    
    if period not in ["1y", "5y"]:
        raise HTTPException(status_code=400, detail=t("period_must_be", lang))
    if save not in [0, 1]:
        raise HTTPException(status_code=400, detail=t("save_must_be", lang))

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

    unit_info = _get_symbol_unit(kode)
    result = {
        "symbol": kode,
        "period": period,
        "source": "mock_tradingview",
        "unit": unit_info["unit"],
        "type": unit_info["type"],
        "description": unit_info["description"],
        "note": f"{t('mock_data_note', lang)} | {unit_info['note']}",
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
            raise HTTPException(status_code=500, detail=t("failed_to_save", lang, error=str(e)))

    return result
