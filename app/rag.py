import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd
from prophet import Prophet
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/rag", tags=["rag"])

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# simple in-memory cache for loaded data and models
_cache: Dict[str, Any] = {"models": {}, "data": {}}


def load_data(pair: str) -> pd.DataFrame:
    """
    Load time series for a logical pair.
    - 'idr-usd' reads USDIDR_X.json (IDR per USD) and uses the Close price.
    - 'idr-sar' is derived as (USDIDR_Close / SAR_Close) -> IDR per SAR.
    Raises HTTPException(status_code=503) with an Indonesian message if source files are missing or malformed.
    """
    if pair in _cache["data"]:
        return _cache["data"][pair]

    # Helper to load yfinance-like JSON file and return DataFrame with ds and close
    def _load_yf_file(path: Path, name: str) -> pd.DataFrame:
        if not path.exists():
            raise HTTPException(status_code=503, detail=f"Model data belum disiapkan: {name}")
        raw = json.loads(path.read_text())
        data = raw.get("data") or raw.get("prices") or raw
        df = pd.DataFrame(data)
        if df.empty:
            raise HTTPException(status_code=503, detail=f"Model data belum disiapkan: {name} (empty)")
        # Normalize column names to lowercase for robustness
        df.columns = [c.lower() for c in df.columns]
        # Date parsing
        if "date" in df.columns:
            df["ds"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
        else:
            # try to infer index or other date-like columns
            possible = [c for c in df.columns if "time" in c or "date" in c or "ds" in c]
            if possible:
                df["ds"] = pd.to_datetime(df[possible[0]], utc=True).dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
            else:
                # As a last resort, try the index
                try:
                    df["ds"] = pd.to_datetime(df.index, utc=True).tz_convert('UTC').tz_localize(None).dt.normalize()
                except Exception:
                    raise HTTPException(status_code=503, detail=f"Model data belum disiapkan: {name} (no date column)")
        # Close price
        if "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
        else:
            raise HTTPException(status_code=503, detail=f"Model data belum disiapkan: {name} (no close column)")
        df = df[["ds", "close"]].dropna().sort_values("ds")
        return df

    if pair == "idr-usd":
        p = DATA_DIR / "USDIDR_X.json"
        df = _load_yf_file(p, "USDIDR")
        # use 1/Close as y (USD per IDR)
        df["y"] = 1 / df["close"]
        out = df[["ds", "y"]].dropna()
        _cache["data"][pair] = out
        return out

    if pair == "idr-sar":
        p_usd = DATA_DIR / "USDIDR_X.json"
        p_sar = DATA_DIR / "SAR_X.json"
        df_usd = _load_yf_file(p_usd, "USDIDR")
        df_sar = _load_yf_file(p_sar, "SAR")
        # merge on date (inner join) to ensure aligned observations
        merged = pd.merge(df_usd, df_sar, on="ds", how="inner", suffixes=("_usd", "_sar"))
        if merged.empty:
            raise HTTPException(status_code=503, detail="Model data belum disiapkan: idr-sar (no overlapping dates)")
        # avoid division by zero
        merged = merged[(merged["close_sar"] != 0) & (merged["close_usd"] != 0)]
        merged["y"] = merged["close_sar"] / merged["close_usd"]  # SAR per IDR
        out = merged[["ds", "y"]].dropna().sort_values("ds")
        if out.empty:
            raise HTTPException(status_code=503, detail="Model data belum disiapkan: idr-sar (no valid values)")
        _cache["data"][pair] = out
        return out

    if pair == "idr-gold-gram":
        p_usd = DATA_DIR / "USDIDR_X.json"
        p_gold = DATA_DIR / "GC_F.json"
        df_usd = _load_yf_file(p_usd, "USDIDR")
        df_gold = _load_yf_file(p_gold, "GC=F")
        
        # merge on date (inner join)
        merged = pd.merge(df_usd, df_gold, on="ds", how="inner", suffixes=("_usd", "_gold"))
        if merged.empty:
            raise HTTPException(status_code=503, detail="Model data belum disiapkan: idr-gold-gram (no overlapping dates)")
            
        # 1 Troy Ounce = 31.1034768 grams
        # Gold price in USD/gram = Gold price in USD/oz t / 31.1034768
        # Gold price in IDR/gram = (Gold price in USD/gram) * (IDR/USD)
        
        troy_oz_to_gram = 31.1034768
        merged["y"] = (merged["close_gold"] / troy_oz_to_gram) * merged["close_usd"]
        
        out = merged[["ds", "y"]].dropna().sort_values("ds")
        if out.empty:
            raise HTTPException(status_code=503, detail="Model data belum disiapkan: idr-gold-gram (no valid values)")
        _cache["data"][pair] = out
        return out

    raise FileNotFoundError("Unknown pair")


def get_model(pair: str) -> Prophet:
    if pair in _cache["models"]:
        return _cache["models"][pair]
    try:
        df = load_data(pair)
    except HTTPException:
        # propagate the HTTPException (e.g., 503 model not ready)
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model data belum disiapkan")
    # sanity check
    if df.empty:
        raise HTTPException(status_code=503, detail="Model data belum disiapkan (empty)")
    # train Prophet
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    _cache["models"][pair] = model
    return model


@router.get("/idr-sar")
async def predict_idr_sar(days: int = 0, amount: float = 1.0):
    return await _predict("idr-sar", days, amount)


@router.get("/idr-usd")
async def predict_idr_usd(days: int = 0, amount: float = 1.0):
    return await _predict("idr-usd", days, amount)


@router.get("/idr-gold-gram")
async def predict_idr_gold_gram(days: int = 0, amount_gram: float = 1.0):
    return await _predict("idr-gold-gram", days, amount_gram)


async def _predict(pair: str, days: int = 0, amount: float = 1.0):
    # validation
    if days < 0:
        raise HTTPException(status_code=400, detail="days must be >= 0")
    if days > 7:
        raise HTTPException(status_code=400, detail="requested forecast horizon too long (max 7 days)")
    if amount <= 0:
        raise HTTPException(status_code=400, detail="amount must be > 0")

    df = load_data(pair)
    model = get_model(pair)

    # forecast up to requested day (0=today)
    last_date = df["ds"].max().normalize()
    target_date = datetime.utcnow().date() + timedelta(days=days)
    horizon_days = (target_date - last_date.date()).days
    if horizon_days < 0:
        # if target within historical data, just return observed
        observed = df[df["ds"].dt.date == target_date]
        if observed.empty:
            raise HTTPException(status_code=404, detail="no data for requested date in historical range")
        val = float(observed.iloc[-1]["y"])
        return {
            "pair": pair,
            "method": "prophet",
            "date": target_date.isoformat(),
            "predicted": val,
            "amount": amount,
            "predicted_for_amount": val * amount,
            "note": "returned historical observed value",
        }

    # build dates to forecast from last_date +1 .. target_date
    future = model.make_future_dataframe(periods=horizon_days + 1, freq="D")
    forecast = model.predict(future)
    # select the target row
    target_row = forecast[forecast["ds"].dt.date == target_date]
    if target_row.empty:
        raise HTTPException(status_code=500, detail="prediction not available")
    yhat = float(target_row.iloc[0]["yhat"])
    yhat_lower = float(target_row.iloc[0]["yhat_lower"])
    yhat_upper = float(target_row.iloc[0]["yhat_upper"])

    return {
        "pair": pair,
        "method": "prophet",
        "date": target_date.isoformat(),
        "amount": amount,
        "predicted": yhat,
        "predicted_lower": yhat_lower,
        "predicted_upper": yhat_upper,
        "predicted_for_amount": yhat * amount,
        "predicted_for_amount_lower": yhat_lower * amount,
        "predicted_for_amount_upper": yhat_upper * amount,
        "trained_until": str(last_date.date()),
    }
