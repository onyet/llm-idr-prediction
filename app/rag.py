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
    if pair in _cache["data"]:
        return _cache["data"][pair]
    file_map = {"idr-sar": "idr-sar.json", "idr-usd": "idr-usd.json"}
    if pair not in file_map:
        raise FileNotFoundError("Unknown pair")
    p = DATA_DIR / file_map[pair]
    if not p.exists():
        raise FileNotFoundError(p)
    raw = json.loads(p.read_text())
    # convert to DataFrame with ds,y
    df = pd.DataFrame(raw)
    df["ds"] = pd.to_datetime(df["time"], unit="ms")
    df = df.sort_values("ds").rename(columns={"value": "y"})[["ds", "y"]]
    _cache["data"][pair] = df
    return df


def get_model(pair: str) -> Prophet:
    if pair in _cache["models"]:
        return _cache["models"][pair]
    df = load_data(pair)
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
