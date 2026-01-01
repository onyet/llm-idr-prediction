import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

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


def _get_trend_analysis(pair: str, days: int = 7) -> Dict[str, Any]:
    """
    Analyze trend for a pair over the specified number of days.
    Returns trend direction, percentage change, and volatility.
    """
    df = load_data(pair)
    model = get_model(pair)
    
    last_date = df["ds"].max().normalize()
    
    # Get recent historical data (last 30 days for trend analysis)
    recent_df = df[df["ds"] >= (last_date - timedelta(days=30))]
    
    if len(recent_df) < 5:
        recent_df = df.tail(30)
    
    # Current value (last observed)
    current_value = float(recent_df.iloc[-1]["y"])
    
    # Value from 7 days ago (or earliest in recent)
    if len(recent_df) >= 7:
        past_value = float(recent_df.iloc[-7]["y"])
    else:
        past_value = float(recent_df.iloc[0]["y"])
    
    # Calculate percentage change
    pct_change_7d = ((current_value - past_value) / past_value) * 100
    
    # Calculate 30-day moving average
    ma_30 = float(recent_df["y"].mean())
    
    # Volatility (standard deviation)
    volatility = float(recent_df["y"].std() / recent_df["y"].mean() * 100)
    
    # Forecast for tomorrow
    target_date = datetime.utcnow().date() + timedelta(days=1)
    horizon_days = (target_date - last_date.date()).days
    
    if horizon_days > 0:
        future = model.make_future_dataframe(periods=horizon_days + 1, freq="D")
        forecast = model.predict(future)
        target_row = forecast[forecast["ds"].dt.date == target_date]
        if not target_row.empty:
            predicted_tomorrow = float(target_row.iloc[0]["yhat"])
        else:
            predicted_tomorrow = current_value
    else:
        predicted_tomorrow = current_value
    
    # Determine trend direction
    pct_change_predicted = ((predicted_tomorrow - current_value) / current_value) * 100
    
    if pct_change_predicted > 0.5:
        trend = "UP"
        trend_id = "NAIK"
    elif pct_change_predicted < -0.5:
        trend = "DOWN"
        trend_id = "TURUN"
    else:
        trend = "STABLE"
        trend_id = "STABIL"
    
    return {
        "current_value": current_value,
        "past_value_7d": past_value,
        "predicted_tomorrow": predicted_tomorrow,
        "pct_change_7d": round(pct_change_7d, 2),
        "pct_change_predicted": round(pct_change_predicted, 2),
        "ma_30": ma_30,
        "volatility_pct": round(volatility, 2),
        "trend": trend,
        "trend_id": trend_id,
    }


def _generate_recommendation(analyses: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Generate investment recommendation based on all analyses.
    """
    usd = analyses.get("idr-usd", {})
    sar = analyses.get("idr-sar", {})
    gold = analyses.get("idr-gold-gram", {})
    
    recommendations = []
    scores = {
        "USD": 0,
        "SAR": 0,
        "GOLD": 0,
        "IDR": 0,
    }

    # Analyze USD trend (value is USD per IDR, so UP means IDR weakening)
    if usd:
        if usd["trend"] == "DOWN":  # IDR strengthening against USD
            scores["IDR"] += 2
            recommendations.append("IDR menguat terhadap USD, pertimbangkan untuk hold IDR.")
        elif usd["trend"] == "UP":  # IDR weakening
            scores["USD"] += 2
            recommendations.append("IDR melemah terhadap USD, pertimbangkan diversifikasi ke USD.")
        
        if usd["volatility_pct"] > 2:
            recommendations.append(f"Volatilitas USD/IDR tinggi ({usd['volatility_pct']}%), pasar sedang tidak stabil.")
    
    # Analyze SAR trend (value is SAR per IDR)
    if sar:
        if sar["trend"] == "DOWN":  # IDR strengthening against SAR
            scores["IDR"] += 1
            recommendations.append("IDR menguat terhadap SAR.")
        elif sar["trend"] == "UP":  # IDR weakening
            scores["SAR"] += 1
            recommendations.append("IDR melemah terhadap SAR, pertimbangkan SAR untuk kebutuhan Timur Tengah.")
    
    # Analyze Gold trend (value is IDR per gram, UP means gold more expensive in IDR)
    if gold:
        if gold["trend"] == "UP":  # Gold price increasing
            scores["GOLD"] += 3
            recommendations.append(f"Harga emas naik {gold['pct_change_predicted']}%, emas masih menjadi safe haven yang baik.")
        elif gold["trend"] == "DOWN":
            scores["IDR"] += 1
            recommendations.append("Harga emas turun, mungkin saat yang baik untuk membeli emas.")
        else:
            recommendations.append("Harga emas cenderung stabil.")
        
        if gold["volatility_pct"] < 1:
            recommendations.append("Volatilitas emas rendah, cocok untuk investasi jangka panjang.")
    
    # Determine best recommendation
    best_option = max(scores, key=scores.get)
    
    if best_option == "IDR":
        main_recommendation = "HOLD IDR - Rupiah dalam kondisi cukup kuat, simpan dalam deposito IDR."
    elif best_option == "USD":
        main_recommendation = "DIVERSIFIKASI USD - Pertimbangkan menyimpan sebagian dana dalam USD sebagai lindung nilai."
    elif best_option == "SAR":
        main_recommendation = "DIVERSIFIKASI SAR - Untuk kebutuhan Timur Tengah atau haji/umrah, SAR bisa dipertimbangkan."
    elif best_option == "GOLD":
        main_recommendation = "INVESTASI EMAS - Emas menunjukkan tren positif, cocok untuk diversifikasi dan lindung nilai."
    else:
        main_recommendation = "DIVERSIFIKASI - Sebaiknya sebarkan investasi ke berbagai instrumen."
    
    return {
        "main_recommendation": main_recommendation,
        "best_option": best_option,
        "scores": scores,
        "details": recommendations,
    }


@router.get("/idr-summary")
async def get_idr_summary():
    """
    Get comprehensive IDR analysis summary comparing against USD, SAR, and Gold.
    Provides trend analysis, predictions, and investment recommendations.
    """
    analyses = {}
    errors = []
    
    # Analyze each pair
    pairs = ["idr-usd", "idr-sar", "idr-gold-gram"]
    pair_labels = {
        "idr-usd": {"label": "IDR vs USD", "unit": "USD per IDR"},
        "idr-sar": {"label": "IDR vs SAR", "unit": "SAR per IDR"},
        "idr-gold-gram": {"label": "Harga Emas (IDR/gram)", "unit": "IDR per gram"},
    }
    
    for pair in pairs:
        try:
            analysis = _get_trend_analysis(pair)
            analyses[pair] = {
                **pair_labels[pair],
                **analysis,
            }
        except HTTPException as e:
            errors.append(f"{pair}: {e.detail}")
        except Exception as e:
            errors.append(f"{pair}: {str(e)}")
    
    if not analyses:
        raise HTTPException(status_code=503, detail=f"Tidak dapat menganalisis data. Errors: {errors}")
    
    # Generate recommendation
    recommendation = _generate_recommendation(analyses)
    
    # Build summary
    summary_parts = []
    
    for pair, data in analyses.items():
        label = data["label"]
        trend_id = data["trend_id"]
        pct = data["pct_change_predicted"]
        summary_parts.append(f"{label}: {trend_id} ({pct:+.2f}%)")
    
    summary_text = " | ".join(summary_parts)
    
    return {
        "date": datetime.utcnow().date().isoformat(),
        "summary": summary_text,
        "analyses": analyses,
        "recommendation": recommendation,
        "errors": errors if errors else None,
        "note": "Analisis ini berdasarkan data historis dan model prediksi. Bukan saran investasi finansial.",
    }
