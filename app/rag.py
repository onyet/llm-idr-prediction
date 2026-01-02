import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

import pandas as pd
from prophet import Prophet
from fastapi import APIRouter, HTTPException, Request

from .i18n import get_lang_from_request, t, get_current_lang, tr

router = APIRouter(prefix="/rag", tags=["rag"])

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# simple in-memory cache for loaded data and models
_cache: Dict[str, Any] = {"models": {}, "data": {}}


def load_data(pair: str, lang: str = "en") -> pd.DataFrame:
    """
    Memuat deret waktu untuk pasangan logis.
    - 'idr-usd' membaca USDIDR_X.json (IDR per USD) dan memakai harga Close.
    - 'idr-sar' dihitung dari (USDIDR_Close / SAR_Close) -> IDR per SAR.
    Mengeluarkan HTTPException(status_code=503) dengan pesan terlokalisasi jika file sumber hilang atau rusak.
    """
    if pair in _cache["data"]:
        return _cache["data"][pair]

    # Helper untuk memuat file JSON bergaya yfinance dan mengembalikannya sebagai DataFrame dengan kolom ds dan close
    def _load_yf_file(path: Path, name: str) -> pd.DataFrame:
        if not path.exists():
            raise HTTPException(status_code=503, detail=t("model_data_not_ready_named", lang, name=name))
        raw = json.loads(path.read_text())
        data = raw.get("data") or raw.get("prices") or raw
        df = pd.DataFrame(data)
        if df.empty:
            raise HTTPException(status_code=503, detail=t("model_data_empty", lang, name=name))
        # Normalisasi nama kolom menjadi lowercase untuk ketahanan
        df.columns = [c.lower() for c in df.columns]
        # Parsing tanggal
        if "date" in df.columns:
            df["ds"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
        else:
            # coba infer dari index atau kolom lain yang mirip tanggal
            possible = [c for c in df.columns if "time" in c or "date" in c or "ds" in c]
            if possible:
                df["ds"] = pd.to_datetime(df[possible[0]], utc=True).dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
            else:
                # Jika semua gagal, coba gunakan index
                try:
                    df["ds"] = pd.to_datetime(df.index, utc=True).dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
                except Exception:
                    raise HTTPException(status_code=503, detail=t("model_data_no_date", lang, name=name))
        # Harga penutupan (close)
        if "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
        else:
            raise HTTPException(status_code=503, detail=t("model_data_no_close", lang, name=name))
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
            raise HTTPException(status_code=503, detail=t("no_overlapping_dates", lang, pair="idr-sar"))
        # avoid division by zero
        merged = merged[(merged["close_sar"] != 0) & (merged["close_usd"] != 0)]
        merged["y"] = merged["close_sar"] / merged["close_usd"]  # SAR per IDR
        out = merged[["ds", "y"]].dropna().sort_values("ds")
        if out.empty:
            raise HTTPException(status_code=503, detail=t("no_valid_values", lang, pair="idr-sar"))
        _cache["data"][pair] = out
        return out

    if pair == "idr-gold-gram":
        p_usd = DATA_DIR / "USDIDR_X.json"
        p_gold = DATA_DIR / "GC_F.json"
        df_usd = _load_yf_file(p_usd, "USDIDR")
        df_gold = _load_yf_file(p_gold, "GC=F")

        # Merge berdasarkan tanggal (inner join)
        merged = pd.merge(df_usd, df_gold, on="ds", how="inner", suffixes=("_usd", "_gold"))
        if merged.empty:
            raise HTTPException(status_code=503, detail=t("no_overlapping_dates", lang, pair="idr-gold-gram"))
            
        # 1 Troy Ounce = 31.1034768 gram
        # Harga emas (USD/gram) = Harga emas (USD/oz) / 31.1034768
        # Harga emas (IDR/gram) = (Harga emas USD/gram) * (IDR/USD)
        
        troy_oz_to_gram = 31.1034768
        merged["y"] = (merged["close_gold"] / troy_oz_to_gram) * merged["close_usd"]
        
        out = merged[["ds", "y"]].dropna().sort_values("ds")
        if out.empty:
            raise HTTPException(status_code=503, detail=t("no_valid_values", lang, pair="idr-gold-gram"))
        _cache["data"][pair] = out
        return out

    raise FileNotFoundError("Unknown pair")


def get_model(pair: str, lang: str = "en") -> Prophet:
    if pair in _cache["models"]:
        return _cache["models"][pair]
    try:
        df = load_data(pair, lang)
    except HTTPException:
        # propagate the HTTPException (e.g., 503 model not ready)
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=t("model_data_not_ready", lang))
    # pemeriksaan sanity
    if df.empty:
        raise HTTPException(status_code=503, detail=t("model_data_empty", lang, name=pair))
    # latih model Prophet
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    _cache["models"][pair] = model
    return model


@router.get("/idr-sar")
async def predict_idr_sar(request: Request, days: int = 0, amount: float = 1.0):
    lang = get_lang_from_request(request)
    return await _predict("idr-sar", days, amount, lang)


@router.get("/idr-usd")
async def predict_idr_usd(request: Request, days: int = 0, amount: float = 1.0):
    lang = get_lang_from_request(request)
    return await _predict("idr-usd", days, amount, lang)


@router.get("/idr-gold-gram")
async def predict_idr_gold_gram(request: Request, days: int = 0, amount_gram: float = 1.0):
    lang = get_lang_from_request(request)
    return await _predict("idr-gold-gram", days, amount_gram, lang)


async def _predict(pair: str, days: int = 0, amount: float = 1.0, lang: str = "en"):
    # validasi input
    if days < 0:
        raise HTTPException(status_code=400, detail=t("days_must_be_positive", lang))
    if days > 7:
        raise HTTPException(status_code=400, detail=t("forecast_too_long", lang))
    if amount <= 0:
        raise HTTPException(status_code=400, detail=t("amount_must_be_positive", lang))

    df = load_data(pair, lang)
    model = get_model(pair, lang)

    # forecast up to requested day (0=today)
    last_date = df["ds"].max().normalize()
    target_date = datetime.utcnow().date() + timedelta(days=days)
    horizon_days = (target_date - last_date.date()).days
    if horizon_days < 0:
        # if target within historical data, just return observed
        observed = df[df["ds"].dt.date == target_date]
        if observed.empty:
            raise HTTPException(status_code=404, detail=t("no_data_for_date", lang))
        val = float(observed.iloc[-1]["y"])
        return {
            "pair": pair,
            "method": "prophet",
            "date": target_date.isoformat(),
            "predicted": val,
            "amount": amount,
            "predicted_for_amount": val * amount,
            "note": t("note_historical_value", lang),
        }

    # build dates to forecast from last_date +1 .. target_date
    future = model.make_future_dataframe(periods=horizon_days + 1, freq="D")
    forecast = model.predict(future)
    # select the target row
    target_row = forecast[forecast["ds"].dt.date == target_date]
    if target_row.empty:
        raise HTTPException(status_code=500, detail=t("prediction_not_available", lang))
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


def _get_trend_analysis(pair: str, days: int = 7, lang: str = "en") -> Dict[str, Any]:
    """
    Menganalisis trend untuk pasangan selama jumlah hari yang ditentukan.
    Mengembalikan arah trend, perubahan persentase, dan volatilitas.
    """
    df = load_data(pair, lang)
    model = get_model(pair, lang)
    
    last_date = df["ds"].max().normalize()
    
    # Ambil data historis terbaru (30 hari terakhir untuk analisis trend)
    recent_df = df[df["ds"] >= (last_date - timedelta(days=30))]
    
    if len(recent_df) < 5:
        recent_df = df.tail(30)
    
    # Nilai saat ini (pengamatan terakhir)
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
        trend_id = t("trend_up", lang)
    elif pct_change_predicted < -0.5:
        trend = "DOWN"
        trend_id = t("trend_down", lang)
    else:
        trend = "STABLE"
        trend_id = t("trend_stable", lang)
    
    return {
        "current_value": current_value,
        "past_value_7d": past_value,
        "predicted_tomorrow": predicted_tomorrow,
        "pct_change_7d": round(pct_change_7d, 2),
        "pct_change_predicted": round(pct_change_predicted, 2),
        "ma_30": ma_30,
        "volatility_pct": round(volatility, 2),
        "trend": trend,
        "trend_localized": trend_id,
    }


def _generate_recommendation(analyses: Dict[str, Dict], lang: str = "en") -> Dict[str, Any]:
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
            recommendations.append(t("idr_strengthening_usd", lang))
        elif usd["trend"] == "UP":  # IDR weakening
            scores["USD"] += 2
            recommendations.append(t("idr_weakening_usd", lang))
        
        if usd["volatility_pct"] > 2:
            recommendations.append(t("high_volatility_usd", lang, pct=usd['volatility_pct']))
    
    # Analyze SAR trend (value is SAR per IDR)
    if sar:
        if sar["trend"] == "DOWN":  # IDR strengthening against SAR
            scores["IDR"] += 1
            recommendations.append(t("idr_strengthening_sar", lang))
        elif sar["trend"] == "UP":  # IDR weakening
            scores["SAR"] += 1
            recommendations.append(t("idr_weakening_sar", lang))
    
    # Analyze Gold trend (value is IDR per gram, UP means gold more expensive in IDR)
    if gold:
        if gold["trend"] == "UP":  # Gold price increasing
            scores["GOLD"] += 3
            recommendations.append(t("gold_price_rising", lang, pct=gold['pct_change_predicted']))
        elif gold["trend"] == "DOWN":
            scores["IDR"] += 1
            recommendations.append(t("gold_price_falling", lang))
        else:
            recommendations.append(t("gold_price_stable", lang))
        
        if gold["volatility_pct"] < 1:
            recommendations.append(t("gold_low_volatility", lang))
    
    # Determine best recommendation
    best_option = max(scores, key=scores.get)
    
    if best_option == "IDR":
        main_recommendation = t("rec_hold_idr", lang)
    elif best_option == "USD":
        main_recommendation = t("rec_diversify_usd", lang)
    elif best_option == "SAR":
        main_recommendation = t("rec_diversify_sar", lang)
    elif best_option == "GOLD":
        main_recommendation = t("rec_invest_gold", lang)
    else:
        main_recommendation = t("rec_diversify_general", lang)
    
    return {
        "main_recommendation": main_recommendation,
        "best_option": best_option,
        "scores": scores,
        "details": recommendations,
    }


@router.get("/idr-summary")
async def get_idr_summary(request: Request):
    """
    Get comprehensive IDR analysis summary comparing against USD, SAR, and Gold.
    Provides trend analysis, predictions, and investment recommendations.
    """
    lang = get_lang_from_request(request)
    analyses = {}
    errors = []
    
    # Analyze each pair
    pairs = ["idr-usd", "idr-sar", "idr-gold-gram"]
    pair_labels = {
        "idr-usd": {"label": t("pair_idr_usd_label", lang), "unit": t("pair_idr_usd_unit", lang)},
        "idr-sar": {"label": t("pair_idr_sar_label", lang), "unit": t("pair_idr_sar_unit", lang)},
        "idr-gold-gram": {"label": t("pair_gold_label", lang), "unit": t("pair_gold_unit", lang)},
    }
    
    for pair in pairs:
        try:
            analysis = _get_trend_analysis(pair, lang=lang)
            analyses[pair] = {
                **pair_labels[pair],
                **analysis,
            }
        except HTTPException as e:
            errors.append(f"{pair}: {e.detail}")
        except Exception as e:
            errors.append(f"{pair}: {str(e)}")
    
    if not analyses:
        raise HTTPException(status_code=503, detail=t("cannot_analyze_data", lang, errors=errors))
    
    # Generate recommendation
    recommendation = _generate_recommendation(analyses, lang)
    
    # Build summary
    summary_parts = []
    
    for pair, data in analyses.items():
        label = data["label"]
        trend_localized = data["trend_localized"]
        pct = data["pct_change_predicted"]
        summary_parts.append(f"{label}: {trend_localized} ({pct:+.2f}%)")
    
    summary_text = " | ".join(summary_parts)
    
    return {
        "date": datetime.utcnow().date().isoformat(),
        "summary": summary_text,
        "analyses": analyses,
        "recommendation": recommendation,
        "errors": errors if errors else None,
        "note": t("note_analysis_disclaimer", lang),
    }
