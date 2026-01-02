"""
Advanced Analysis Module for IDR
Provides comprehensive fundamental, technical, and AI-powered analysis.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from fastapi import APIRouter, HTTPException, Request, Query

from .i18n import get_lang_from_request, t
from .technical import full_technical_analysis, calculate_volatility, detect_trend, calculate_sma, calculate_ema
from .llm_agent import get_llm_analyzer, AnalysisContext

router = APIRouter(prefix="/rag", tags=["rag"])

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Cache for models and data
_analyze_cache: Dict[str, Any] = {"models": {}, "data": {}}


def _load_json_data(file_path: Path, name: str, lang: str) -> pd.DataFrame:
    """Load data from JSON file and return as DataFrame."""
    if not file_path.exists():
        raise HTTPException(
            status_code=503,
            detail=t("model_data_not_ready_named", lang, module="rag", name=name)
        )

    raw = json.loads(file_path.read_text())
    data = raw.get("data") or raw.get("prices") or raw
    df = pd.DataFrame(data)

    if df.empty:
        raise HTTPException(
            status_code=503,
            detail=t("model_data_empty", lang, module="rag", name=name)
        )

    # Normalize columns
    df.columns = [c.lower() for c in df.columns]

    # Parse date
    if "date" in df.columns:
        df["ds"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
    elif "timestamp" in df.columns:
        df["ds"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert('UTC').dt.tz_localize(None).dt.normalize()
    else:
        raise HTTPException(
            status_code=503,
            detail=t("model_data_no_date", lang, module="rag", name=name)
        )

    # Get price columns
    for col in ["close", "price", "value", "y"]:
        if col in df.columns:
            df["close"] = pd.to_numeric(df[col], errors="coerce")
            break

    if "close" not in df.columns:
        raise HTTPException(
            status_code=503,
            detail=t("model_data_no_close", lang, module="rag", name=name)
        )

    # Get OHLC if available
    for col in ["open", "high", "low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["ds", "close"]).sort_values("ds")


def _get_analysis_methods(lang: str) -> Dict[str, Any]:
    """Return the analysis methods used with explanations."""
    return {
        "fundamental": {
            "title": t("method_fundamental_title", lang, module="analyze"),
            "description": t("method_fundamental_desc", lang, module="analyze"),
            "indicators": [
                {
                    "name": t("method_price_momentum", lang, module="analyze"),
                    "description": t("method_price_momentum_desc", lang, module="analyze")
                },
                {
                    "name": t("method_avg_comparison", lang, module="analyze"),
                    "description": t("method_avg_comparison_desc", lang, module="analyze")
                },
                {
                    "name": t("method_trend_consistency", lang, module="analyze"),
                    "description": t("method_trend_consistency_desc", lang, module="analyze")
                }
            ]
        },
        "technical": {
            "title": t("method_technical_title", lang, module="analyze"),
            "description": t("method_technical_desc", lang, module="analyze"),
            "indicators": [
                {
                    "name": "RSI (Relative Strength Index)",
                    "description": t("method_rsi_desc", lang, module="analyze")
                },
                {
                    "name": "MACD (Moving Average Convergence Divergence)",
                    "description": t("method_macd_desc", lang, module="analyze")
                },
                {
                    "name": "Bollinger Bands",
                    "description": t("method_bollinger_desc", lang, module="analyze")
                },
                {
                    "name": "Moving Averages (SMA/EMA)",
                    "description": t("method_ma_desc", lang, module="analyze")
                },
                {
                    "name": t("method_support_resistance", lang, module="analyze"),
                    "description": t("method_support_resistance_desc", lang, module="analyze")
                }
            ]
        }
    }


def _analyze_trend(df: pd.DataFrame, lang: str) -> Dict[str, Any]:
    """
    Perform comprehensive trend analysis.
    """
    data = df["y"].copy()
    
    # Short, medium, long term trends
    sma_5 = calculate_sma(data, 5)
    sma_10 = calculate_sma(data, 10)
    sma_20 = calculate_sma(data, 20)
    sma_50 = calculate_sma(data, 50) if len(data) >= 50 else calculate_sma(data, len(data) // 2)
    
    # Current values
    current = float(data.iloc[-1])
    sma_5_val = float(sma_5.iloc[-1])
    sma_10_val = float(sma_10.iloc[-1])
    sma_20_val = float(sma_20.iloc[-1])
    sma_50_val = float(sma_50.iloc[-1])
    
    # Determine short-term trend (5-day)
    short_trend = _classify_trend(data, 5)
    
    # Medium-term trend (10-20 days)
    medium_trend = _classify_trend(data, 20)
    
    # Long-term trend (30-50 days)
    long_trend = _classify_trend(data, min(50, len(data) - 1))
    
    # Trend strength based on MA alignment
    trend_strength = _calculate_trend_strength(current, sma_5_val, sma_10_val, sma_20_val, sma_50_val)
    
    # Price momentum
    momentum_5d = ((current - float(data.iloc[-6])) / float(data.iloc[-6]) * 100) if len(data) > 5 else 0
    momentum_10d = ((current - float(data.iloc[-11])) / float(data.iloc[-11]) * 100) if len(data) > 10 else 0
    momentum_20d = ((current - float(data.iloc[-21])) / float(data.iloc[-21]) * 100) if len(data) > 20 else 0
    
    # Determine overall trend
    overall_trend = _determine_overall_trend(short_trend, medium_trend, long_trend)
    
    # Trend reversal detection
    reversal_signal = _detect_trend_reversal(data, sma_5, sma_20)
    
    return {
        "short_term": {
            "period": "5 " + t("days", lang, module="analyze"),
            "direction": t(f"trend_{short_trend}", lang, module="analyze"),
            "momentum": round(momentum_5d, 2)
        },
        "medium_term": {
            "period": "20 " + t("days", lang, module="analyze"),
            "direction": t(f"trend_{medium_trend}", lang, module="analyze"),
            "momentum": round(momentum_20d, 2)
        },
        "long_term": {
            "period": "50 " + t("days", lang, module="analyze"),
            "direction": t(f"trend_{long_trend}", lang, module="analyze"),
            "momentum": round(momentum_20d, 2)
        },
        "overall": {
            "direction": overall_trend,
            "direction_localized": t(f"trend_{overall_trend}", lang, module="analyze"),
            "strength": trend_strength,
            "strength_localized": t(f"strength_{trend_strength}", lang, module="analyze")
        },
        "reversal_signal": reversal_signal,
        "reversal_signal_localized": t(f"reversal_{reversal_signal}", lang, module="analyze") if reversal_signal else None,
        "ma_alignment": {
            "sma_5": round(sma_5_val, 8),
            "sma_10": round(sma_10_val, 8),
            "sma_20": round(sma_20_val, 8),
            "sma_50": round(sma_50_val, 8),
            "price_vs_sma_20": round((current / sma_20_val - 1) * 100, 2)
        }
    }


def _classify_trend(data: pd.Series, period: int) -> str:
    """Classify trend direction based on period."""
    if len(data) < period + 1:
        return "sideways"
    
    start_val = float(data.iloc[-period - 1])
    end_val = float(data.iloc[-1])
    pct_change = ((end_val - start_val) / start_val) * 100
    
    if pct_change > 2:
        return "bullish"
    elif pct_change < -2:
        return "bearish"
    else:
        return "sideways"


def _calculate_trend_strength(current: float, sma5: float, sma10: float, sma20: float, sma50: float) -> str:
    """Calculate trend strength based on MA alignment."""
    # Perfect bullish alignment: price > sma5 > sma10 > sma20 > sma50
    # Perfect bearish alignment: price < sma5 < sma10 < sma20 < sma50
    
    bullish_score = 0
    bearish_score = 0
    
    if current > sma5:
        bullish_score += 1
    else:
        bearish_score += 1
    
    if sma5 > sma10:
        bullish_score += 1
    else:
        bearish_score += 1
    
    if sma10 > sma20:
        bullish_score += 1
    else:
        bearish_score += 1
    
    if sma20 > sma50:
        bullish_score += 1
    else:
        bearish_score += 1
    
    max_score = max(bullish_score, bearish_score)
    
    if max_score >= 4:
        return "strong"
    elif max_score >= 3:
        return "moderate"
    else:
        return "weak"


def _determine_overall_trend(short: str, medium: str, long: str) -> str:
    """Determine overall trend from multiple timeframes."""
    trends = [short, medium, long]
    bullish_count = trends.count("bullish")
    bearish_count = trends.count("bearish")
    
    if bullish_count >= 2:
        return "bullish"
    elif bearish_count >= 2:
        return "bearish"
    else:
        return "sideways"


def _detect_trend_reversal(data: pd.Series, sma_short: pd.Series, sma_long: pd.Series) -> Optional[str]:
    """Detect potential trend reversal signals."""
    if len(data) < 3:
        return None
    
    # Check for MA crossover
    curr_short = float(sma_short.iloc[-1])
    curr_long = float(sma_long.iloc[-1])
    prev_short = float(sma_short.iloc[-2])
    prev_long = float(sma_long.iloc[-2])
    
    if curr_short > curr_long and prev_short <= prev_long:
        return "bullish_crossover"
    elif curr_short < curr_long and prev_short >= prev_long:
        return "bearish_crossover"
    
    # Check for price divergence
    recent = data.tail(5)
    if recent.is_monotonic_increasing and curr_short < curr_long:
        return "potential_bullish_reversal"
    elif recent.is_monotonic_decreasing and curr_short > curr_long:
        return "potential_bearish_reversal"
    
    return None


def _generate_human_explanation(
    analyses: Dict[str, Any],
    trend_analysis: Dict[str, Any],
    lang: str
) -> Dict[str, Any]:
    """Generate human-readable explanation of the analysis."""
    
    # Get primary analysis (IDR/USD)
    primary = analyses.get("idr-usd", list(analyses.values())[0])
    
    # Build situation explanation
    trend_dir = trend_analysis.get("overall", {}).get("direction", "sideways")
    trend_strength = trend_analysis.get("overall", {}).get("strength", "moderate")

    # Current situation
    situation = t(f"explanation_situation_{trend_dir}_{trend_strength}", lang, module="analyze")
    
    # Technical condition
    tech = primary.get("technical", {})
    rsi_condition = tech.get("rsi", {}).get("condition", "neutral")
    macd_trend = tech.get("macd", {}).get("trend", "neutral")
    
    tech_explanation = t("explanation_technical_summary", lang, module="analyze",
                         rsi=rsi_condition,
                         macd=macd_trend,
                         volatility=round(tech.get("volatility", 0), 1))
    
    # Fundamental condition
    fund = primary.get("fundamental", {})
    fund_strength = fund.get("strength", "neutral")
    pct_7d = fund.get("pct_change_7d", 0)
    pct_30d = fund.get("pct_change_30d", 0)
    
    fund_explanation = t("explanation_fundamental_summary", lang, module="analyze",
                         strength=t(f"fundamental_{fund_strength.replace('strong_', '')}", lang, module="analyze"),
                         pct_7d=pct_7d,
                         pct_30d=pct_30d)
    
    # Market outlook
    if trend_dir == "bullish":
        outlook_key = "explanation_outlook_bullish"
    elif trend_dir == "bearish":
        outlook_key = "explanation_outlook_bearish"
    else:
        outlook_key = "explanation_outlook_sideways"
    
    market_outlook = t(outlook_key, lang, module="analyze")
    
    # Risk warning based on volatility
    volatility = tech.get("volatility", 0)
    if volatility > 25:
        risk_warning = t("explanation_risk_high", lang, module="analyze")
    elif volatility > 15:
        risk_warning = t("explanation_risk_medium", lang, module="analyze")
    else:
        risk_warning = t("explanation_risk_low", lang, module="analyze")
    
    return {
        "title": t("explanation_title", lang, module="analyze"),
        "current_situation": situation,
        "technical_condition": tech_explanation,
        "fundamental_condition": fund_explanation,
        "market_outlook": market_outlook,
        "risk_warning": risk_warning
    }


def _generate_recommendations(
    analyses: Dict[str, Any],
    trend_analysis: Dict[str, Any],
    lang: str
) -> Dict[str, Any]:
    """Generate actionable recommendations based on analysis."""
    
    primary = analyses.get("idr-usd", list(analyses.values())[0])
    tech = primary.get("technical", {})
    fund = primary.get("fundamental", {})
    
    trend_dir = trend_analysis.get("overall", {}).get("direction", "sideways")
    trend_strength = trend_analysis.get("overall", {}).get("strength", "moderate")
    signal = tech.get("signal", "hold")
    volatility = tech.get("volatility", 0)
    rsi = tech.get("rsi", {}).get("value", 50)
    
    # Primary recommendation based on trend and signals
    recommendations = []
    actions = []
    
    # Trend-based recommendation
    if trend_dir == "bullish" and trend_strength in ["strong", "moderate"]:
        if signal in ["strong_buy", "buy"]:
            recommendations.append(t("rec_bullish_trend_buy", lang, module="analyze"))
            actions.append(t("action_consider_buy", lang, module="analyze"))
        else:
            recommendations.append(t("rec_bullish_trend_hold", lang, module="analyze"))
            actions.append(t("action_wait_confirmation", lang, module="analyze"))
    
    elif trend_dir == "bearish" and trend_strength in ["strong", "moderate"]:
        if signal in ["strong_sell", "sell"]:
            recommendations.append(t("rec_bearish_trend_sell", lang, module="analyze"))
            actions.append(t("action_consider_sell", lang, module="analyze"))
        else:
            recommendations.append(t("rec_bearish_trend_caution", lang, module="analyze"))
            actions.append(t("action_reduce_exposure", lang, module="analyze"))
    
    else:  # sideways or weak trend
        recommendations.append(t("rec_sideways_trend", lang, module="analyze"))
        actions.append(t("action_wait_breakout", lang, module="analyze"))
    
    # RSI-based caution
    if rsi > 70:
        recommendations.append(t("rec_rsi_overbought", lang, module="analyze"))
        actions.append(t("action_avoid_buying", lang, module="analyze"))
    elif rsi < 30:
        recommendations.append(t("rec_rsi_oversold", lang, module="analyze"))
        actions.append(t("action_look_for_entry", lang, module="analyze"))
    
    # Volatility-based risk management
    if volatility > 25:
        recommendations.append(t("rec_high_volatility", lang, module="analyze"))
        actions.append(t("action_use_stop_loss", lang, module="analyze"))
    
    # Reversal signal
    reversal = trend_analysis.get("reversal_signal")
    if reversal:
        if "bullish" in reversal:
            recommendations.append(t("rec_bullish_reversal", lang, module="analyze"))
        elif "bearish" in reversal:
            recommendations.append(t("rec_bearish_reversal", lang, module="analyze"))
    
    # Determine primary action
    if signal == "strong_buy":
        primary_action = t("primary_action_strong_buy", lang, module="analyze")
    elif signal == "buy":
        primary_action = t("primary_action_buy", lang, module="analyze")
    elif signal == "strong_sell":
        primary_action = t("primary_action_strong_sell", lang, module="analyze")
    elif signal == "sell":
        primary_action = t("primary_action_sell", lang, module="analyze")
    else:
        primary_action = t("primary_action_hold", lang, module="analyze")
    
    # Confidence level
    confidence = _calculate_recommendation_confidence(trend_dir, trend_strength, signal, volatility, rsi)
    
    return {
        "title": t("recommendations_title", lang, module="analyze"),
        "primary_action": primary_action,
        "confidence": confidence,
        "confidence_localized": t(f"confidence_{confidence}", lang, module="analyze"),
        "recommendations": recommendations,
        "suggested_actions": actions,
        "risk_management": {
            "stop_loss_suggestion": t("stop_loss_suggestion", lang, module="analyze", pct=round(volatility / 2, 1)),
            "position_sizing": t(f"position_sizing_{confidence}", lang, module="analyze")
        }
    }


def _calculate_recommendation_confidence(
    trend_dir: str, 
    trend_strength: str, 
    signal: str, 
    volatility: float, 
    rsi: float
) -> str:
    """Calculate confidence level for recommendations."""
    score = 0
    
    # Trend alignment
    if trend_dir != "sideways":
        score += 1
        if trend_strength == "strong":
            score += 2
        elif trend_strength == "moderate":
            score += 1
    
    # Signal strength
    if signal in ["strong_buy", "strong_sell"]:
        score += 2
    elif signal in ["buy", "sell"]:
        score += 1
    
    # Volatility penalty
    if volatility > 30:
        score -= 2
    elif volatility > 20:
        score -= 1
    
    # RSI extremes
    if 30 <= rsi <= 70:
        score += 1
    
    if score >= 5:
        return "high"
    elif score >= 3:
        return "medium"
    else:
        return "low"


def _get_pair_data(pair: str, lang: str) -> pd.DataFrame:
    """Get processed data for a currency pair."""
    cache_key = f"analyze_{pair}"
    if cache_key in _analyze_cache["data"]:
        return _analyze_cache["data"][cache_key]
    
    if pair == "idr-usd":
        df = _load_json_data(DATA_DIR / "USDIDR_X.json", "USDIDR", lang)
        df["y"] = 1 / df["close"]  # USD per IDR
        
    elif pair == "idr-sar":
        df_usd = _load_json_data(DATA_DIR / "USDIDR_X.json", "USDIDR", lang)
        df_sar = _load_json_data(DATA_DIR / "SAR_X.json", "SAR", lang)
        df = pd.merge(df_usd, df_sar, on="ds", how="inner", suffixes=("_usd", "_sar"))
        df = df[(df["close_sar"] != 0) & (df["close_usd"] != 0)]
        df["y"] = df["close_sar"] / df["close_usd"]  # SAR per IDR
        df["close"] = df["y"]
        
    elif pair == "idr-gold-gram":
        df_usd = _load_json_data(DATA_DIR / "USDIDR_X.json", "USDIDR", lang)
        df_gold = _load_json_data(DATA_DIR / "GC_F.json", "GC=F", lang)
        df = pd.merge(df_usd, df_gold, on="ds", how="inner", suffixes=("_usd", "_gold"))
        troy_oz_to_gram = 31.1034768
        df["y"] = (df["close_gold"] / troy_oz_to_gram) * df["close_usd"]  # IDR per gram
        df["close"] = df["y"]
        
    else:
        raise HTTPException(status_code=400, detail=f"Unknown pair: {pair}")
    
    result = df[["ds", "y"]].dropna().copy()
    result["close"] = result["y"]
    _analyze_cache["data"][cache_key] = result
    return result


def _get_target_tracer_data(symbol: str, lang: str) -> Optional[pd.DataFrame]:
    """Load data for target tracer symbol. Fetches from yfinance if not cached locally or data is outdated."""
    import yfinance as yf
    import re
    
    # Try multiple file naming conventions
    possible_files = [
        DATA_DIR / f"{symbol}.json",
        DATA_DIR / f"{symbol.upper()}.json",
        DATA_DIR / f"{symbol.lower()}.json",
        DATA_DIR / f"{symbol.replace('.', '_')}.json",
    ]
    
    existing_file = None
    for file_path in possible_files:
        if file_path.exists():
            existing_file = file_path
            break
    
    # Check if existing data needs update
    should_fetch = True
    if existing_file:
        try:
            # Load existing data to check last date
            existing_df = _load_json_data(existing_file, symbol, lang)
            last_date = existing_df["ds"].max().date()
            today = datetime.utcnow().date()
            
            # If last date is today, data is fresh
            if last_date >= today:
                return existing_df
            
            # If last date is yesterday or older, need to fetch new data
            print(f"Data for {symbol} is outdated (last: {last_date}, today: {today}). Fetching new data...", file=__import__('sys').stderr)
            should_fetch = True
            
        except HTTPException:
            # If loading fails, fetch new data
            should_fetch = True
    
    # Fetch from yfinance if needed
    if should_fetch:
        try:
            # For Indonesian stocks, add .JK suffix if not present
            ticker_symbol = symbol.upper()
            suffixes = ['.JK', '.SI', '.HK', '.L', '.TO', '.AX', '.KS', '.TW']
        
        # Determine ticker to use
            if not any(ticker_symbol.endswith(suffix) for suffix in suffixes):
                # Try Indonesian stock exchange first
                test_ticker_jk = yf.Ticker(f"{ticker_symbol}.JK")
                hist_test = test_ticker_jk.history(period="5d")
                
                if not hist_test.empty:
                    ticker_symbol = f"{ticker_symbol}.JK"
                    ticker = test_ticker_jk
                else:
                    # Try without suffix (US stocks or forex)
                    ticker = yf.Ticker(ticker_symbol)
            else:
                ticker = yf.Ticker(ticker_symbol)
            
            # Fetch 5 years of data
            hist = ticker.history(period="5y")
            
            if hist.empty:
                return None
            
            # Convert to expected format
            hist = hist.reset_index()
            hist.columns = [c.lower() for c in hist.columns]
            
            # Prepare data for saving
            save_data = {
                "symbol": ticker_symbol,
                "period": "5y",
                "last_updated": datetime.utcnow().isoformat(),
                "data": []
            }
            
            for _, row in hist.iterrows():
                date_val = row.get("date")
                if pd.isna(date_val):
                    continue
                
                # Convert to string format
                if hasattr(date_val, 'strftime'):
                    date_str = date_val.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_val)[:10]
                
                save_data["data"].append({
                    "Date": date_str,
                    "Open": float(row.get("open", 0)),
                    "High": float(row.get("high", 0)),
                    "Low": float(row.get("low", 0)),
                    "Close": float(row.get("close", 0)),
                    "Volume": int(row.get("volume", 0))
                })
            
            # Save to file with sanitized filename
            safe_symbol = re.sub(r'[^a-zA-Z0-9]', '_', ticker_symbol)
            save_path = DATA_DIR / f"{safe_symbol}.json"
            
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"Successfully fetched and saved {len(save_data['data'])} records for {ticker_symbol}", file=__import__('sys').stderr)
            
            # Now load it using our standard function
            return _load_json_data(save_path, ticker_symbol, lang)
            
        except Exception as e:
            # Log error for debugging but don't crash
            import sys
            print(f"Failed to fetch {symbol} from yfinance: {e}", file=sys.stderr)
            
            # If fetch failed but we have existing data, return that
            if existing_file:
                try:
                    return _load_json_data(existing_file, symbol, lang)
                except:
                    pass
            
            return None

def _get_prophet_model(pair: str, df: pd.DataFrame) -> Prophet:
    """Get or create Prophet model for prediction."""
    cache_key = f"prophet_{pair}"
    if cache_key in _analyze_cache["models"]:
        return _analyze_cache["models"][cache_key]
    
    model = Prophet(daily_seasonality=True)
    model.fit(df[["ds", "y"]])
    _analyze_cache["models"][cache_key] = model
    return model


def _perform_fundamental_analysis(df: pd.DataFrame, pair: str, lang: str) -> Dict[str, Any]:
    """Perform fundamental analysis on the data."""
    recent_30d = df.tail(30)
    recent_7d = df.tail(7)
    recent_14d = df.tail(14)
    
    current_price = float(df["y"].iloc[-1])
    avg_30d = float(recent_30d["y"].mean())
    avg_7d = float(recent_7d["y"].mean())
    
    # High/Low analysis
    high_30d = float(recent_30d["y"].max())
    low_30d = float(recent_30d["y"].min())
    high_7d = float(recent_7d["y"].max())
    low_7d = float(recent_7d["y"].min())
    
    # Price momentum
    pct_change_30d = ((current_price - float(df["y"].iloc[-30])) / float(df["y"].iloc[-30]) * 100) if len(df) >= 30 else 0
    pct_change_7d = ((current_price - float(df["y"].iloc[-7])) / float(df["y"].iloc[-7]) * 100) if len(df) >= 7 else 0
    pct_change_1d = ((current_price - float(df["y"].iloc[-2])) / float(df["y"].iloc[-2]) * 100) if len(df) >= 2 else 0
    
    # Trend consistency (how many days price moved in same direction)
    price_changes = df["y"].diff().tail(14)
    positive_days = int((price_changes > 0).sum())
    negative_days = int((price_changes < 0).sum())
    
    # Distance from 30-day high/low
    pct_from_high = ((current_price - high_30d) / high_30d) * 100
    pct_from_low = ((current_price - low_30d) / low_30d) * 100
    
    # Determine fundamental strength
    if positive_days >= 10:
        strength = "strong_bullish"
        strength_text = t("fundamental_strong", lang, module="analyze")
    elif negative_days >= 10:
        strength = "strong_bearish"
        strength_text = t("fundamental_weak", lang, module="analyze")
    elif positive_days >= 8:
        strength = "bullish"
        strength_text = t("fundamental_bullish", lang, module="analyze")
    elif negative_days >= 8:
        strength = "bearish"
        strength_text = t("fundamental_bearish", lang, module="analyze")
    else:
        strength = "neutral"
        strength_text = t("fundamental_neutral", lang, module="analyze")
    
    # Generate fundamental insights
    insights = []
    if pct_change_7d > 5:
        insights.append(t("fund_insight_strong_gain_7d", lang, module="analyze", pct=round(pct_change_7d, 2)))
    elif pct_change_7d < -5:
        insights.append(t("fund_insight_strong_loss_7d", lang, module="analyze", pct=round(abs(pct_change_7d), 2)))
    
    if current_price > avg_30d * 1.05:
        insights.append(t("fund_insight_above_avg", lang, module="analyze"))
    elif current_price < avg_30d * 0.95:
        insights.append(t("fund_insight_below_avg", lang, module="analyze"))
    
    if abs(pct_from_high) < 5:
        insights.append(t("fund_insight_near_high", lang, module="analyze"))
    elif abs(pct_from_low) < 5:
        insights.append(t("fund_insight_near_low", lang, module="analyze"))
    
    return {
        "current_price": round(current_price, 8),
        "price_change": {
            "1d": round(pct_change_1d, 2),
            "7d": round(pct_change_7d, 2),
            "30d": round(pct_change_30d, 2)
        },
        "averages": {
            "avg_7d": round(avg_7d, 8),
            "avg_30d": round(avg_30d, 8),
            "price_vs_avg_30d": round((current_price / avg_30d - 1) * 100, 2)
        },
        "price_range": {
            "high_7d": round(high_7d, 8),
            "low_7d": round(low_7d, 8),
            "high_30d": round(high_30d, 8),
            "low_30d": round(low_30d, 8),
            "pct_from_30d_high": round(pct_from_high, 2),
            "pct_from_30d_low": round(pct_from_low, 2)
        },
        "trend_consistency": {
            "positive_days_14d": positive_days,
            "negative_days_14d": negative_days
        },
        "strength": strength,
        "strength_description": strength_text,
        "insights": insights
    }


def _generate_technical_insights(tech: Dict[str, Any], lang: str) -> List[str]:
    """Generate human-readable insights from technical analysis."""
    insights = []
    
    # RSI insight
    rsi = tech.get("rsi", {})
    rsi_value = rsi.get("value", 50)
    if rsi.get("condition") == "overbought":
        insights.append(t("rsi_overbought", lang, module="analyze", value=rsi_value))
    elif rsi.get("condition") == "oversold":
        insights.append(t("rsi_oversold", lang, module="analyze", value=rsi_value))
    else:
        insights.append(t("rsi_neutral", lang, module="analyze", value=rsi_value))
    
    # MACD insight
    macd = tech.get("macd", {})
    if macd.get("trend") == "bullish":
        insights.append(t("macd_bullish", lang, module="analyze"))
    else:
        insights.append(t("macd_bearish", lang, module="analyze"))
    
    # Bollinger Band insight
    bb = tech.get("bollinger_bands", {})
    bb_pos = bb.get("position", "middle")
    if bb_pos == "upper":
        insights.append(t("bollinger_upper", lang, module="analyze"))
    elif bb_pos == "lower":
        insights.append(t("bollinger_lower", lang, module="analyze"))
    else:
        insights.append(t("bollinger_middle", lang, module="analyze"))
    
    # MA crossover
    ma_cross = tech.get("ma_crossover")
    if ma_cross == "golden_cross":
        insights.append(t("ma_cross_bullish", lang, module="analyze", short=10, long=20))
    elif ma_cross == "death_cross":
        insights.append(t("ma_cross_bearish", lang, module="analyze", short=10, long=20))
    
    # Volatility
    vol = tech.get("volatility", 0)
    if vol > 25:
        insights.append(t("volatility_high", lang, module="analyze", value=vol))
    elif vol > 10:
        insights.append(t("volatility_medium", lang, module="analyze", value=vol))
    else:
        insights.append(t("volatility_low", lang, module="analyze", value=vol))
    
    return insights


def _get_signal_text(signal: str, lang: str) -> str:
    """Get localized signal text."""
    signal_map = {
        "strong_buy": "signal_strong_buy",
        "buy": "signal_buy",
        "hold": "signal_hold",
        "sell": "signal_sell",
        "strong_sell": "signal_strong_sell"
    }
    return t(signal_map.get(signal, "signal_hold"), lang, module="analyze")


def _compare_with_target(
    base_df: pd.DataFrame, 
    target_df: pd.DataFrame, 
    target_symbol: str,
    lang: str
) -> Dict[str, Any]:
    """Compare base pair performance with target tracer."""
    # Merge on date
    merged = pd.merge(
        base_df[["ds", "y"]], 
        target_df[["ds", "close"]], 
        on="ds",
        how="inner",
        suffixes=("_base", "_target")
    )
    
    if merged.empty or len(merged) < 7:
        return {"error": t("no_data_available", lang, module="analyze")}
    
    recent = merged.tail(30)
    
    # Calculate returns
    base_return = (float(recent["y"].iloc[-1]) / float(recent["y"].iloc[0]) - 1) * 100
    target_return = (float(recent["close"].iloc[-1]) / float(recent["close"].iloc[0]) - 1) * 100
    
    # Correlation
    correlation = float(recent["y"].corr(recent["close"]))
    
    # Comparison text
    if target_return > base_return + 2:
        comparison = t("comparison_better", lang, module="analyze", symbol=target_symbol)
    elif target_return < base_return - 2:
        comparison = t("comparison_worse", lang, module="analyze", symbol=target_symbol)
    else:
        comparison = t("comparison_similar", lang, module="analyze", symbol=target_symbol)
    
    return {
        "symbol": target_symbol,
        "base_return_30d": round(base_return, 2),
        "target_return_30d": round(target_return, 2),
        "correlation": round(correlation, 4),
        "comparison": comparison,
        "current_price": round(float(target_df["close"].iloc[-1]), 4),
        "data_points": len(merged),
    }


@router.get("/idr-analyze")
async def analyze_idr(
    request: Request,
    date: Optional[str] = Query(None, description="Target date (YYYY-MM-DD). If past date with data, uses historical. If future, uses prediction."),
    target_tracer: Optional[str] = Query(None, description="Additional symbol to compare (e.g., BBRI, AAPL)"),
    use_llm: bool = Query(True, description="Whether to use LLM for advanced analysis"),
):
    """
    Advanced IDR analysis endpoint.
    
    Analyzes IDR against USD, SAR, and Gold with:
    - Fundamental analysis (price momentum, averages, trend consistency)
    - Technical analysis (RSI, MACD, Bollinger Bands, Support/Resistance)
    - Trend analysis (short, medium, long term)
    - AI-powered insights (optional)
    - Human-readable explanations and recommendations
    - Target tracer comparison (optional)
    
    If date is in the past and data exists, returns historical analysis.
    If date is in the future, returns prediction-based analysis.
    """
    lang = get_lang_from_request(request)

    # Parse target date
    today = datetime.utcnow().date()
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=t("date_must_be_valid", lang, module="analyze")
            )
    else:
        target_date = today
    
    # Determine analysis type
    is_prediction = target_date > today
    analysis_type = t("analysis_type_prediction", lang, module="analyze") if is_prediction else t("analysis_type_historical", lang, module="analyze")
    
    # Pairs to analyze
    pairs = ["idr-usd", "idr-sar", "idr-gold-gram"]
    pair_names = {
        "idr-usd": "IDR/USD",
        "idr-sar": "IDR/SAR", 
        "idr-gold-gram": "IDR/Gold"
    }
    
    analyses = {}
    trend_analyses = {}
    errors = []
    
    for pair in pairs:
        try:
            df = _get_pair_data(pair, lang)
            last_data_date = df["ds"].max().date()
            
            # Determine if we use historical or prediction
            if target_date <= last_data_date:
                # Use historical data
                target_df = df[df["ds"].dt.date <= target_date]
                if target_df.empty:
                    errors.append(f"{pair}: No data for {target_date}")
                    continue
                data_source = t("data_source_historical", lang, module="analyze")
            else:
                # Use prediction
                model = _get_prophet_model(pair, df)
                horizon = (target_date - last_data_date).days
                future = model.make_future_dataframe(periods=horizon + 1, freq="D")
                forecast = model.predict(future)
                
                # Create predicted dataframe
                pred_df = forecast[["ds", "yhat"]].copy()
                pred_df["y"] = pred_df["yhat"]
                pred_df["close"] = pred_df["yhat"]
                pred_df = pred_df[pred_df["ds"].dt.date <= target_date]
                target_df = pd.concat([df, pred_df[pred_df["ds"] > df["ds"].max()]])
                data_source = t("data_source_prediction", lang, module="analyze")
            
            # Perform fundamental analysis (separate)
            fundamental = _perform_fundamental_analysis(target_df, pair, lang)
            
            # Perform technical analysis (separate)
            technical = full_technical_analysis(target_df, "y")
            tech_insights = _generate_technical_insights(technical, lang)
            
            # Perform trend analysis
            trend = _analyze_trend(target_df, lang)
            trend_analyses[pair] = trend
            
            # Get signal text
            signal_text = _get_signal_text(technical["signal"], lang)
            
            # Trend text
            trend_key = f"trend_{technical['trend']}"
            trend_text = t(trend_key, lang, module="analyze")
            
            analyses[pair] = {
                "name": pair_names[pair],
                "data_source": data_source,
                "target_date": target_date.isoformat(),
                "last_data_date": last_data_date.isoformat(),
                "fundamental": {
                    "title": t("fundamental_analysis", lang, module="analyze"),
                    **fundamental
                },
                "technical": {
                    "title": t("technical_analysis", lang, module="analyze"),
                    **technical,
                    "trend_localized": trend_text,
                    "signal_localized": signal_text,
                    "insights": tech_insights
                },
                "trend": trend
            }
            
        except HTTPException as e:
            errors.append(f"{pair}: {e.detail}")
        except Exception as e:
            errors.append(f"{pair}: {str(e)}")
    
    if not analyses:
        raise HTTPException(
            status_code=503,
            detail=t("no_data_available", lang, module="analyze")
        )
    
    # Get analysis methods used
    analysis_methods = _get_analysis_methods(lang)
    
    # Get primary trend analysis for overall assessment
    primary_trend = trend_analyses.get("idr-usd", list(trend_analyses.values())[0] if trend_analyses else {})
    
    # Generate human-readable explanation
    explanation = _generate_human_explanation(analyses, primary_trend, lang)
    
    # Generate trend-based recommendations
    recommendations = _generate_recommendations(analyses, primary_trend, lang)
    
    # Target tracer analysis
    tracer_analysis = None
    if target_tracer:
        tracer_df = _get_target_tracer_data(target_tracer, lang)
        if tracer_df is not None:
            # Compare with IDR/USD as base
            try:
                base_df = _get_pair_data("idr-usd", lang)
                tracer_analysis = _compare_with_target(base_df, tracer_df, target_tracer, lang)
                
                # Add technical analysis for tracer
                tracer_tech = full_technical_analysis(tracer_df, "close")
                tracer_analysis["technical"] = {
                    "trend": tracer_tech["trend"],
                    "signal": tracer_tech["signal"],
                    "rsi": tracer_tech["rsi"]["value"],
                    "volatility": tracer_tech["volatility"],
                }
            except Exception as e:
                tracer_analysis = {"error": str(e)}
        else:
            tracer_analysis = {
                "error": t("target_tracer_not_found", lang, module="analyze", symbol=target_tracer)
            }
    
    # LLM Analysis
    llm_analysis = None
    if use_llm:
        try:
            # Prepare context for LLM
            primary_pair = "idr-usd"
            primary_data = analyses.get(primary_pair, list(analyses.values())[0])
            
            # Build historical summary
            history_parts = []
            for p, a in analyses.items():
                fund = a["fundamental"]
                pct_7d = fund.get("price_change", {}).get("7d", 0)
                pct_30d = fund.get("price_change", {}).get("30d", 0)
                history_parts.append(
                    f"{a['name']}: {pct_7d:+.2f}% (7d), {pct_30d:+.2f}% (30d)"
                )
            historical_summary = " | ".join(history_parts)
            
            context = AnalysisContext(
                pair=primary_pair,
                current_price=primary_data["fundamental"]["current_price"],
                technical_indicators=primary_data["technical"],
                fundamental_data={p: a["fundamental"] for p, a in analyses.items()},
                historical_summary=historical_summary,
                target_date=target_date.isoformat(),
                analysis_type="prediction" if is_prediction else "historical",
                language=lang,
            )
            
            analyzer = get_llm_analyzer()
            llm_analysis = await analyzer.analyze(context)
            
        except Exception as e:
            llm_analysis = {
                "error": t("llm_analysis_unavailable", lang, module="analyze"),
                "details": str(e)
            }
    
    # Generate overall signal based on trend
    overall_trend = primary_trend.get("overall", {}).get("direction", "sideways")
    signals = [a["technical"]["signal_score"] for a in analyses.values()]
    avg_signal = sum(signals) / len(signals) if signals else 0
    
    # Combine signal with trend for overall recommendation
    if overall_trend == "bullish":
        if avg_signal >= 0:
            overall_signal = "strong_buy" if avg_signal >= 1 else "buy"
            outlook = t("outlook_positive", lang, module="analyze")
        else:
            overall_signal = "hold"
            outlook = t("outlook_neutral", lang, module="analyze")
    elif overall_trend == "bearish":
        if avg_signal <= 0:
            overall_signal = "strong_sell" if avg_signal <= -1 else "sell"
            outlook = t("outlook_negative", lang, module="analyze")
        else:
            overall_signal = "hold"
            outlook = t("outlook_neutral", lang, module="analyze")
    else:
        overall_signal = "hold"
        outlook = t("outlook_neutral", lang, module="analyze")
    
    # Risk assessment
    volatilities = [a["technical"]["volatility"] for a in analyses.values()]
    avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0
    
    if avg_volatility > 25:
        risk_level = t("risk_high", lang, module="analyze")
    elif avg_volatility > 15:
        risk_level = t("risk_medium", lang, module="analyze")
    else:
        risk_level = t("risk_low", lang, module="analyze")
    
    return {
        "title": t("analyze_title", lang, module="analyze"),
        "request": {
            "target_date": target_date.isoformat(),
            "analysis_type": analysis_type,
            "target_tracer": target_tracer,
            "use_llm": use_llm,
        },
        "generated_at": datetime.utcnow().isoformat(),
        
        # Analysis methods used
        "methodology": analysis_methods,
        
        # Detailed analyses per pair
        "analyses": analyses,
        
        # Overall trend analysis
        "trend_summary": {
            "title": t("trend_analysis_title", lang, module="analyze"),
            "primary_pair": "idr-usd",
            **primary_trend
        },
        
        # Human-readable explanation
        "explanation": explanation,
        
        # Trend-based recommendations
        "recommendations": recommendations,
        
        # Target tracer comparison
        "target_tracer": tracer_analysis,
        
        # AI analysis
        "ai_analysis": llm_analysis,
        
        # Summary
        "summary": {
            "overall_signal": overall_signal,
            "overall_signal_localized": _get_signal_text(overall_signal, lang),
            "overall_trend": overall_trend,
            "overall_trend_localized": t(f"trend_{overall_trend}", lang, module="analyze"),
            "outlook": outlook,
            "risk_level": risk_level,
            "avg_volatility": round(avg_volatility, 2),
        },
        
        "errors": errors if errors else None,
        "note": t("note_analysis_disclaimer", lang, module="rag"),
    }
