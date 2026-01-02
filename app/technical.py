"""
Modul Analisis Teknikal
Menyediakan indikator teknikal untuk analisis keuangan.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta


def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Hitung Simple Moving Average (SMA)."""
    return data.rolling(window=period, min_periods=1).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Hitung Exponential Moving Average (EMA)."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Hitung Relative Strength Index (RSI).
    RSI = 100 - (100 / (1 + RS))
    RS = Rata-rata Gain / Rata-rata Loss
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    Hitung MACD (Moving Average Convergence Divergence).
    """
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }


def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """
    Hitung Bollinger Bands.
    """
    sma = calculate_sma(data, period)
    std = data.rolling(window=period, min_periods=1).std()
    
    return {
        "middle": sma,
        "upper": sma + (std * std_dev),
        "lower": sma - (std * std_dev)
    }


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Hitung Average True Range (ATR).
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period, min_periods=1).mean()
    return atr


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """
    Calculate Stochastic Oscillator.
    """
    lowest_low = low.rolling(window=k_period, min_periods=1).min()
    highest_high = high.rolling(window=k_period, min_periods=1).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
    d = k.rolling(window=d_period, min_periods=1).mean()
    
    return {
        "k": k.fillna(50),
        "d": d.fillna(50)
    }


def find_support_resistance(data: pd.Series, window: int = 20) -> Dict[str, List[float]]:
    """
    Temukan level support dan resistance menggunakan minima/maksima lokal.
    """
    supports = []
    resistances = []
    
    for i in range(window, len(data) - window):
        # Cek minima lokal (support)
        if data.iloc[i] == data.iloc[i-window:i+window+1].min():
            supports.append(float(data.iloc[i]))
        # Cek maksimum lokal (resistance)
        if data.iloc[i] == data.iloc[i-window:i+window+1].max():
            resistances.append(float(data.iloc[i]))
    
    # Dapatkan level unik (kelompokkan nilai yang berdekatan)
    def cluster_levels(levels: List[float], threshold: float = 0.02) -> List[float]:
        if not levels:
            return []
        sorted_levels = sorted(set(levels))
        clustered = [sorted_levels[0]]
        for level in sorted_levels[1:]:
            if (level - clustered[-1]) / clustered[-1] > threshold:
                clustered.append(level)
        return clustered[-3:] if len(clustered) > 3 else clustered  # Return top 3
    
    return {
        "support": cluster_levels(supports),
        "resistance": cluster_levels(resistances)
    }


def calculate_volatility(data: pd.Series, period: int = 30) -> float:
    """Hitung volatilitas yang dianualisasi (annualized volatility)."""
    returns = data.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    volatility = returns.tail(period).std() * np.sqrt(252) * 100  # Annualized
    return float(volatility) if not np.isnan(volatility) else 0.0


def detect_trend(data: pd.Series, short_period: int = 10, long_period: int = 30) -> str:
    """
    Deteksi trend menggunakan crossover moving average.
    Mengembalikan: 'bullish', 'bearish', atau 'sideways'
    """
    if len(data) < long_period:
        return "sideways"
    
    sma_short = calculate_sma(data, short_period)
    sma_long = calculate_sma(data, long_period)
    
    current_short = sma_short.iloc[-1]
    current_long = sma_long.iloc[-1]
    prev_short = sma_short.iloc[-2] if len(sma_short) > 1 else current_short
    prev_long = sma_long.iloc[-2] if len(sma_long) > 1 else current_long
    
    # Check trend direction
    if current_short > current_long and prev_short <= prev_long:
        return "bullish"  # Golden cross
    elif current_short < current_long and prev_short >= prev_long:
        return "bearish"  # Death cross
    elif current_short > current_long:
        return "bullish"
    elif current_short < current_long:
        return "bearish"
    else:
        return "sideways"


def calculate_momentum(data: pd.Series, period: int = 10) -> float:
    """Calculate price momentum."""
    if len(data) < period:
        return 0.0
    momentum = ((data.iloc[-1] - data.iloc[-period]) / data.iloc[-period]) * 100
    return float(momentum) if not np.isnan(momentum) else 0.0


def generate_signal(rsi: float, macd_hist: float, bb_position: str, trend: str) -> Tuple[str, int]:
    """
    Hasilkan sinyal trading berdasarkan beberapa indikator.
    Mengembalikan: (nama_sinyal, score dari -2 sampai 2)
    """
    score = 0
    
    # RSI contribution
    if rsi > 70:
        score -= 1
    elif rsi < 30:
        score += 1
    
    # MACD contribution
    if macd_hist > 0:
        score += 1
    elif macd_hist < 0:
        score -= 1
    
    # Bollinger Band contribution
    if bb_position == "upper":
        score -= 1
    elif bb_position == "lower":
        score += 1
    
    # Trend contribution
    if trend == "bullish":
        score += 1
    elif trend == "bearish":
        score -= 1
    
    # Convert score to signal
    if score >= 3:
        return ("strong_buy", score)
    elif score >= 1:
        return ("buy", score)
    elif score <= -3:
        return ("strong_sell", score)
    elif score <= -1:
        return ("sell", score)
    else:
        return ("hold", score)


def full_technical_analysis(df: pd.DataFrame, value_column: str = "y") -> Dict[str, Any]:
    """
    Lakukan analisis teknikal lengkap pada sebuah DataFrame.
    
    Args:
        df: DataFrame dengan kolom 'ds' (tanggal) dan kolom nilai
        value_column: Nama kolom yang mengandung harga/nilai
    
    Returns:
        Dictionary berisi semua indikator teknikal dan hasil analisis
    """
    data = df[value_column].copy()
    
    # Hitung semua indikator
    sma_10 = calculate_sma(data, 10)
    sma_20 = calculate_sma(data, 20)
    sma_50 = calculate_sma(data, 50)
    ema_12 = calculate_ema(data, 12)
    ema_26 = calculate_ema(data, 26)
    
    rsi = calculate_rsi(data, 14)
    macd = calculate_macd(data)
    bollinger = calculate_bollinger_bands(data, 20)
    
    # Current values
    current_price = float(data.iloc[-1])
    current_rsi = float(rsi.iloc[-1])
    current_macd = float(macd["macd"].iloc[-1])
    current_macd_signal = float(macd["signal"].iloc[-1])
    current_macd_hist = float(macd["histogram"].iloc[-1])
    
    # Bollinger position
    bb_upper = float(bollinger["upper"].iloc[-1])
    bb_lower = float(bollinger["lower"].iloc[-1])
    bb_middle = float(bollinger["middle"].iloc[-1])
    
    if current_price > bb_upper * 0.98:
        bb_position = "upper"
    elif current_price < bb_lower * 1.02:
        bb_position = "lower"
    else:
        bb_position = "middle"
    
    # Trend detection
    trend = detect_trend(data)
    
    # Support/Resistance
    sr_levels = find_support_resistance(data)
    
    # Volatility
    volatility = calculate_volatility(data)
    
    # Momentum
    momentum = calculate_momentum(data)
    
    # Generate signal
    signal, signal_score = generate_signal(current_rsi, current_macd_hist, bb_position, trend)
    
    # MA crossover detection
    ma_crossover = None
    if len(sma_10) > 1 and len(sma_20) > 1:
        curr_10, prev_10 = float(sma_10.iloc[-1]), float(sma_10.iloc[-2])
        curr_20, prev_20 = float(sma_20.iloc[-1]), float(sma_20.iloc[-2])
        
        if curr_10 > curr_20 and prev_10 <= prev_20:
            ma_crossover = "golden_cross"
        elif curr_10 < curr_20 and prev_10 >= prev_20:
            ma_crossover = "death_cross"
    
    return {
        "current_price": current_price,
        "moving_averages": {
            "sma_10": float(sma_10.iloc[-1]),
            "sma_20": float(sma_20.iloc[-1]),
            "sma_50": float(sma_50.iloc[-1]) if len(sma_50) >= 50 else None,
            "ema_12": float(ema_12.iloc[-1]),
            "ema_26": float(ema_26.iloc[-1]),
        },
        "rsi": {
            "value": round(current_rsi, 2),
            "condition": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral"
        },
        "macd": {
            "macd_line": round(current_macd, 6),
            "signal_line": round(current_macd_signal, 6),
            "histogram": round(current_macd_hist, 6),
            "trend": "bullish" if current_macd_hist > 0 else "bearish"
        },
        "bollinger_bands": {
            "upper": round(bb_upper, 6),
            "middle": round(bb_middle, 6),
            "lower": round(bb_lower, 6),
            "position": bb_position
        },
        "support_resistance": sr_levels,
        "volatility": round(volatility, 2),
        "momentum": round(momentum, 2),
        "trend": trend,
        "ma_crossover": ma_crossover,
        "signal": signal,
        "signal_score": signal_score
    }
