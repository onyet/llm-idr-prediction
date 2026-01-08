from fastapi import APIRouter, Query, Request
import json
import os
from yahooquery import search, Ticker
import requests, time, copy
import yfinance as yf
import pandas as pd
from .i18n import get_lang_from_request, set_current_lang, t, tr, get_current_lang

# Caches for Prophet models and predictions to avoid repeated fitting
MODEL_CACHE = {}
PREDICTION_CACHE = {}
PREDICTION_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "predictions")
os.makedirs(PREDICTION_DIR, exist_ok=True)

def get_currency_for_ticker(ticker):
    try:
        stock = Ticker(ticker)
        info = stock.summary_detail.get(ticker, {})
        return info.get('currency', 'USD')
    except:
        return 'USD'

def load_latest_price(filepath):
    try:
        with open(os.path.join(os.path.dirname(__file__), "..", "data", filepath), 'r') as f:
            data = json.load(f)
        return data['data'][-1]['Close']
    except:
        return 1  # fallback

def update_symbol_data(symbol, meta=None):
    # Sanitize filename
    safe_symbol = symbol.replace('=', '_').replace('/', '_').replace('.', '_')
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", f"{safe_symbol}.json")

    if not os.path.exists(filepath):
        # Fetch 5y
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="5y")
        if data.empty:
            # raise localized error using current language context
            raise ValueError(tr('no_data_for_symbol', symbol=symbol))
        data = data.reset_index()
        result = {
            "symbol": symbol,
            "period": "5y",
            "meta": meta or {},
            "data": data.to_dict('records')
        }
        with open(filepath, 'w') as f:
            json.dump(result, f, default=str)
    else:
        # Check existing and possibly merge meta
        with open(filepath, 'r') as f:
            existing = json.load(f)
        if meta:
            existing_meta = existing.get('meta', {})
            updated = False
            for k, v in (meta.items() if isinstance(meta, dict) else []):
                if existing_meta.get(k) != v:
                    existing_meta[k] = v
                    updated = True
            if updated:
                existing['meta'] = existing_meta
                with open(filepath, 'w') as f:
                    json.dump(existing, f, default=str)
        # Check if latest is today
        if existing.get('data'):
            last_date = existing['data'][-1]['Date']
            today = str(pd.Timestamp.now().date())
            if last_date.split(' ')[0] != today:
                # Fetch 1d and append
                ticker = yf.Ticker(symbol)
                new_data = ticker.history(period="1d")
                if not new_data.empty:
                    new_data = new_data.reset_index()
                    new_entry = new_data.to_dict('records')[0]
                    existing['data'].append(new_entry)
                    with open(filepath, 'w') as f:
                        json.dump(existing, f, default=str)
    
    # Return latest Close
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['data'][-1]['Close']

def clean_symbol_name(symbol):
    if symbol == "GC=F":
        return "GOLD"
    elif symbol == "USDIDR=X":
        return "USD"
    elif symbol == "SAR=X":
        return "SAR"
    else:
        # Remove leading ^
        if symbol.startswith('^'):
            symbol = symbol[1:]
        # Remove =X or =F
        if '=X' in symbol:
            symbol = symbol.replace('=X', '')
        elif '=F' in symbol:
            symbol = symbol.replace('=F', '')
        # Remove .suffix
        if '.' in symbol:
            symbol = symbol.split('.')[0]
        return symbol


def get_price_on_date(symbol, req_date):
    """Return (price, price_date_str, is_prediction, yhat_lower, yhat_upper)
    req_date is a date object (datetime.date)
    """
    # ensure cached file exists and up-to-date
    safe_symbol = symbol.replace('=', '_').replace('/', '_').replace('.', '_')
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", f"{safe_symbol}.json")
    if not os.path.exists(filepath):
        # create with 5y
        update_symbol_data(symbol)
    with open(filepath, 'r') as f:
        data = json.load(f)
    records = data.get('data', [])
    if not records:
        raise ValueError(tr('no_historical_data_for_symbol', symbol=symbol))
    df = pd.DataFrame(records)
    # parse Date column robustly (handle tz-aware and mixed formats)
    if 'Date' in df.columns:
        # try parse with utc to handle timezone-aware timestamps, coerce errors to NaT
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
        # convert to timezone-naive if tz-aware
        try:
            df['Date'] = df['Date'].dt.tz_convert(None)
        except Exception:
            try:
                df['Date'] = df['Date'].dt.tz_localize(None)
            except Exception:
                pass
        # fallback: if not datetimelike, try without utc
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # drop invalid dates
        df = df.dropna(subset=['Date']).reset_index(drop=True)
        if df.empty:
            raise ValueError(tr('no_valid_date_entries'))
    else:
        raise ValueError(tr('no_date_column'))
    # normalize to date
    df['date_only'] = df['Date'].dt.date

    # determine last available date
    last_date = df['date_only'].max()

    today = pd.Timestamp.now().date()

    # If requested date is today, try to refresh data first (do not predict for today)
    if req_date == today and last_date < today:
        try:
            # attempt one-day update and re-read file
            update_symbol_data(symbol)
            with open(filepath, 'r') as f:
                data = json.load(f)
            records = data.get('data', [])
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
            try:
                df['Date'] = df['Date'].dt.tz_convert(None)
            except Exception:
                try:
                    df['Date'] = df['Date'].dt.tz_localize(None)
                except Exception:
                    pass
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).reset_index(drop=True)
            df['date_only'] = df['Date'].dt.date
            last_date = df['date_only'].max()
            # if data for today now exists, return it
            if last_date >= today:
                le = df[df['date_only'] <= req_date]
                if not le.empty:
                    row = le.iloc[-1]
                    return row['Close'], str(row['date_only']), False, None, None
        except Exception:
            pass

    # if requested date is after last available, predict (future dates)
    if req_date > last_date:
        # check persisted predictions first
        pred_file = os.path.join(PREDICTION_DIR, f"{safe_symbol}_pred.json")
        req_key = req_date.isoformat()
        if os.path.exists(pred_file):
            try:
                with open(pred_file, 'r') as pf:
                    preds = json.load(pf)
                if req_key in preds:
                    p = preds[req_key]
                    return float(p['yhat']), req_key, True, float(p.get('yhat_lower', p['yhat'])), float(p.get('yhat_upper', p['yhat']))
            except Exception:
                pass
        # check in-memory cache
        cache_key = (safe_symbol, req_key)
        if cache_key in PREDICTION_CACHE:
            yhat, lower, upper = PREDICTION_CACHE[cache_key]
            return yhat, req_key, True, lower, upper
        # fit or reuse model
        try:
            from prophet import Prophet
            # reuse fitted model if available
            if safe_symbol in MODEL_CACHE:
                m = MODEL_CACHE[safe_symbol]
            else:
                dfp = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                m = Prophet()
                m.fit(dfp)
                MODEL_CACHE[safe_symbol] = m
            days = (req_date - last_date).days
            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            forecast['ds_date'] = pd.to_datetime(forecast['ds']).dt.date
            rowf = forecast[forecast['ds_date'] == req_date]
            if not rowf.empty:
                yhat = float(rowf.iloc[0]['yhat'])
                yhat_lower = float(rowf.iloc[0].get('yhat_lower', yhat))
                yhat_upper = float(rowf.iloc[0].get('yhat_upper', yhat))
                # cache in-memory
                PREDICTION_CACHE[cache_key] = (yhat, yhat_lower, yhat_upper)
                # persist
                try:
                    preds = {}
                    if os.path.exists(pred_file):
                        with open(pred_file, 'r') as pf:
                            preds = json.load(pf)
                    preds[req_key] = {"yhat": yhat, "yhat_lower": yhat_lower, "yhat_upper": yhat_upper}
                    with open(pred_file, 'w') as pf:
                        json.dump(preds, pf)
                except Exception:
                    pass
                return yhat, req_key, True, yhat_lower, yhat_upper
        except Exception:
            # fallback to last available
            row = df.iloc[-1]
            return row['Close'], str(row['date_only']), False, None, None

    # find rows with date <= req_date and return the closest
    le = df[df['date_only'] <= req_date]
    if not le.empty:
        row = le.iloc[-1]
        return row['Close'], str(row['date_only']), False, None, None

    # req_date is before earliest data, choose earliest
    first_row = df.iloc[0]
    return first_row['Close'], str(first_row['date_only']), False, None, None

# Caches and persistence for investment projections and AI analyses
INVEST_CACHE = {}
INVEST_PERSIST_DIR = PREDICTION_DIR

# Trending tickers cache (short TTL to avoid repeated external calls)
TREND_CACHE = {}
TREND_CACHE_TTL = 60  # seconds


import statistics
from .llm_agent import get_llm_analyzer, AnalysisContext, MockLLMAgent, GroqAgent


def _safe_amount_key(amount: float) -> int:
    return int(round(amount))


def _safe_symbol_file(symbol: str) -> str:
    return symbol.replace('=', '_').replace('/', '_').replace('.', '_')


def build_series_for_symbol(symbol: str, date_list, currencies, usd_map, usd_pred_map, usd_date_map, lang: str):
    series = []
    try:
        if symbol == "GC=F":
            for d in date_list:
                gold_p, gold_pd, gold_pred, gold_lower, gold_upper = get_price_on_date('GC=F', d)
                usd_p = usd_map[d]
                gold_per_gram_usd = gold_p / 31.1034768
                idr_value = gold_per_gram_usd * usd_p
                is_pred = gold_pred or usd_pred_map[d]['is_pred']
                series.append({
                    "date": str(d),
                    "idr_value": idr_value,
                    "is_prediction": is_pred,
                    "price_date": gold_pd,
                    "details": {"unit": "per gram", "price_usd_per_ounce": gold_p, "price_usd_per_gram": gold_per_gram_usd, "prediction": {"lower": gold_lower, "upper": gold_upper} if gold_pred else None}
                })
            return series, "IDR"
        elif symbol == "USDIDR=X":
            for d in date_list:
                usd_p = usd_map[d]
                is_pred = usd_pred_map[d]['is_pred']
                series.append({"date": str(d), "idr_value": usd_p, "is_prediction": is_pred, "price_date": usd_date_map.get(d), "details": {"unit": "IDR per USD", "prediction": {"lower": usd_pred_map[d]['lower'], "upper": usd_pred_map[d]['upper']} if is_pred else None}})
            return series, "IDR"
        elif symbol == "SAR=X":
            for d in date_list:
                sar_p, sar_pd, sar_pred, sar_lower, sar_upper = get_price_on_date('SAR_X', d)
                usd_p = usd_map[d]
                # avoid division by zero
                idr_value = usd_p / sar_p if sar_p else None
                is_pred = sar_pred or usd_pred_map[d]['is_pred']
                series.append({"date": str(d), "idr_value": idr_value, "is_prediction": is_pred, "price_date": sar_pd, "details": {"unit": "IDR per SAR", "price_usd_per_sar": sar_p, "prediction": {"lower": sar_lower, "upper": sar_upper} if sar_pred else None}})
            return series, "IDR"
        elif symbol in currencies:
            curr = currencies[symbol]
            yf_symbol = curr['yfinance']['symbol']
            unit = curr['yfinance']['unit']
            for d in date_list:
                price, pd_str, is_pred, lower, upper = get_price_on_date(yf_symbol, d)
                usd_p = usd_map[d]
                if 'IDR' in unit:
                    idr_value = price
                elif unit.startswith('USD/'):
                    idr_value = usd_p / price if price else None
                else:
                    idr_value = price * usd_p
                is_pred = is_pred or usd_pred_map[d]['is_pred']
                series.append({"date": str(d), "idr_value": idr_value, "is_prediction": is_pred, "price_date": pd_str, "details": {"unit": unit, "name": curr.get('name'), "code": curr.get('code'), "original_yf_symbol": yf_symbol, "prediction": {"lower": lower, "upper": upper} if is_pred else None}})
            return series, "IDR"
        else:
            # treat as stock or other ticker
            currency = get_currency_for_ticker(symbol)
            for d in date_list:
                price, pd_str, is_price_pred, p_lower, p_upper = get_price_on_date(symbol, d)
                if currency == 'IDR':
                    idr_value = price
                    conversion_meta = {"rate": 1, "source_symbol": None, "is_prediction": False}
                elif currency in currencies:
                    curr_meta = currencies[currency]
                    curr_yf = curr_meta['yfinance']['symbol']
                    curr_unit = curr_meta['yfinance']['unit']
                    curr_price, curr_pd, curr_pred, curr_lower, curr_upper = get_price_on_date(curr_yf, d)
                    if 'IDR' in curr_unit:
                        idr_per_unit = curr_price
                    elif curr_unit.startswith('USD/'):
                        idr_per_unit = usd_map[d] / curr_price if curr_price else None
                    else:
                        idr_per_unit = curr_price * usd_map[d] if curr_price else None
                    idr_value = price * idr_per_unit if price and idr_per_unit else None
                    conversion_meta = {"rate": idr_per_unit, "source_symbol": curr_yf, "is_prediction": curr_pred, "prediction": {"lower": curr_lower, "upper": curr_upper} if curr_pred else None}
                else:
                    idr_per_unit = usd_map[d]
                    idr_value = price * idr_per_unit if price else None
                    conversion_meta = {"rate": idr_per_unit, "source_symbol": "USDIDR=X", "note": "assumed USD"}
                try:
                    info = Ticker(symbol).info
                except Exception:
                    info = {}
                is_pred = is_price_pred or conversion_meta.get('is_prediction', False)
                series.append({"date": str(d), "idr_value": idr_value, "is_prediction": is_pred, "price_date": pd_str, "details": {"last_price": price, "name": info.get('shortName') or info.get('longName') or clean_symbol_name(symbol), "longName": info.get('longName'), "sector": info.get('sector'), "industry": info.get('industry'), "exchange": info.get('exchange') or info.get('fullExchangeName'), "original_yf_symbol": symbol, "conversion": conversion_meta, "prediction": {"lower": p_lower, "upper": p_upper} if is_price_pred else None}})
            return series, currency
    except Exception as e:
        raw = str(e)
        localized = t('symbol_error', lang=lang, symbol=clean_symbol_name(symbol), err=raw)
        return None, {"error": {"message": localized, "raw": raw}}


async def generate_ai_analysis(series, symbol: str = "IDR", lang: str = "en"):
    # Build simple technical summary
    values = [s.get('idr_value') for s in series if s.get('idr_value') is not None]
    if not values or len(values) < 2:
        return {"text": t('ai.analysis_insufficient', lang, module='llm_agent'), "confidence": 50}

    start = values[0]
    end = values[-1]
    change_pct = ((end - start) / start) * 100 if start else 0
    vol_pct = 0
    try:
        vol_pct = statistics.pstdev(values) / statistics.mean(values) * 100
    except Exception:
        vol_pct = 0

    # estimate RSI-like value (very rough)
    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [c for c in changes if c > 0]
    losses = [-c for c in changes if c < 0]
    avg_gain = sum(gains) / len(gains) if gains else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    rsi = 0
    try:
        rs = avg_gain / avg_loss if avg_loss else float('inf')
        rsi = 100 - (100 / (1 + rs)) if avg_loss else 100
    except Exception:
        rsi = 50

    trend = 'bullish' if change_pct > 0.1 else 'bearish' if change_pct < -0.1 else 'sideways'
    signal = 'buy' if change_pct > 0.5 else 'sell' if change_pct < -0.5 else 'hold'

    tech = {
        'trend': trend,
        'signal': signal,
        'rsi': {'value': round(rsi, 2)},
        'volatility': round(vol_pct, 2)
    }

    hist_summary = t('summary.analysis', lang, module='llm_agent', pair=symbol, target_date=series[-1].get('date'), trend=trend, signal=signal, current_price=format(end, '.6f'))

    ctx = AnalysisContext(
        pair=symbol,
        current_price=end,
        technical_indicators=tech,
        fundamental_data={},
        historical_summary=hist_summary,
        target_date=series[-1].get('date'),
        analysis_type='prediction' if any(s.get('is_prediction') for s in series) else 'historical',
        language=lang
    )

    analyzer = get_llm_analyzer()

    # Prefer GroqAgent if available
    try:
        for agent in analyzer.agents:
            if isinstance(agent, GroqAgent) and await agent.is_available():
                res = await agent.analyze(ctx)
                # Normalize
                text = res.get('summary') or ' '.join(res.get('insights', [])) or res.get('recommendation', '')
                conf = res.get('confidence', res.get('confidence', 0.5))
                if isinstance(conf, float) and conf <= 1:
                    conf = int(conf * 100)
                return {"text": text, "confidence": int(conf)}
        # fallback to first available agent
        res = await analyzer.analyze(ctx)
        text = res.get('summary') or ' '.join(res.get('insights', [])) or res.get('recommendation', '')
        conf = res.get('confidence', 0.5)
        if isinstance(conf, float) and conf <= 1:
            conf = int(conf * 100)
        return {"text": text, "confidence": int(conf)}
    except Exception:
        # Final fallback: craft translated message similar to previous heuristic
        direction = t('ai.direction_strengthen', lang, module='llm_agent') if change_pct < 0 else t('ai.direction_weaken', lang, module='llm_agent') if change_pct > 0 else t('ai.direction_stable', lang, module='llm_agent')
        confidence = int(max(40, min(95, 90 - vol_pct)))
        text = t('ai.analysis_summary', lang, module='llm_agent', direction=direction, confidence=confidence)
        return {"text": text, "confidence": confidence}


router = APIRouter(prefix="/exchange/idr")

@router.get("/invest")
async def invest_projection(symbol: str = Query(None, description="(Deprecated) Target symbol e.g., AAPL, GC=F, USDIDR=X"), symbols: str = Query(None, description="Comma-separated list of target symbols, e.g., AAPL,GC=F,USDIDR=X"), amount: float = Query(..., description="Investment amount in IDR"), start_date: str = Query(None, description="Start date YYYY-MM-DD"), end_date: str = Query(None, description="End date YYYY-MM-DD"), max_days: int = Query(90, description="Maximum allowed days in range"), request: Request = None):
    lang = get_lang_from_request(request)
    set_current_lang(lang)

    if amount is None or amount <= 0:
        return {"error": t('invalid_amount', lang=lang)}

    # Determine symbol list: prefer `symbols` if provided, otherwise support legacy `symbol`
    symbol_input = symbols if symbols else symbol
    if not symbol_input:
        return {"error": t('symbol_error', lang=lang, symbol='(missing)')}
    symbol_list = [s.strip() for s in symbol_input.split(',') if s.strip()]
    if not symbol_list:
        return {"error": t('symbol_error', lang=lang, symbol='(missing)')}

    if start_date:
        try:
            start = pd.to_datetime(start_date).date()
        except Exception:
            return {"error": t('invalid_start_date_format', lang=lang)}
        if end_date:
            try:
                end = pd.to_datetime(end_date).date()
            except Exception:
                return {"error": t('invalid_end_date_format', lang=lang)}
        else:
            end = start
    else:
        # default window: today
        start = pd.Timestamp.now().date()
        end = start

    if end < start:
        return {"error": t('end_before_start', lang=lang)}

    total_days = (end - start).days + 1
    if total_days > max_days:
        return {"error": t('range_too_large', lang=lang, total_days=total_days, max_days=max_days)}

    date_list = [d.date() for d in pd.date_range(start=start, end=end, freq='D')]

    with open(CURRENCY_FILE, "r") as f:
        currency_data = json.load(f)

    currencies = {c['code']: c for c in currency_data['currencies'] if c.get('yfinance')}

    # precompute USD map
    usd_map = {}
    usd_pred_map = {}
    usd_date_map = {}
    for d in date_list:
        p, pd_str, is_pred, lower, upper = get_price_on_date('USDIDR=X', d)
        usd_map[d] = p
        usd_pred_map[d] = {'is_pred': is_pred, 'lower': lower, 'upper': upper}
        usd_date_map[d] = pd_str

    results_list = []

    for sym in symbol_list:
        cache_key = (sym, start.isoformat(), end.isoformat(), _safe_amount_key(amount))
        if cache_key in INVEST_CACHE:
            results_list.append(INVEST_CACHE[cache_key])
            continue

        series, currency = build_series_for_symbol(sym, date_list, currencies, usd_map, usd_pred_map, usd_date_map, lang)
        if series is None:
            # error object returned
            err_obj = currency
            results_list.append({"symbol": clean_symbol_name(sym), "error": err_obj})
            continue

        # find start and end prices (fallback to nearest non-null)
        start_price = None
        end_price = None
        start_pred_band = None
        end_pred_band = None
        for s in series:
            if s['date'] == str(start) and s.get('idr_value') is not None:
                start_price = s['idr_value']
                start_pred_band = s['details'].get('prediction') if s.get('details') else None
            if s['date'] == str(end) and s.get('idr_value') is not None:
                end_price = s['idr_value']
                end_pred_band = s['details'].get('prediction') if s.get('details') else None
        if start_price is None:
            # fallback first non-null
            for s in series:
                if s.get('idr_value') is not None:
                    start_price = s['idr_value']
                    start_pred_band = s['details'].get('prediction') if s.get('details') else None
                    break
        if end_price is None:
            for s in reversed(series):
                if s.get('idr_value') is not None:
                    end_price = s['idr_value']
                    end_pred_band = s['details'].get('prediction') if s.get('details') else None
                    break

        if start_price in (None, 0):
            results_list.append({"symbol": clean_symbol_name(sym), "error": {"message": t('no_valid_price_for_start', lang=lang, symbol=clean_symbol_name(sym))}})
            continue

        units_bought = amount / start_price
        projected_final = units_bought * end_price if end_price is not None else None

        projected_lower = None
        projected_upper = None
        # propagate uncertainty when prediction bands exist
        try:
            if start_pred_band and end_pred_band:
                s_low = start_pred_band.get('lower', start_price)
                s_up = start_pred_band.get('upper', start_price)
                e_low = end_pred_band.get('lower', end_price)
                e_up = end_pred_band.get('upper', end_price)
                # conservative bounds
                projected_lower = (amount / s_up) * e_low if s_up and e_low else None
                projected_upper = (amount / s_low) * e_up if s_low and e_up else None
        except Exception:
            projected_lower = projected_upper = None

        ai_analysis = await generate_ai_analysis(series, symbol=sym, lang=lang)

        per_res = {
            "symbol": clean_symbol_name(sym),
            "start_date": str(start),
            "end_date": str(end),
            "amount_idr": amount,
            "start_price_idr": start_price,
            "end_price_idr": end_price,
            "units_bought": units_bought,
            "projected_final": projected_final,
            "projected_final_lower": projected_lower,
            "projected_final_upper": projected_upper,
            "is_prediction": any([s.get('is_prediction', False) for s in series]),
            "series": series,
            "ai_analysis": ai_analysis
        }

        INVEST_CACHE[cache_key] = per_res
        # persist result for reproducibility
        try:
            safe = _safe_symbol_file(sym)
            fname = f"invest_{safe}_{start.isoformat()}_{end.isoformat()}_{_safe_amount_key(amount)}.json"
            with open(os.path.join(INVEST_PERSIST_DIR, fname), 'w') as pf:
                json.dump(per_res, pf, default=str)
        except Exception:
            pass

        results_list.append(per_res)

    # If caller requested a single symbol, keep backward-compatible response
    if len(symbol_list) == 1:
        return results_list[0]

    final_result = {
        "start_date": str(start),
        "end_date": str(end),
        "amount_idr": amount,
        "results": results_list
    }

    return final_result

# Path to the currency data file
CURRENCY_FILE = os.path.join(os.path.dirname(__file__), "..", "currency", "BIXYFINANCE.json")

@router.get("/currencies")
async def get_currencies(search: str = Query(None, description="Search by name, code, symbol, or country")):
    # Load currency data
    with open(CURRENCY_FILE, "r") as f:
        data = json.load(f)

    currencies = data.get("currencies", [])

    # Filter currencies where yfinance is not null
    filtered_currencies = [c for c in currencies if c.get("yfinance") is not None]

    # If search parameter is provided, filter further
    if search:
        search_lower = search.lower()
        filtered_currencies = [
            c for c in filtered_currencies
            if (
                search_lower in c.get("name", "").lower() or
                search_lower in c.get("code", "").lower() or
                search_lower in c.get("symbol", "").lower() or
                search_lower in c.get("country", "").lower()
            )
        ]
    
    # Return the filtered list
    return {"currencies": filtered_currencies}

@router.get("/market/search")
async def search_market(request: Request, query: str = Query(..., description="Search query for stocks only")):
    lang = get_lang_from_request(request)
    try:
        results = search(query)
        # Filter to return only equity stocks
        quotes = results.get("quotes", [])
        filtered_quotes = [q for q in quotes if q.get("quoteType") == "EQUITY"]
        return {"results": filtered_quotes}
    except Exception as e:
        return {"error": t('market_search_error', lang=lang, err=str(e))}

@router.get("/market/detail")
async def get_ticker_detail(request: Request, ticker: str = Query(..., description="Ticker symbol to get detailed information")):
    lang = get_lang_from_request(request)
    try:
        stock = Ticker(ticker)
        info = stock.summary_detail
        return {"detail": info.get(ticker, {})}
    except Exception as e:
        return {"error": t('market_detail_error', lang=lang, err=str(e))}

@router.get("/market/history")
async def get_ticker_history(request: Request, ticker: str = Query(..., description="Ticker symbol"), period: str = Query("1d", description="Time period: 1d, 5d, 1mo, 6mo, 1y, 5y")):
    lang = get_lang_from_request(request)
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        # Reset index to include date as column, then convert to list of dicts
        hist = hist.reset_index()
        return {"history": hist.to_dict('records')}
    except Exception as e:
        return {"error": t('market_history_error', lang=lang, err=str(e))}

@router.get("/exchange")
async def get_exchange_rates(symbols: str = Query("", description="Comma-separated list of additional symbols, e.g., AAPL,BBCA.JK (default GOLD, USD, SAR always included)"), date: str = Query(None, description="Date YYYY-MM-DD, default today"), start_date: str = Query(None, description="Start date YYYY-MM-DD for range queries"), end_date: str = Query(None, description="End date YYYY-MM-DD for range queries"), max_days: int = Query(90, description="Maximum allowed days in range"), request: Request = None):
    """Existing exchange endpoint (unchanged)"""

    # Determine date range
    today = pd.Timestamp.now().date()
    # Determine language from request headers and set context
    lang = get_lang_from_request(request)
    set_current_lang(lang)

    if start_date:
        try:
            start = pd.to_datetime(start_date).date()
        except Exception:
            return {"error": t('invalid_start_date_format', lang=lang)}
        if end_date:
            try:
                end = pd.to_datetime(end_date).date()
            except Exception:
                return {"error": t('invalid_end_date_format', lang=lang)}
        else:
            end = start
    elif date:
        try:
            start = pd.to_datetime(date).date()
            end = start
        except Exception:
            return {"error": t('invalid_start_date_format', lang=lang)}
    else:
        start = today
        end = today

    if end < start:
        return {"error": t('end_before_start', lang=lang)}

    total_days = (end - start).days + 1
    if total_days > max_days:
        return {"error": t('range_too_large', lang=lang, total_days=total_days, max_days=max_days)}

    # Build list of dates
    date_list = [d.date() for d in pd.date_range(start=start, end=end, freq='D')]

    # Load currency data
    with open(CURRENCY_FILE, "r") as f:
        currency_data = json.load(f)

    currencies = {c['code']: c for c in currency_data['currencies'] if c.get('yfinance')}
    
    # Always include defaults
    default_symbols = ["GC=F", "USDIDR=X", "SAR=X"]
    additional_symbols = [s.strip() for s in symbols.split(',') if s.strip()]
    symbol_list = default_symbols + [s for s in additional_symbols if s not in default_symbols]

    # Precompute USDIDR series for range
    usd_map = {}
    usd_pred_map = {}
    usd_date_map = {}
    for d in date_list:
        p, pd_str, is_pred, lower, upper = get_price_on_date('USDIDR=X', d)
        usd_map[d] = p
        usd_pred_map[d] = {'is_pred': is_pred, 'lower': lower, 'upper': upper}
        usd_date_map[d] = pd_str

    results = []
    any_prediction = False

    for symbol in symbol_list:
        series = []
        try:
            if symbol == "GC=F":
                for d in date_list:
                    gold_p, gold_pd, gold_pred, gold_lower, gold_upper = get_price_on_date('GC=F', d)
                    usd_p = usd_map[d]
                    gold_per_gram_usd = gold_p / 31.1034768
                    idr_value = gold_per_gram_usd * usd_p
                    is_pred = gold_pred or usd_pred_map[d]['is_pred']
                    any_prediction = any_prediction or is_pred
                    series.append({
                        "date": str(d),
                        "idr_value": idr_value,
                        "is_prediction": is_pred,
                        "price_date": gold_pd,
                        "details": {"unit": "per gram", "price_usd_per_ounce": gold_p, "price_usd_per_gram": gold_per_gram_usd, "prediction": {"lower": gold_lower, "upper": gold_upper} if gold_pred else None}
                    })
                results.append({"symbol": clean_symbol_name(symbol), "series": series, "currency": "IDR"})
            elif symbol == "USDIDR=X":
                for d in date_list:
                    usd_p = usd_map[d]
                    is_pred = usd_pred_map[d]['is_pred']
                    any_prediction = any_prediction or is_pred
                    series.append({"date": str(d), "idr_value": usd_p, "is_prediction": is_pred, "price_date": usd_date_map.get(d), "details": {"unit": "IDR per USD", "prediction": {"lower": usd_pred_map[d]['lower'], "upper": usd_pred_map[d]['upper']} if is_pred else None}})
                results.append({"symbol": clean_symbol_name(symbol), "series": series, "currency": "IDR"})
            elif symbol == "SAR=X":
                for d in date_list:
                    sar_p, sar_pd, sar_pred, sar_lower, sar_upper = get_price_on_date('SAR_X', d)
                    usd_p = usd_map[d]
                    idr_value = usd_p / sar_p
                    is_pred = sar_pred or usd_pred_map[d]['is_pred']
                    any_prediction = any_prediction or is_pred
                    series.append({"date": str(d), "idr_value": idr_value, "is_prediction": is_pred, "price_date": sar_pd, "details": {"unit": "IDR per SAR", "price_usd_per_sar": sar_p, "prediction": {"lower": sar_lower, "upper": sar_upper} if sar_pred else None}})
                results.append({"symbol": clean_symbol_name(symbol), "series": series, "currency": "IDR"})
            elif symbol in currencies:
                curr = currencies[symbol]
                yf_symbol = curr['yfinance']['symbol']
                unit = curr['yfinance']['unit']
                for d in date_list:
                    price, pd_str, is_pred, lower, upper = get_price_on_date(yf_symbol, d)
                    usd_p = usd_map[d]
                    if 'IDR' in unit:
                        idr_value = price
                    elif unit.startswith('USD/'):
                        idr_value = usd_p / price
                    else:
                        idr_value = price * usd_p
                    any_prediction = any_prediction or is_pred
                    series.append({"date": str(d), "idr_value": idr_value, "is_prediction": is_pred, "price_date": pd_str, "details": {"unit": unit, "name": curr.get('name'), "code": curr.get('code'), "original_yf_symbol": yf_symbol, "prediction": {"lower": lower, "upper": upper} if is_pred else None}})
                results.append({"symbol": clean_symbol_name(symbol), "series": series, "currency": "IDR"})
            else:
                # Stock or other
                currency = get_currency_for_ticker(symbol)
                for d in date_list:
                    price, pd_str, is_price_pred, p_lower, p_upper = get_price_on_date(symbol, d)
                    if currency == 'IDR':
                        idr_value = price
                    elif currency in currencies:
                        curr_meta = currencies[currency]
                        curr_yf = curr_meta['yfinance']['symbol']
                        curr_unit = curr_meta['yfinance']['unit']
                        curr_price, curr_pd, curr_pred, curr_lower, curr_upper = get_price_on_date(curr_yf, d)
                        if 'IDR' in curr_unit:
                            idr_per_unit = curr_price
                        elif curr_unit.startswith('USD/'):
                            idr_per_unit = usd_map[d] / curr_price
                        else:
                            idr_per_unit = curr_price * usd_map[d]
                        idr_value = price * idr_per_unit
                        conversion_meta = {"rate": idr_per_unit, "source_symbol": curr_yf, "is_prediction": curr_pred, "prediction": {"lower": curr_lower, "upper": curr_upper} if curr_pred else None}
                    else:
                        idr_per_unit = usd_map[d]
                        idr_value = price * idr_per_unit
                        conversion_meta = {"rate": idr_per_unit, "source_symbol": "USDIDR=X", "note": "assumed USD"}
                    is_pred = is_price_pred or conversion_meta.get('is_prediction', False)
                    any_prediction = any_prediction or is_pred
                    # try fetch static company info once (not per date) - done outside
                    try:
                        info = yf.Ticker(symbol).info
                    except:
                        info = {}
                    series.append({"date": str(d), "idr_value": idr_value, "is_prediction": is_pred, "price_date": pd_str, "details": {"last_price": price, "name": info.get('shortName') or info.get('longName') or clean_symbol_name(symbol), "longName": info.get('longName'), "sector": info.get('sector'), "industry": info.get('industry'), "exchange": info.get('exchange') or info.get('fullExchangeName'), "original_yf_symbol": symbol, "conversion": conversion_meta, "prediction": {"lower": p_lower, "upper": p_upper} if is_price_pred else None}})
                results.append({"symbol": clean_symbol_name(symbol), "series": series, "currency": currency})
        except Exception as e:
            raw = str(e)
            localized = t('symbol_error', lang=lang, symbol=clean_symbol_name(symbol), err=raw)
            results.append({"symbol": clean_symbol_name(symbol), "error": {"message": localized, "raw": raw}})

    # If the request was for a single date, flatten the series for easier consumption
    if start == end:
        flat = []
        for r in results:
            try:
                s = r.get('series', [{}])[0]
            except Exception:
                s = {}
            entry = {
                "symbol": r.get('symbol'),
                "date": s.get('date', str(start)),
                "idr_value": s.get('idr_value'),
                "is_prediction": s.get('is_prediction', False),
                "price_date": s.get('price_date'),
                "details": s.get('details'),
                "currency": r.get('currency')
            }
            # If the original result included an error object, include a localized/translated error
            if r.get('error'):
                # if it's already an object with message/raw, pass through; otherwise convert
                if isinstance(r.get('error'), dict):
                    entry['error'] = r['error']
                else:
                    raw = str(r.get('error'))
                    entry['error'] = {"message": t('symbol_error', lang=lang, symbol=entry.get('symbol'), err=raw), "raw": raw}
                # ensure details are null when there's an error
                entry['details'] = None
                entry['idr_value'] = None
            flat.append(entry)
        return {"date": str(start), "is_prediction": any_prediction, "exchanges": flat}

    return {"date_range": {"start": str(start), "end": str(end)}, "is_prediction": any_prediction, "exchanges": results}


@router.get('/get_trend')
async def get_trend(request: Request, region: str = Query('US', description='Region code, e.g., US, GB, HK'), count: int = Query(10, description='Maximum number of trending tickers to return')):
    lang = get_lang_from_request(request)
    cache_key = region.upper()
    now = time.time()

    cached = TREND_CACHE.get(cache_key)
    if cached and now - cached['ts'] < TREND_CACHE_TTL:
        quotes = copy.deepcopy(cached['data'])
        return {"region": region, "count": min(count, len(quotes)), "results": quotes[:count]}

    url = f"https://query1.finance.yahoo.com/v1/finance/trending/{region}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return {"error": t('trend_fetch_error', lang=lang, err=f"HTTP {resp.status_code}")}
        payload = resp.json()
        quotes = []
        # Keep only EQUITY quoteType
        for r in payload.get('finance', {}).get('result', [])[:1]:
            for q in r.get('quotes', []):
                if q.get('quoteType') != 'EQUITY':
                    continue
                quotes.append({
                    'symbol': q.get('symbol'),
                    'shortName': q.get('shortName') or q.get('longName'),
                    'quoteType': q.get('quoteType'),
                    'exchange': q.get('exchDisp') or q.get('exchange'),
                })
        # cache the full equity results
        TREND_CACHE[cache_key] = {'ts': now, 'data': quotes}
        return {"region": region, "count": min(count, len(quotes)), "results": quotes[:count]}
    except Exception as e:
        return {"error": t('trend_fetch_error', lang=lang, err=str(e))} 

