from fastapi import APIRouter, Query, Request
import json
import os
from yahooquery import search, Ticker
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

router = APIRouter(prefix="/exchange/idr")

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

