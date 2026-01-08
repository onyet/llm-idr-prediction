from fastapi import APIRouter, Query
import json
import os
from yahooquery import search, Ticker
import yfinance as yf
import pandas as pd

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
            raise ValueError(f"No data for {symbol}")
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
async def search_market(query: str = Query(..., description="Search query for stocks only")):
    try:
        results = search(query)
        # Filter to return only equity stocks
        quotes = results.get("quotes", [])
        filtered_quotes = [q for q in quotes if q.get("quoteType") == "EQUITY"]
        return {"results": filtered_quotes}
    except Exception as e:
        return {"error": str(e)}

@router.get("/market/detail")
async def get_ticker_detail(ticker: str = Query(..., description="Ticker symbol to get detailed information")):
    try:
        stock = Ticker(ticker)
        info = stock.summary_detail
        return {"detail": info.get(ticker, {})}
    except Exception as e:
        return {"error": str(e)}

@router.get("/market/history")
async def get_ticker_history(ticker: str = Query(..., description="Ticker symbol"), period: str = Query("1d", description="Time period: 1d, 5d, 1mo, 6mo, 1y, 5y")):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        # Reset index to include date as column, then convert to list of dicts
        hist = hist.reset_index()
        return {"history": hist.to_dict('records')}
    except Exception as e:
        return {"error": str(e)}

@router.get("/exchange")
async def get_exchange_rates(symbols: str = Query("", description="Comma-separated list of additional symbols, e.g., AAPL,BBCA.JK (default GOLD, USD, SAR always included)")):
    # Load currency data
    with open(CURRENCY_FILE, "r") as f:
        currency_data = json.load(f)

    currencies = {c['code']: c for c in currency_data['currencies'] if c.get('yfinance')}
    
    # Always include defaults
    default_symbols = ["GC=F", "USDIDR=X", "SAR=X"]
    additional_symbols = [s.strip() for s in symbols.split(',') if s.strip()]
    symbol_list = default_symbols + [s for s in additional_symbols if s not in default_symbols]

    # Preload local data for defaults
    usd_idr_rate = load_latest_price('USDIDR_X.json')
    gold_usd = load_latest_price('GC_F.json')
    sar_usd = load_latest_price('SAR_X.json')

    results = []
    for symbol in symbol_list:
        try:
            if symbol == "GC=F":
                # GOLD per gram in IDR
                gold_per_gram_usd = gold_usd / 31.1034768
                idr_value = gold_per_gram_usd * usd_idr_rate
                results.append({
                    "symbol": clean_symbol_name(symbol),
                    "idr_value": idr_value,
                    "currency": "IDR",
                    "details": {
                        "unit": "per gram",
                        "source_symbol": "GC=F",
                        "price_usd_per_ounce": gold_usd,
                        "price_usd_per_gram": gold_per_gram_usd
                    }
                })
            elif symbol == "USDIDR=X":
                idr_value = usd_idr_rate
                results.append({
                    "symbol": clean_symbol_name(symbol),
                    "idr_value": idr_value,
                    "currency": "IDR",
                    "details": {
                        "unit": "IDR per USD",
                        "source_symbol": "USDIDR=X"
                    }
                })
            elif symbol == "SAR=X":
                idr_value = usd_idr_rate / sar_usd
                results.append({
                    "symbol": clean_symbol_name(symbol),
                    "idr_value": idr_value,
                    "currency": "IDR",
                    "details": {
                        "unit": "IDR per SAR",
                        "source_symbol": "SAR=X",
                        "price_usd_per_sar": sar_usd
                    }
                })
            elif symbol in currencies:
                # From currency list - update data first
                curr = currencies[symbol]
                yf_symbol = curr['yfinance']['symbol']
                unit = curr['yfinance']['unit']
                # save unit/currency meta into the cached file if missing
                price = update_symbol_data(yf_symbol, meta={"unit": unit, "source": "currency_list"})
                if 'IDR' in unit:
                    idr_value = price
                elif unit.startswith('USD/'):
                    usd_per_unit = price
                    idr_value = usd_idr_rate / usd_per_unit
                else:
                    idr_value = price * usd_idr_rate
                results.append({
                    "symbol": clean_symbol_name(symbol),
                    "idr_value": idr_value,
                    "currency": "IDR",
                    "details": {
                        "unit": unit,
                        "name": curr.get('name'),
                        "code": curr.get('code'),
                        "original_yf_symbol": yf_symbol
                    }
                })
            else:
                # Stock or other - update data first
                price = update_symbol_data(symbol)
                currency = get_currency_for_ticker(symbol)

                # Determine conversion rate from stock currency -> IDR
                conversion_info = {"from": currency, "to": "IDR", "rate": None, "source_symbol": None}
                if currency == 'IDR':
                    idr_value = price
                    conversion_info["rate"] = 1
                    conversion_info["source_symbol"] = "IDR"
                elif currency in currencies:
                    # Use currency list mapping to compute currency -> IDR
                    curr_meta = currencies[currency]
                    curr_yf = curr_meta['yfinance']['symbol']
                    curr_unit = curr_meta['yfinance']['unit']
                    # ensure currency data cached and get latest
                    curr_price = update_symbol_data(curr_yf, meta={"unit": curr_unit, "source": "currency_list"})
                    if 'IDR' in curr_unit:
                        idr_per_unit = curr_price
                    elif curr_unit.startswith('USD/'):
                        # e.g., USD/AED -> AED/IDR = USDIDR / (USD/AED)
                        idr_per_unit = usd_idr_rate / curr_price
                    else:
                        idr_per_unit = curr_price * usd_idr_rate
                    idr_value = price * idr_per_unit
                    conversion_info.update({"rate": idr_per_unit, "source_symbol": curr_yf})
                else:
                    # Fallback assume USD
                    idr_per_unit = usd_idr_rate
                    idr_value = price * idr_per_unit
                    conversion_info.update({"rate": idr_per_unit, "source_symbol": "USDIDR=X", "note": "assumed USD"})

                # Try to fetch company details
                try:
                    info = yf.Ticker(symbol).info
                except:
                    info = {}

                results.append({
                    "symbol": clean_symbol_name(symbol),
                    "idr_value": idr_value,
                    "currency": currency,
                    "details": {
                        "last_price": price,
                        "name": info.get('shortName') or info.get('longName') or clean_symbol_name(symbol),
                        "longName": info.get('longName'),
                        "sector": info.get('sector'),
                        "industry": info.get('industry'),
                        "exchange": info.get('exchange') or info.get('fullExchangeName'),
                        "original_yf_symbol": symbol,
                        "conversion": conversion_info
                    }
                })
        except Exception as e:
            results.append({
                "symbol": symbol,
                "error": str(e)
            })
    
    return {"exchanges": results}

