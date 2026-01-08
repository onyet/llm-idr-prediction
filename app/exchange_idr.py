from fastapi import APIRouter, Query
import json
import os
from yahooquery import search, Ticker
import yfinance as yf

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