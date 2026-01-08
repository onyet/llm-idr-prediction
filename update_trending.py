#!/usr/bin/env python3
"""Fetch trending stocks from Yahoo Finance using the official trending JSON API and save to trending/stocks.json

This script fetches the trending JSON endpoint
https://query1.finance.yahoo.com/v1/finance/trending/US (with query parameters)
and persists only EQUITY quotes into a small JSON file for downstream use.
"""

import json
import os
import sys
from datetime import datetime

import requests

ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "trending")
OUT_FILE = os.path.join(OUT_DIR, "stocks.json")
URL_JSON = "https://query1.finance.yahoo.com/v1/finance/trending/US?count=25&fields=logoUrl,longName,shortName,regularMarketChange,regularMarketChangePercent,regularMarketPrice,ticker,symbol,longName,sparkline,shortName,forward_dividend_yield,beta,total_revenue_market_currency.annual,relative_volume_1day,netincomeismarketcurrency.annual,cash_on_hand_quarterly_market_currency,net_income_per_employee_annual_market_currency,relative_strength_index_14day,total_revenue_per_employee_annual_market_currency,fulltimeemployees.annual,all_time_high.value,all_time_high.datetime,regularMarketPrice,eodprice,regularMarketChange,regularMarketChangePercent,regularMarketOpen,fiftytwowkpercentchange,regularMarketVolume,averageDailyVolume3Month,marketCap,trailingPE,fiftyTwoWeekChangePercent,fiftyTwoWeekRange,regularMarketOpen,dividendyield,peratio.lasttwelvemonths&format=true&useQuotes=true&quoteType=equity&lang=en-US&region=US"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; LLM-Exchange/1.0)"} 





def fetch_trending_via_api():
    resp = requests.get(URL_JSON, headers=HEADERS, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"API fetch failed: HTTP {resp.status_code}")
    payload = resp.json()
    quotes = []
    for r in payload.get('finance', {}).get('result', [])[:1]:
        for q in r.get('quotes', []):
            quotes.append(q)
    return quotes


def _raw_value(q, key):
    v = q.get(key)
    if isinstance(v, dict):
        return v.get('raw') if 'raw' in v else v.get('fmt')
    return v


def normalize_quotes(quotes):
    equities = []
    for q in quotes:
        logo = q.get('logoUrl') or q.get('companyLogoUrl') or q.get('logo')
        equities.append({
            'symbol': q.get('symbol'),
            'shortName': q.get('shortName') or q.get('longName'),
            'longName': q.get('longName'),
            'quoteType': q.get('quoteType'),
            'exchange': q.get('exchDisp') or q.get('exchange') or q.get('fullExchangeName'),
            'currency': q.get('currency'),
            'price': _raw_value(q, 'regularMarketPrice'),
            'marketCap': _raw_value(q, 'marketCap'),
            'trailingPE': _raw_value(q, 'trailingPE'),
            'trendingScore': _raw_value(q, 'trendingScore'),
            'logoUrl': logo
        })
    return equities


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    now = datetime.utcnow().isoformat() + 'Z'

    # Fetch via JSON API only
    try:
        quotes = fetch_trending_via_api()
        source = 'api'
    except Exception as e:
        print(f"Failed to fetch trending stocks: {e}")
        return 2

    equities = normalize_quotes(quotes)
    data = {
        'fetched_at': now,
        'source': f'yahoo_trending_{source}',
        'count': len(equities),
        'data': equities
    }

    with open(OUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(equities)} equities to {OUT_FILE}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
