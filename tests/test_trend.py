from fastapi.testclient import TestClient
from app.main import app
from pathlib import Path
import json

client = TestClient(app)


def test_get_trend_success(monkeypatch, tmp_path):
    # create a trending/stocks.json file as produced by the scraper (EQUITY-only)
    tdir = tmp_path / 'trending'
    tdir.mkdir()
    payload = {
        'fetched_at': '2026-01-09T00:00:00Z',
        'source': 'yahoo_trending_api',
        'count': 2,
        'data': [
            {"symbol": "AAPL", "shortName": "Apple Inc.", "quoteType": "EQUITY", "exchDisp": "NASDAQ"},
            {"symbol": "TSLA", "shortName": "Tesla, Inc.", "quoteType": "EQUITY", "exchDisp": "NASDAQ"}
        ]
    }
    f = tdir / 'stocks.json'
    f.write_text(json.dumps(payload))

    # create repo trending file for the app to read
    repo_trend = Path('trending')
    if not repo_trend.exists():
        repo_trend.mkdir()
    (repo_trend / 'stocks.json').write_text(json.dumps(payload))

    r = client.get('/exchange/idr/get_trend?region=US&count=2')
    assert r.status_code == 200
    data = r.json()
    assert data['region'] == 'US'
    assert data['count'] == 2
    assert len(data['results']) == 2
    assert data['results'][0]['symbol'] == 'AAPL'
    assert data['results'][1]['symbol'] == 'TSLA'


def test_get_trend_caching(monkeypatch, tmp_path):
    from app import exchange_idr
    exchange_idr.TREND_CACHE.clear()

    # create repo trending file
    payload = {
        'fetched_at': '2026-01-09T00:00:00Z',
        'source': 'yahoo_trending_api',
        'count': 1,
        'data': [
            {"symbol": "AAPL", "shortName": "Apple Inc.", "quoteType": "EQUITY", "exchDisp": "NASDAQ"}
        ]
    }
    repo_trend = Path('trending')
    if not repo_trend.exists():
        repo_trend.mkdir()
    (repo_trend / 'stocks.json').write_text(json.dumps(payload))

    calls = {'n': 0}
    import builtins
    orig_open = builtins.open
    def counting_open(file, *args, **kwargs):
        if str(file).endswith('trending/stocks.json'):
            calls['n'] += 1
        return orig_open(file, *args, **kwargs)
    monkeypatch.setattr('builtins.open', counting_open)

    r1 = client.get('/exchange/idr/get_trend?region=US')
    assert r1.status_code == 200
    r2 = client.get('/exchange/idr/get_trend?region=US')
    assert r2.status_code == 200
    # file should be read only once due to caching
    assert calls['n'] == 1


def test_get_trend_failure(monkeypatch):
    from app import exchange_idr
    exchange_idr.TREND_CACHE.clear()
    # ensure no file exists
    repo_trend = Path('trending')
    fp = repo_trend / 'stocks.json'
    if fp.exists():
        fp.unlink()

    r = client.get('/exchange/idr/get_trend?region=US')
    assert r.status_code == 200
    data = r.json()
    assert 'error' in data
