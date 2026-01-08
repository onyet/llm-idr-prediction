from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_get_trend_success(monkeypatch):
    class DummyResp:
        status_code = 200
        def json(self):
            return {
                "finance": {
                    "result": [
                        {"quotes": [
                            {"symbol": "AAPL", "shortName": "Apple Inc.", "quoteType": "EQUITY", "exchDisp": "NASDAQ"},
                            {"symbol": "USO", "shortName": "Oil ETF", "quoteType": "ETF", "exchDisp": "NYSE"},
                            {"symbol": "TSLA", "shortName": "Tesla, Inc.", "quoteType": "EQUITY", "exchDisp": "NASDAQ"}
                        ]}
                    ]
                }
            }
    monkeypatch.setattr('requests.get', lambda url, timeout=5: DummyResp())

    r = client.get('/exchange/idr/get_trend?region=US&count=2')
    assert r.status_code == 200
    data = r.json()
    assert data['region'] == 'US'
    assert data['count'] == 2
    assert len(data['results']) == 2
    # ensure non-EQUITY (USO) was filtered out
    assert data['results'][0]['symbol'] == 'AAPL'
    assert data['results'][1]['symbol'] == 'TSLA' 


def test_get_trend_caching(monkeypatch):
    from app import exchange_idr
    exchange_idr.TREND_CACHE.clear()
    calls = {'n': 0}
    class DummyResp:
        status_code = 200
        def json(self):
            return {
                "finance": {
                    "result": [
                        {
                            "quotes": [
                                {"symbol": "AAPL", "shortName": "Apple Inc.", "quoteType": "EQUITY", "exchDisp": "NASDAQ"}
                            ]
                        }
                    ]
                }
            }

    def get_stub(url, timeout=5):
        calls['n'] += 1
        return DummyResp()

    monkeypatch.setattr('requests.get', get_stub)
    r1 = client.get('/exchange/idr/get_trend?region=US')
    assert r1.status_code == 200
    r2 = client.get('/exchange/idr/get_trend?region=US')
    assert r2.status_code == 200
    # requests.get should be called only once due to caching
    assert calls['n'] == 1


def test_get_trend_failure(monkeypatch):
    from app import exchange_idr
    exchange_idr.TREND_CACHE.clear()
    class BadResp:
        status_code = 500
        def json(self):
            return {}
    monkeypatch.setattr('requests.get', lambda url, timeout=5: BadResp())

    r = client.get('/exchange/idr/get_trend?region=US')
    assert r.status_code == 200
    data = r.json()
    assert 'error' in data
