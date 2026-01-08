from pathlib import Path
import os
import json
from update_trending import main


def test_update_trending_from_api(monkeypatch):
    class DummyAPI:
        status_code = 200
        def json(self):
            payload = {
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
            return payload

    monkeypatch.setattr('requests.get', lambda url, headers=None, timeout=10: DummyAPI())

    out_dir = Path('trending')
    if out_dir.exists():
        for p in out_dir.iterdir():
            p.unlink()
        out_dir.rmdir()

    code = main()
    assert code == 0
    out_file = out_dir / 'stocks.json'
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data['count'] == 1
    assert data['data'][0]['symbol'] == 'AAPL'
    assert data['source'] == 'yahoo_trending_api' 


def test_update_trending_api_error(monkeypatch):
    class DummyAPIError:
        status_code = 500
    monkeypatch.setattr('requests.get', lambda url, headers=None, timeout=10: DummyAPIError())

    out_dir = Path('trending')
    if out_dir.exists():
        for p in out_dir.iterdir():
            p.unlink()
        out_dir.rmdir()

    code = main()
    assert code == 2
    out_file = out_dir / 'stocks.json'
    assert not out_file.exists()
