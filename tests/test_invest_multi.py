from fastapi.testclient import TestClient
from app.main import app
import pandas as pd

client = TestClient(app)


def test_invest_multiple_symbols():
    today = pd.Timestamp.now().date()
    r = client.get(f"/exchange/idr/invest?symbols=USDIDR=X,GC=F&amount=100000&start_date={today}&end_date={today}")
    assert r.status_code == 200
    data = r.json()
    assert data['start_date'] == str(today)
    assert data['end_date'] == str(today)
    assert data['amount_idr'] == 100000
    assert 'results' in data
    # Should contain two results, one per symbol
    syms = {res['symbol'] for res in data['results']}
    assert 'USD' in syms and 'GOLD' in syms
    # Each result should include projected_final (same-day -> equals amount)
    for res in data['results']:
        assert 'projected_final' in res
        # on same day, projected_final equals amount
        assert abs(res['projected_final'] - 100000) < 1e-6
