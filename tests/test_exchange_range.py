import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app import exchange_idr

client = TestClient(app)


def test_exchange_range_historical_and_pred(monkeypatch):
    # Mock Prophet to speed up predictions (simple deterministic model)
    import sys, types
    fake = types.SimpleNamespace()
    class FakeProphet:
        def __init__(self):
            self.last = None
        def fit(self, df):
            self.last = pd.to_datetime(df['ds']).max()
        def make_future_dataframe(self, periods):
            start = pd.to_datetime(self.last)
            dates = pd.date_range(start=start, periods=periods+1, freq='D')
            return pd.DataFrame({'ds': dates})
        def predict(self, future):
            ds = pd.to_datetime(future['ds'])
            yhat = [3000.0 + i for i in range(len(ds))]
            yhat_lower = [v - 10 for v in yhat]
            yhat_upper = [v + 10 for v in yhat]
            return pd.DataFrame({'ds': ds, 'yhat': yhat, 'yhat_lower': yhat_lower, 'yhat_upper': yhat_upper})
    fake.Prophet = FakeProphet
    monkeypatch.setitem(sys.modules, 'prophet', fake)

    # choose a range that includes historical and future days
    import app.exchange_idr as ex
    import os, json
    path = os.path.join(os.path.dirname(ex.__file__), '..', 'data', 'USDIDR_X.json')
    with open(path, 'r') as f:
        d = json.load(f)
    last_date = pd.to_datetime(d['data'][-1]['Date']).date()
    start = (last_date - pd.Timedelta(days=2)).isoformat()
    end = (last_date + pd.Timedelta(days=3)).isoformat()

    rv = client.get(f"/exchange/idr/exchange?start_date={start}&end_date={end}")
    assert rv.status_code == 200
    body = rv.json()
    assert 'date_range' in body
    assert body['date_range']['start'] == start
    assert body['date_range']['end'] == end
    assert 'exchanges' in body
    # ensure series length equals days count
    days = (pd.to_datetime(end).date() - pd.to_datetime(start).date()).days + 1
    for e in body['exchanges']:
        assert len(e['series']) == days
        # ensure at least one predicted day exists (since end extends beyond last)
    assert body['is_prediction'] is True or any(any(day['is_prediction'] for day in e['series']) for e in body['exchanges'])
