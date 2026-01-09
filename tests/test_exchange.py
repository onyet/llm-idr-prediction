import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app
from app.exchange_idr import get_price_on_date

client = TestClient(app)


def test_get_price_on_date_historical():
    # Historical date that exists or is before latest - should not be a prediction
    req_date = pd.to_datetime("2025-02-02").date()
    price, price_date, is_pred, lower, upper = get_price_on_date("USDIDR=X", req_date)
    assert isinstance(price, (int, float)) or hasattr(price, '__float__')
    assert isinstance(price_date, str)
    # price_date should be <= requested date
    assert pd.to_datetime(price_date).date() <= req_date
    assert is_pred is False


def test_exchange_endpoint_with_date():
    rv = client.get("/exchange/idr/exchange?symbols=BBCA.JK,XMLF.JK,BND&date=2025-02-02")
    assert rv.status_code == 200
    body = rv.json()
    assert "date" in body and body["date"] == "2025-02-02"
    assert "exchanges" in body and isinstance(body["exchanges"], list)
    # Default symbols GOLD, USD, SAR should be present
    symbols = [e["symbol"] for e in body["exchanges"]]
    assert "GOLD" in symbols
    assert "USD" in symbols
    assert any(s in symbols for s in ("SAR", "SAR=X"))
    # Provided symbols cleaned should be present
    assert "BBCA" in symbols
    assert "XMLF" in symbols
    assert "BND" in symbols
    # Each exchange entry has price_date or is_prediction and details
    for e in body["exchanges"]:
        assert "details" in e
        assert "idr_value" in e
        assert ("price_date" in e) or (e.get("is_prediction") is True)


def test_get_price_on_date_prediction(monkeypatch):
    # Mock prophet to make prediction deterministic and fast
    import sys, types
    fake = types.SimpleNamespace()
    class FakeProphet:
        def __init__(self):
            self.last = None
        def fit(self, df):
            self.last = pd.to_datetime(df['ds']).max()
        def make_future_dataframe(self, periods):
            # include last + periods days
            start = pd.to_datetime(self.last)
            dates = pd.date_range(start=start, periods=periods+1, freq='D')
            return pd.DataFrame({'ds': dates})
        def predict(self, future):
            ds = pd.to_datetime(future['ds'])
            yhat = [1000.0 + i for i in range(len(ds))]
            yhat_lower = [v - 5 for v in yhat]
            yhat_upper = [v + 5 for v in yhat]
            return pd.DataFrame({'ds': ds, 'yhat': yhat, 'yhat_lower': yhat_lower, 'yhat_upper': yhat_upper})
    fake.Prophet = FakeProphet
    monkeypatch.setitem(sys.modules, 'prophet', fake)

    # pick a date after last available in USDIDR data
    import app.exchange_idr as ex
    # get last date from cached file
    import os
    path = os.path.join(os.path.dirname(ex.__file__), '..', 'data', 'USDIDR_X.json')
    import json
    with open(path, 'r') as f:
        d = json.load(f)
    last_date = pd.to_datetime(d['data'][-1]['Date']).date()
    req_date = last_date + pd.Timedelta(days=3)

    price, price_date, is_pred, lower, upper = ex.get_price_on_date('USDIDR=X', req_date)
    assert is_pred is True
    assert isinstance(price, float) or isinstance(price, (int,))
    assert lower is not None and upper is not None


def test_exchange_future_date_with_mocked_prophet(monkeypatch):
    # Mock prophet same as above and test endpoint
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
            yhat = [2000.0 + i for i in range(len(ds))]
            yhat_lower = [v - 10 for v in yhat]
            yhat_upper = [v + 10 for v in yhat]
            return pd.DataFrame({'ds': ds, 'yhat': yhat, 'yhat_lower': yhat_lower, 'yhat_upper': yhat_upper})
    fake.Prophet = FakeProphet
    monkeypatch.setitem(sys.modules, 'prophet', fake)

    # choose a future date relative to USDIDR last date
    import app.exchange_idr as ex
    import os, json
    path = os.path.join(os.path.dirname(ex.__file__), '..', 'data', 'USDIDR_X.json')
    with open(path, 'r') as f:
        d = json.load(f)
    last_date = pd.to_datetime(d['data'][-1]['Date']).date()
    req_date = (last_date + pd.Timedelta(days=5)).isoformat()

    rv = client.get(f"/exchange/idr/exchange?date={req_date}")
    assert rv.status_code == 200
    body = rv.json()
    assert body['date'] == req_date
    assert body['is_prediction'] is True or any(e.get('is_prediction') for e in body['exchanges'])


def test_today_without_update_returns_last_non_prediction(monkeypatch):
    # Simulate update failing and last available date < today
    import app.exchange_idr as ex
    import os, json
    path = os.path.join(os.path.dirname(ex.__file__), '..', 'data', 'USDIDR_X.json')
    with open(path, 'r') as f:
        d = json.load(f)
    last_row = d['data'][-1]
    last_date = pd.to_datetime(last_row['Date']).date()
    today = pd.Timestamp.now().date()
    assert last_date <= today

    # make update_symbol_data raise to simulate failure
    monkeypatch.setattr('app.exchange_idr.update_symbol_data', lambda symbol, meta=None: (_ for _ in ()).throw(Exception('update failed')))

    price, price_date, is_pred, lower, upper = ex.get_price_on_date('USDIDR=X', today)
    assert is_pred is False
    assert float(price) == float(last_row['Close'])


def test_prediction_sanity_check(monkeypatch):
    # Mock Prophet to return an absurdly large prediction; get_price_on_date should ignore it
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
            # return a huge value far from last close
            yhat = [1e6 for _ in range(len(ds))]
            yhat_lower = [v - 10 for v in yhat]
            yhat_upper = [v + 10 for v in yhat]
            return pd.DataFrame({'ds': ds, 'yhat': yhat, 'yhat_lower': yhat_lower, 'yhat_upper': yhat_upper})
    fake.Prophet = FakeProphet
    monkeypatch.setitem(sys.modules, 'prophet', fake)

    import app.exchange_idr as ex
    import os, json
    path = os.path.join(os.path.dirname(ex.__file__), '..', 'data', 'USDIDR_X.json')
    with open(path, 'r') as f:
        d = json.load(f)
    last_date = pd.to_datetime(d['data'][-1]['Date']).date()
    req_date = last_date + pd.Timedelta(days=3)

    # Ensure no persisted prediction for this date exists (so the mock Prophet is used)
    pred_file = os.path.join(os.path.dirname(ex.__file__), '..', 'data', 'predictions', 'USDIDR_X_pred.json')
    if os.path.exists(pred_file):
        try:
            with open(pred_file, 'r') as pf:
                preds = json.load(pf)
            key = req_date.isoformat()
            if key in preds:
                del preds[key]
                with open(pred_file, 'w') as pf:
                    json.dump(preds, pf)
        except Exception:
            os.remove(pred_file)

    # clear in-memory prediction/model cache to avoid stale values from other tests
    ex.PREDICTION_CACHE.clear()
    ex.MODEL_CACHE.pop('USDIDR_X', None)

    price, price_date, is_pred, lower, upper = ex.get_price_on_date('USDIDR=X', req_date)
    # prediction should be ignored as it's too far from last_close
    assert is_pred is False
    assert float(price) == float(d['data'][-1]['Close'])
