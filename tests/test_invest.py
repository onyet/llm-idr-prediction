from fastapi.testclient import TestClient
from app.main import app
import pandas as pd
from datetime import timedelta
import json

client = TestClient(app)


def test_invest_same_day_no_change():
    today = pd.Timestamp.now().date()
    r = client.get(f"/exchange/idr/invest?symbol=USDIDR=X&amount=100000&start_date={today}&end_date={today}")
    assert r.status_code == 200
    data = r.json()
    assert data['start_date'] == str(today)
    assert data['end_date'] == str(today)
    assert data['amount_idr'] == 100000
    # if start and end are same, projected_final should equal input amount (units*same price)
    assert abs(data['projected_final'] - 100000) < 1e-6
    assert 'ai_analysis' in data


def test_invest_validation_negative_amount():
    today = pd.Timestamp.now().date()
    r = client.get(f"/exchange/idr/invest?symbol=USDIDR=X&amount=-1&start_date={today}&end_date={today}")
    assert r.status_code == 200
    assert 'error' in r.json()


def test_invest_with_prediction(monkeypatch):
    # mock Prophet to return predictable yhat for future date
    import types
    class DummyProphet:
        def fit(self, df):
            return self
        def make_future_dataframe(self, periods=1):
            import pandas as pd
            return pd.DataFrame({'ds': pd.date_range(start=pd.Timestamp.now().date(), periods=periods+1)})
        def predict(self, future):
            import pandas as pd
            # produce a forecast with yhat increasing linearly
            ds = future['ds']
            yhat = [10000 + i*10 for i in range(len(ds))]
            return pd.DataFrame({'ds': ds, 'yhat': yhat, 'yhat_lower': [yh-5 for yh in yhat], 'yhat_upper': [yh+5 for yh in yhat]})
    monkeypatch.setattr('prophet.Prophet', DummyProphet)

    start = pd.Timestamp.now().date() + timedelta(days=1)
    end = start + timedelta(days=3)
    r = client.get(f"/exchange/idr/invest?symbol=USDIDR=X&amount=100000&start_date={start}&end_date={end}")
    assert r.status_code == 200
    data = r.json()
    assert data['is_prediction'] is True
    assert data['projected_final'] is not None
    assert 'ai_analysis' in data


def test_invest_gold_alias():
    today = pd.Timestamp.now().date()
    r1 = client.get(f"/exchange/idr/invest?symbol=GC=F&amount=100000&start_date={today}&end_date={today}")
    r2 = client.get(f"/exchange/idr/invest?symbol=GOLD&amount=100000&start_date={today}&end_date={today}")
    assert r1.status_code == 200 and r2.status_code == 200
    d1 = r1.json()
    d2 = r2.json()
    # GOLD should normalize to GC=F and produce identical results for same date
    assert d1['start_price_idr'] == d2['start_price_idr']
    assert d1['end_price_idr'] == d2['end_price_idr']
    assert d1['series'][0]['idr_value'] == d2['series'][0]['idr_value']
