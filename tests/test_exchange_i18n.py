import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_single_date_error_message_in_english():
    headers = {"accept-language": "en-US,en;q=0.9"}
    # FOOBAR likely has no data and will cause an error
    rv = client.get("/exchange/idr/exchange?symbols=FOOBAR&date=2025-02-02", headers=headers)
    assert rv.status_code == 200
    body = rv.json()
    assert "date" in body
    ex = next((e for e in body['exchanges'] if e['symbol'] == 'FOOBAR'), None)
    assert ex is not None
    assert 'error' in ex and isinstance(ex['error'], dict)
    # English message contains 'Error processing symbol'
    assert 'Error processing symbol' in ex['error']['message']


def test_single_date_error_message_in_indonesian():
    headers = {"accept-language": "id"}
    rv = client.get("/exchange/idr/exchange?symbols=FOOBAR&date=2025-02-02", headers=headers)
    assert rv.status_code == 200
    body = rv.json()
    ex = next((e for e in body['exchanges'] if e['symbol'] == 'FOOBAR'), None)
    assert ex is not None
    assert 'error' in ex and isinstance(ex['error'], dict)
    # Indonesian message should contain 'Terjadi kesalahan memproses simbol'
    assert 'Terjadi kesalahan memproses simbol' in ex['error']['message']
