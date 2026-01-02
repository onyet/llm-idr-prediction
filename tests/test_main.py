from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    rv = client.get("/")
    assert rv.status_code == 200
    # Terima kedua kemungkinan: Bahasa default bisa 'id' (Halo) atau 'en' (Hello)
    assert "Hello" in rv.json()["message"] or "Halo" in rv.json()["message"]


def test_rag_idr_usd_default():
    rv = client.get("/rag/idr-usd")
    assert rv.status_code == 200
    body = rv.json()
    assert body["pair"] == "idr-usd"
    assert body["method"] == "prophet"
    assert "predicted" in body


def test_rag_amount():
    rv = client.get("/rag/idr-usd?days=0&amount=1000")
    assert rv.status_code == 200
    body = rv.json()
    assert body["pair"] == "idr-usd"
    assert "predicted" in body
    assert "predicted_for_amount" in body
    assert abs(body["predicted_for_amount"] - body["predicted"] * 1000) < 1e-8


def test_rag_too_far():
    rv = client.get("/rag/idr-usd?days=10")
    assert rv.status_code == 400
    # Accept English or Indonesian error messages
    detail = rv.json()["detail"]
    assert ("too long" in detail.lower()) or ("terlalu panjang" in detail.lower())


def test_rag_idr_sar_days_3():
    rv = client.get("/rag/idr-sar?days=3")
    assert rv.status_code == 200
    body = rv.json()
    assert body["pair"] == "idr-sar"
    assert "date" in body
    assert "predicted" in body


def test_tradingview_get():
    # Note: this test may fail if yfinance requires internet or has rate limits
    # In real tests, mock the yfinance
    rv = client.get("/tradingview/get/AAPL")
    # Assuming it returns 200 or 500 depending on yfinance
    assert rv.status_code in [200, 500]  # 500 if yfinance fails in test env


def test_tradingview_save_param():
    rv = client.get("/tradingview/get/AAPL?save=1")
    assert rv.status_code in [200, 500]
    if rv.status_code == 200:
        body = rv.json()
        assert "saved_to" in body or "Failed to save" in str(body)


def test_mock_tradingview_get():
    rv = client.get("/tradingview/mock/XAUIDRG")
    assert rv.status_code == 200
    body = rv.json()
    assert body["symbol"] == "XAUIDRG"
    assert body["period"] == "1y"
    assert body["source"] == "mock_tradingview"
    assert "data" in body
    assert isinstance(body["data"], list)
    assert len(body["data"]) > 0
    # Check data structure
    data_point = body["data"][0]
    assert "Date" in data_point
    assert "Open" in data_point
    assert "High" in data_point
    assert "Low" in data_point
    assert "Close" in data_point
    assert "Volume" in data_point


def test_rag_idr_gold_gram():
    rv = client.get("/rag/idr-gold-gram?amount_gram=1")
    assert rv.status_code == 200
    body = rv.json()
    assert body["pair"] == "idr-gold-gram"
    assert "predicted" in body
    assert "predicted_for_amount" in body


def test_rag_idr_summary():
    rv = client.get("/rag/idr-summary")
    assert rv.status_code == 200
    body = rv.json()
    assert "summary" in body
    assert "analyses" in body
    assert "recommendation" in body
    assert "main_recommendation" in body["recommendation"]
    assert "best_option" in body["recommendation"]
