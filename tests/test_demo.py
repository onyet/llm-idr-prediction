from fastapi.testclient import TestClient
from app.main import app
from pathlib import Path

client = TestClient(app)


def test_demo_serves_html():
    r = client.get("/demo/1")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    content = r.text
    assert "<title>AI Finance Dashboard</title>" in content
    # confirm some visible section exists
    assert "Market Overview" in content
