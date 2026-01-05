import sys
import pathlib
import asyncio

# ensure project root is importable for tests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app
import httpx

client = TestClient(app)


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, headers=None, content=b""):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {"content-type": "application/json"}
        self._content = content

    async def json(self):
        return self._json

    async def aread(self):
        return self._content

    async def aiter_bytes(self):
        # yield content in one chunk
        yield self._content


class DummyStreamContext:
    def __init__(self, chunks):
        self._chunks = chunks
        self.status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_bytes(self):
        for c in self._chunks:
            await asyncio.sleep(0)
            yield c


def test_version(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        return DummyResponse(200, {"version": "0.5.1"})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    rv = client.get("/ollama/api/version")
    assert rv.status_code == 200
    assert rv.json()["version"] == "0.5.1"


def test_tags(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        return DummyResponse(200, {"models": []})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    rv = client.get("/ollama/api/tags")
    assert rv.status_code == 200
    assert "models" in rv.json()


def test_generate_nonstream(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        return DummyResponse(200, {"model": "llama", "response": "ok"})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    rv = client.post("/ollama/api/generate", json={"model": "llama", "prompt": "hi", "stream": False})
    assert rv.status_code == 200
    body = rv.json()
    assert body["model"] == "llama"
    assert "response" in body


def test_chat_streaming(monkeypatch):
    # simulate httpx.AsyncClient.stream returning a context manager with aiter_bytes
    def fake_stream(self, method, url, **kwargs):
        return DummyStreamContext([b'{"message":"Hel', b'lo"}\n', b'{"done": true}\n'])

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    rv = client.post("/ollama/api/chat", json={"model": "llama", "messages": [{"role":"user","content":"hi"}]})
    assert rv.status_code == 200
    # response should be streamed; TestClient collects all content
    content = rv.content
    assert b'Hello' in content or b'"message":"Hello"' in content


def test_generate_streaming(monkeypatch):
    def fake_stream(self, method, url, **kwargs):
        return DummyStreamContext([b'{"response":"Par', b'tial"}\n', b'{"done": true}\n'])

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    rv = client.post("/ollama/api/generate", json={"model": "llama", "prompt": "hi"})
    assert rv.status_code == 200
    assert b'Partial' in rv.content


def test_create_streaming(monkeypatch):
    def fake_stream(self, method, url, **kwargs):
        return DummyStreamContext([b'{"status":"reading', b'..."}\n', b'{"status":"success"}\n'])

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    rv = client.post("/ollama/api/create", json={"model": "mario", "from": "llama3.2"})
    assert rv.status_code == 200
    assert b'success' in rv.content


def test_pull_nonstream(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        return DummyResponse(200, {"status": "success"})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    rv = client.post("/ollama/api/pull", json={"model": "llama3.2", "stream": False})
    assert rv.status_code == 200
    assert rv.json()["status"] == "success"


def test_ps(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        return DummyResponse(200, {"models": []})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    rv = client.get("/ollama/api/ps")
    assert rv.status_code == 200
    assert "models" in rv.json()


def test_show(monkeypatch):
    async def fake_request(self, method, url, **kwargs):
        return DummyResponse(200, {"modelfile": "#modelfile"})

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    rv = client.post("/ollama/api/show", json={"model": "llava"})
    assert rv.status_code == 200
    assert "modelfile" in rv.json()


def test_default_ollama_url_dev(monkeypatch):
    # Ensure no explicit OLLAMA_URL set and ENV=dev
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.setenv("ENV", "dev")
    import importlib
    import app.ollama as ollama_mod
    importlib.reload(ollama_mod)
    assert ollama_mod.OLLAMA_URL == "http://llm.indonesiacore.com:11434"


def test_default_ollama_url_production(monkeypatch):
    # Ensure no explicit OLLAMA_URL set and ENV=production
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.setenv("ENV", "production")
    import importlib
    import app.ollama as ollama_mod
    importlib.reload(ollama_mod)
    assert ollama_mod.OLLAMA_URL == "http://localhost:11434"
