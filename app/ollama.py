from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import os
import httpx

router = APIRouter(prefix="/ollama/api", tags=["ollama"])

OLLAMA_URL = os.environ.get("OLLAMA_URL")
# Default behavior: in development (default ENV), use the internal dev host
# `http://llm.indonesiacore.com:11434`. For production, set ENV=production and
# default will be localhost (useful when running Ollama on the same host).
if not OLLAMA_URL:
    env = os.environ.get("ENV", "dev").lower()
    if env == "production":
        OLLAMA_URL = "http://localhost:11434"
    else:
        OLLAMA_URL = "http://llm.indonesiacore.com:11434"

API_PREFIX = "/api"


async def _forward_request(method: str, path: str, request: Request):
    url = f"{OLLAMA_URL}{API_PREFIX}/{path}"
    # copy query params
    params = dict(request.query_params)

    headers = {}
    # forward content-type if present
    if request.headers.get("content-type"):
        headers["content-type"] = request.headers.get("content-type")

    # read body bytes (if any)
    body = await request.body()

    # For streaming endpoints, the client may use stream API
    # We decide streaming if 'stream' not explicitly false
    stream_flag = params.get("stream")
    if stream_flag is None:
        # try to inspect json body for stream field
        try:
            json_body = await request.json()
            if isinstance(json_body, dict) and "stream" in json_body:
                stream_flag = json_body.get("stream")
        except Exception:
            json_body = None

    # Default for GET/HEAD: non-streaming
    if stream_flag is None and method.upper() in ("GET", "HEAD"):
        stream_flag = False

    # If stream is False (False or 'false' or '0'), do normal request now
    if stream_flag in [False, "false", "False", "0", 0]:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.request(method, url, params=params, content=body, headers=headers)
            return (False, resp)

    # Else return a descriptor so the caller can open a client.stream (to keep client alive)
    return (True, method, url, params, headers, body)


@router.get("/version")
async def version(request: Request):
    """GET /ollama/api/version -> proxies GET /api/version"""
    forward = await _forward_request("GET", "version", request)
    if forward[0] is True:
        _, method, url, params, headers, body = forward

        async def iter_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(method, url, params=params, content=body, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(iter_stream(), media_type="application/json")

    # Non-streaming
    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


# Explicit odd endpoints for documentation/clarity and stable behavior
@router.post("/generate")
async def generate(request: Request):
    """POST /ollama/api/generate"""
    forward = await _forward_request("POST", "generate", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward

        async def iter_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(method, url, params=params, content=body, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(iter_stream(), media_type="application/json")

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.post("/chat")
async def chat(request: Request):
    """POST /ollama/api/chat"""
    forward = await _forward_request("POST", "chat", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward

        async def iter_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(method, url, params=params, content=body, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(iter_stream(), media_type="application/json")

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.post("/embed")
async def embed(request: Request):
    """POST /ollama/api/embed (alias)"""
    forward = await _forward_request("POST", "embed", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(method, url, params=params, content=body, headers=headers) as r:
                content = await r.aread()
                try:
                    import json

                    data = json.loads(content)
                    return JSONResponse(content=data, status_code=r.status_code)
                except Exception:
                    return StreamingResponse(iter([content]), status_code=r.status_code, media_type=r.headers.get("content-type", "application/octet-stream"))

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.post("/embeddings")
async def embeddings(request: Request):
    """POST /ollama/api/embeddings (legacy alias)"""
    forward = await _forward_request("POST", "embeddings", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward
        # Prefer non-streaming for embeddings
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.request(method, url, params=params, content=body, headers=headers)
            try:
                return JSONResponse(content=await resp.json(), status_code=resp.status_code)
            except Exception:
                return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.post("/create")
async def create(request: Request):
    """POST /ollama/api/create (streaming)"""
    forward = await _forward_request("POST", "create", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward

        async def iter_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(method, url, params=params, content=body, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(iter_stream(), media_type="application/json")

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.post("/pull")
async def pull(request: Request):
    """POST /ollama/api/pull (streaming)"""
    forward = await _forward_request("POST", "pull", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward

        async def iter_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(method, url, params=params, content=body, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(iter_stream(), media_type="application/json")

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.post("/push")
async def push(request: Request):
    """POST /ollama/api/push (streaming)"""
    forward = await _forward_request("POST", "push", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward

        async def iter_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(method, url, params=params, content=body, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(iter_stream(), media_type="application/json")

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.post("/show")
async def show(request: Request):
    """POST /ollama/api/show"""
    forward = await _forward_request("POST", "show", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward
        # Endpoint documented as non-streaming; perform a regular request here
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.request(method, url, params=params, content=body, headers=headers)
            try:
                return JSONResponse(content=await resp.json(), status_code=resp.status_code)
            except Exception:
                return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.post("/copy")
async def copy_model(request: Request):
    """POST /ollama/api/copy"""
    forward = await _forward_request("POST", "copy", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward
        # Prefer non-streaming for copy; use a regular request
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.request(method, url, params=params, content=body, headers=headers)
            try:
                return JSONResponse(content=await resp.json(), status_code=resp.status_code)
            except Exception:
                return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.delete("/delete")
async def delete_model(request: Request):
    """DELETE /ollama/api/delete"""
    forward = await _forward_request("DELETE", "delete", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward
        # Prefer non-streaming for delete operations
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.request(method, url, params=params, content=body, headers=headers)
            try:
                return JSONResponse(content=await resp.json(), status_code=resp.status_code)
            except Exception:
                return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.head("/blobs/{digest}")
async def blob_head(digest: str, request: Request):
    """HEAD /ollama/api/blobs/:digest"""
    forward = await _forward_request("HEAD", f"blobs/{digest}", request)
    _, resp = forward
    return JSONResponse(content={}, status_code=resp.status_code)


@router.post("/blobs/{digest}")
async def blob_push(digest: str, request: Request):
    """POST /ollama/api/blobs/:digest (push)"""
    forward = await _forward_request("POST", f"blobs/{digest}", request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.request(method, url, params=params, content=body, headers=headers)
            try:
                return JSONResponse(content=await resp.json(), status_code=resp.status_code)
            except Exception:
                return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))

    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.get("/tags")
async def tags(request: Request):
    """GET /ollama/api/tags"""
    forward = await _forward_request("GET", "tags", request)
    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


@router.get("/ps")
async def ps(request: Request):
    """GET /ollama/api/ps"""
    forward = await _forward_request("GET", "ps", request)
    _, resp = forward
    try:
        return JSONResponse(content=await resp.json(), status_code=resp.status_code)
    except Exception:
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))


# Fallback catch-all (keeps previous behavior)
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(path: str, request: Request):
    """Catch-all proxy for Ollama API under /ollama/api/*"""
    forward = await _forward_request(request.method, path, request)

    if forward[0] is True:
        _, method, url, params, headers, body = forward

        async def iter_stream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(method, url, params=params, content=body, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        media_type = "application/json"
        return StreamingResponse(iter_stream(), media_type=media_type, status_code=200)

    # Non-streaming
    _, resp = forward
    try:
        data = await resp.json()
        return JSONResponse(content=data, status_code=resp.status_code)
    except Exception:
        # Fallback to streaming bytes
        return StreamingResponse(resp.aiter_bytes(), status_code=resp.status_code, media_type=resp.headers.get("content-type", "application/octet-stream"))
