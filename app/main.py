from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .mini import models as admin_models
from . import rag, tradingview, analyze
from . import ollama
from . import exchange_idr
from .i18n import get_lang_from_request, set_current_lang, t

app = FastAPI(title="LLM Exchange API", version="0.1.0")


class LanguageMiddleware(BaseHTTPMiddleware):
    """Middleware untuk mengatur bahasa berdasarkan header Accept-Language."""
    async def dispatch(self, request: Request, call_next):
        lang = get_lang_from_request(request)
        set_current_lang(lang)
        response = await call_next(request)
        return response


# Tambahkan middleware offline terlebih dahulu (admin turnoff)

class OfflineMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # allow admin paths to be called even when offline
        try:
            if admin_models.is_offline():
                if not request.url.path.startswith(admin_models.ADMIN_BASE_PATH):
                    return JSONResponse(status_code=503, content={"detail": "Service is turned off"})
        except Exception:
            # If admin module is missing or has issues, fail fast by refusing requests
            return JSONResponse(status_code=500, content={"detail": "Admin module error"})
        return await call_next(request)

app.add_middleware(OfflineMiddleware)

# Tambahkan middleware bahasa terlebih dahulu
app.add_middleware(LanguageMiddleware)

# Mount admin router early; missing admin module will cause import-time failure
app.include_router(admin_models.router)

app.include_router(rag.router)
app.include_router(tradingview.router)
app.include_router(analyze.router)
app.include_router(ollama.router)
app.include_router(exchange_idr.router)

# Izinkan CORS untuk pengujian cepat di browser (sesuaikan origin di produksi)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root(request: Request):
    lang = get_lang_from_request(request)
    return {"message": t("hello_message", lang)}


@app.get("/healthz")
async def health(request: Request):
    lang = get_lang_from_request(request)
    return {"status": t("status_ok", lang)}
