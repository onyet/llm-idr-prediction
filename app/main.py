from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from . import rag, tradingview, analyze
from .i18n import get_lang_from_request, set_current_lang, t

app = FastAPI(title="LLM Exchange API", version="0.1.0")


class LanguageMiddleware(BaseHTTPMiddleware):
    """Middleware untuk mengatur bahasa berdasarkan header Accept-Language."""
    async def dispatch(self, request: Request, call_next):
        lang = get_lang_from_request(request)
        set_current_lang(lang)
        response = await call_next(request)
        return response


# Tambahkan middleware bahasa terlebih dahulu
app.add_middleware(LanguageMiddleware)

app.include_router(rag.router)
app.include_router(tradingview.router)
app.include_router(analyze.router)

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
