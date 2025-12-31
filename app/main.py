from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import rag

app = FastAPI(title="LLM Exchange API", version="0.1.0")
app.include_router(rag.router)

# allow CORS for quick testing in browser (adjust origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello, FastAPI from LLM-EXHANGE"}


@app.get("/healthz")
async def health():
    return {"status": "ok"}
