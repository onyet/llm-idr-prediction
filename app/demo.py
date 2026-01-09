from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response
from pathlib import Path

router = APIRouter()


@router.get("/demo/1", response_class=HTMLResponse)
async def demo_one():
    p = Path(__file__).resolve().parents[1] / "desain-demo" / "index.html"
    if not p.exists():
        return Response(status_code=404, content="Not found")
    html = p.read_text(encoding="utf-8")
    return HTMLResponse(content=html)
