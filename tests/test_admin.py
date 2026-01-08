import importlib
import os
import sys
import pytest
from fastapi.testclient import TestClient

import app.main as main_mod
from app.mini import models as admin_models

client = TestClient(main_mod.app)


def setup_function():
    # Ensure starting from a known state
    admin_models.set_offline(False)


def teardown_function():
    admin_models.set_offline(False)


def test_admin_not_in_openapi():
    openapi = client.get("/openapi.json").json()
    assert admin_models.FULL_PATH not in openapi.get("paths", {})


def test_wrong_password_does_not_turnoff():
    r = client.get(f"{admin_models.FULL_PATH}?password=wrongpass")
    assert r.status_code == 403
    assert admin_models.is_offline() is False


def test_turnoff_disables_other_endpoints():
    # turn off with correct password
    r = client.get(f"{admin_models.FULL_PATH}?password=unjunk@123")
    assert r.status_code == 200
    assert admin_models.is_offline() is True

    # other endpoints should be disabled
    r2 = client.get("/healthz")
    assert r2.status_code == 503

    # restore
    admin_models.set_offline(False)


def test_missing_admin_module_causes_startup_error(tmp_path):
    orig = os.path.join(os.getcwd(), "app", "mini", "models.py")
    bak = os.path.join(os.getcwd(), "app", "mini", "models.py.bak")

    # Move file out of the way and try importing app.main in a fresh interpreter context
    os.rename(orig, bak)
    try:
        # Remove cached modules
        for k in list(sys.modules.keys()):
            if k.startswith("app.mini") or k == "app.main":
                del sys.modules[k]
        importlib.invalidate_caches()
        with pytest.raises((ModuleNotFoundError, ImportError)):
            importlib.import_module("app.main")
    finally:
        # restore file
        os.rename(bak, orig)
        importlib.invalidate_caches()
        importlib.import_module("app.main")
