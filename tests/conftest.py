from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


def _write_layout(path: Path) -> None:
    payload = {
        "STORE_PURPLLE_001": {
            "city": "Mumbai",
            "zones": [
                {"zone_id": "ENTRY", "camera_id": "CAM_3", "roi": [700, 120, 1280, 720]},
                {"zone_id": "SKINCARE", "camera_id": "CAM_1", "roi": [0, 80, 900, 500]},
                {"zone_id": "MAKEUP", "camera_id": "CAM_2", "roi": [420, 100, 1366, 620]},
                {"zone_id": "BILLING", "camera_id": "CAM_5", "roi": [0, 120, 620, 768]},
                {"zone_id": "STOCKROOM", "camera_id": "CAM_4", "roi": [0, 0, 1366, 768]},
            ],
            "cameras": [
                {"camera_id": "CAM_1", "type": "floor"},
                {"camera_id": "CAM_2", "type": "floor"},
                {"camera_id": "CAM_3", "type": "entry"},
                {"camera_id": "CAM_4", "type": "staff"},
                {"camera_id": "CAM_5", "type": "billing"},
            ],
            "entry_rules": {
                "camera_id": "CAM_3",
                "cross_line": [860, 120, 860, 720],
                "entry_direction": "right_to_left",
                "exit_direction": "left_to_right",
            },
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    layout_path = tmp_path / "store_layout.json"
    _write_layout(layout_path)
    monkeypatch.setenv("DB_PATH", db_path.as_posix())
    monkeypatch.setenv("STORE_LAYOUT_PATH", layout_path.as_posix())
    for module_name in ["app.config", "app.main"]:
        if module_name in sys.modules:
            del sys.modules[module_name]

    app_main = importlib.import_module("app.main")
    with TestClient(app_main.app) as test_client:
        yield test_client


