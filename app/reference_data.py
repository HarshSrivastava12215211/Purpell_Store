from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.utils import parse_ts


def load_store_layout(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)
