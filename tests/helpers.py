from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def build_event(
    *,
    event_type: str = "ENTRY",
    visitor_id: str = "VIS_001",
    store_id: str = "STORE_PURPLLE_001",
    zone_id: str | None = None,
    dwell_ms: int = 0,
    timestamp: datetime | None = None,
    is_staff: bool = False,
    metadata: dict | None = None,
) -> dict:
    ts = (timestamp or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z")
    return {
        "event_id": str(uuid4()),
        "store_id": store_id,
        "camera_id": "CAM_3",
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": ts,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": 0.92,
        "metadata": metadata or {"session_seq": 1},
    }

