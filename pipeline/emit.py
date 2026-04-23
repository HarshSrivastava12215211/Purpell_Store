from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.models import EventIn, EventMetadata, EventType


class EventEmitter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []

    def emit(
        self,
        *,
        store_id: str,
        camera_id: str,
        visitor_id: str,
        event_type: EventType,
        timestamp: datetime,
        zone_id: str | None = None,
        dwell_ms: int = 0,
        is_staff: bool = False,
        confidence: float = 0.9,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = EventIn(
            event_id=str(uuid4()),
            store_id=store_id,
            camera_id=camera_id,
            visitor_id=visitor_id,
            event_type=event_type,
            timestamp=timestamp,
            zone_id=zone_id,
            dwell_ms=dwell_ms,
            is_staff=is_staff,
            confidence=confidence,
            metadata=EventMetadata(**(metadata or {})),
        )
        payload = event.model_dump(mode="json")
        self._buffer.append(payload)
        return payload

    def flush(self) -> int:
        with self.output_path.open("w", encoding="utf-8") as fh:
            for event in sorted(self._buffer, key=lambda row: row["timestamp"]):
                fh.write(json.dumps(event) + "\n")
        return len(self._buffer)

