from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from app.models import EventIn


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self.db_path.as_posix(), check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES
        )
        self._conn.row_factory = sqlite3.Row
        self.init_schema()

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def init_schema(self) -> None:
        with self._txn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    store_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    visitor_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    zone_id TEXT,
                    dwell_ms INTEGER NOT NULL,
                    is_staff INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    ingested_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_store_ts ON events (store_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_store_visitor ON events (store_id, visitor_id);
                """
            )

    def ping(self) -> None:
        with self._txn() as conn:
            conn.execute("SELECT 1").fetchone()

    def insert_events(self, events: list[EventIn]) -> tuple[int, int]:
        inserted = 0
        duplicates = 0
        now = datetime.now(timezone.utc).isoformat()
        with self._txn() as conn:
            for event in events:
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO events (
                        event_id, store_id, camera_id, visitor_id, event_type, timestamp,
                        zone_id, dwell_ms, is_staff, confidence, metadata_json, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.store_id,
                        event.camera_id,
                        event.visitor_id,
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        event.zone_id,
                        event.dwell_ms,
                        int(event.is_staff),
                        event.confidence,
                        event.metadata.model_dump_json(),
                        now,
                    ),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    duplicates += 1
        return inserted, duplicates


    def fetch_events(
        self, store_id: str, start: datetime | None = None, end: datetime | None = None
    ) -> list[dict[str, Any]]:
        clauses = ["store_id = ?"]
        params: list[Any] = [store_id]
        if start is not None:
            clauses.append("timestamp >= ?")
            params.append(start.isoformat())
        if end is not None:
            clauses.append("timestamp <= ?")
            params.append(end.isoformat())
        query = (
            "SELECT * FROM events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY timestamp ASC"
        )
        with self._txn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def fetch_all_store_ids(self) -> list[str]:
        with self._txn() as conn:
            rows = conn.execute("SELECT DISTINCT store_id FROM events").fetchall()
        return [row["store_id"] for row in rows]

    def last_event_timestamp_per_store(self) -> dict[str, datetime | None]:
        with self._txn() as conn:
            rows = conn.execute(
                "SELECT store_id, MAX(timestamp) AS ts FROM events GROUP BY store_id"
            ).fetchall()
        output: dict[str, datetime | None] = {}
        for row in rows:
            output[row["store_id"]] = datetime.fromisoformat(row["ts"]).astimezone(
                timezone.utc
            )
        return output




