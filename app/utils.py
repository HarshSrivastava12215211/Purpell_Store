from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re


REENTRY_SUFFIX = re.compile(r"([_-]R\d+)$")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_visitor_id(visitor_id: str) -> str:
    return REENTRY_SUFFIX.sub("", visitor_id)


def within_minutes(ts: datetime, pivot: datetime, minutes: int) -> bool:
    return pivot - timedelta(minutes=minutes) <= ts <= pivot

