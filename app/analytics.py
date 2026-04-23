from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from typing import Any

from pydantic import ValidationError

from app.errors import ServiceUnavailableError
from app.models import (
    AnomaliesResponse,
    AnomalyItem,
    AnomalySeverity,
    ErrorItem,
    EventIn,
    EventType,
    FunnelResponse,
    FunnelStage,
    HeatmapResponse,
    HeatmapZone,
    HealthResponse,
    IngestResponse,
    MetricsResponse,
)
from app.storage import Storage
from app.utils import canonical_visitor_id, parse_ts, utcnow, within_minutes


ZONE_EVENTS = {EventType.ZONE_ENTER.value, EventType.ZONE_EXIT.value, EventType.ZONE_DWELL.value}


@dataclass
class SessionState:
    canonical_visitor_id: str
    has_entry: bool = False
    zone_visited: bool = False
    billing_joined: bool = False
    billing_abandoned: bool = False
    billing_touchpoints: list[datetime] = field(default_factory=list)
    dwell_by_zone: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _hydrate_events(raw_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    for row in raw_events:
        metadata = row.get("metadata_json")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        hydrated.append(
            {
                **row,
                "timestamp": parse_ts(row["timestamp"]),
                "is_staff": bool(row["is_staff"]),
                "metadata": metadata if isinstance(metadata, dict) else {},
            }
        )
    return hydrated





def _build_sessions(events: list[dict[str, Any]]) -> dict[str, SessionState]:
    sessions: dict[str, SessionState] = {}
    for event in events:
        if event["is_staff"]:
            continue
        canonical_id = canonical_visitor_id(event["visitor_id"])
        state = sessions.setdefault(canonical_id, SessionState(canonical_visitor_id=canonical_id))
        event_type = event["event_type"]
        zone_id = event.get("zone_id")
        if event_type in {EventType.ENTRY.value, EventType.REENTRY.value}:
            state.has_entry = True
        if event_type in ZONE_EVENTS and zone_id:
            state.zone_visited = True
        if event_type == EventType.BILLING_QUEUE_JOIN.value:
            state.billing_joined = True
            state.billing_touchpoints.append(event["timestamp"])
        if event_type == EventType.BILLING_QUEUE_ABANDON.value:
            state.billing_abandoned = True
        if zone_id and zone_id.upper().startswith("BILLING") and event_type in {
            EventType.ZONE_ENTER.value,
            EventType.ZONE_DWELL.value,
        }:
            state.billing_joined = True
            state.billing_touchpoints.append(event["timestamp"])
        if event_type == EventType.ZONE_DWELL.value and zone_id:
            state.dwell_by_zone[zone_id] += int(event.get("dwell_ms", 0))
    return sessions




def _compute_queue_depth(events: list[dict[str, Any]]) -> int:
    queue: set[str] = set()
    for event in events:
        if event["is_staff"]:
            continue
        canonical_id = canonical_visitor_id(event["visitor_id"])
        event_type = event["event_type"]
        if event_type == EventType.BILLING_QUEUE_JOIN.value:
            queue.add(canonical_id)
        if event_type in {
            EventType.BILLING_QUEUE_ABANDON.value,
            EventType.EXIT.value,
        }:
            queue.discard(canonical_id)
    return len(queue)


def compute_metrics_payload(
    store_id: str, events: list[dict[str, Any]], as_of: datetime
) -> MetricsResponse:
    sessions = _build_sessions(events)
    unique_visitors = len(sessions)

    dwell_totals: dict[str, int] = defaultdict(int)
    dwell_counts: dict[str, int] = defaultdict(int)
    for session in sessions.values():
        for zone, dwell_ms in session.dwell_by_zone.items():
            dwell_totals[zone] += dwell_ms
            dwell_counts[zone] += 1
    avg_dwell_per_zone = {
        zone: round(dwell_totals[zone] / dwell_counts[zone], 2) for zone in sorted(dwell_totals)
    }

    joined = sum(1 for s in sessions.values() if s.billing_joined)
    abandoned = sum(1 for s in sessions.values() if s.billing_abandoned)
    abandonment_rate = round((abandoned / joined * 100.0), 2) if joined else 0.0

    return MetricsResponse(
        store_id=store_id,
        as_of=as_of,
        unique_visitors=unique_visitors,
        avg_dwell_per_zone=avg_dwell_per_zone,
        queue_depth=_compute_queue_depth(events),
        abandonment_rate=abandonment_rate,
        data_confidence="HIGH" if unique_visitors >= 20 else "LOW",
    )


def compute_funnel_payload(
    store_id: str, events: list[dict[str, Any]], as_of: datetime
) -> FunnelResponse:
    sessions = _build_sessions(events)
    entry = sum(1 for s in sessions.values() if s.has_entry or s.zone_visited or s.billing_joined)
    zone = sum(1 for s in sessions.values() if s.zone_visited or s.billing_joined)
    billing = sum(1 for s in sessions.values() if s.billing_joined)
    counts = [entry, zone, billing]
    labels = ["ENTRY", "ZONE_VISIT", "BILLING_QUEUE"]
    stages: list[FunnelStage] = []
    for idx, label in enumerate(labels):
        if idx == 0:
            drop_off = 0.0
        else:
            prev = counts[idx - 1]
            drop_off = round(((prev - counts[idx]) / prev * 100.0), 2) if prev else 0.0
        stages.append(FunnelStage(stage=label, visitors=counts[idx], drop_off_pct=drop_off))
    return FunnelResponse(store_id=store_id, as_of=as_of, stages=stages)


def compute_heatmap_payload(
    store_id: str, events: list[dict[str, Any]], as_of: datetime
) -> HeatmapResponse:
    non_staff = [event for event in events if not event["is_staff"]]
    sessions = _build_sessions(non_staff)
    visit_counts: dict[str, int] = defaultdict(int)
    dwell_totals: dict[str, int] = defaultdict(int)
    dwell_counts: dict[str, int] = defaultdict(int)
    for event in non_staff:
        zone_id = event.get("zone_id")
        if not zone_id:
            continue
        if event["event_type"] in {EventType.ZONE_ENTER.value, EventType.ZONE_DWELL.value}:
            visit_counts[zone_id] += 1
        if event["event_type"] == EventType.ZONE_DWELL.value:
            dwell_totals[zone_id] += int(event.get("dwell_ms", 0))
            dwell_counts[zone_id] += 1
    max_visits = max(visit_counts.values(), default=1)
    zones = [
        HeatmapZone(
            zone_id=zone,
            visits=visit_counts[zone],
            avg_dwell_ms=round(dwell_totals[zone] / dwell_counts[zone], 2)
            if dwell_counts[zone]
            else 0.0,
            score=round(visit_counts[zone] / max_visits * 100.0, 2),
        )
        for zone in sorted(visit_counts)
    ]
    return HeatmapResponse(
        store_id=store_id,
        as_of=as_of,
        data_confidence="HIGH" if len(sessions) >= 20 else "LOW",
        zones=zones,
    )



def compute_anomalies_payload(
    store_id: str,
    events: list[dict[str, Any]],
    known_zones: list[str],
    as_of: datetime,
) -> AnomaliesResponse:
    anomalies: list[AnomalyItem] = []

    one_hour_ago = as_of - timedelta(hours=1)
    two_hours_ago = as_of - timedelta(hours=2)
    recent = [e for e in events if one_hour_ago <= e["timestamp"] <= as_of]
    prior = [e for e in events if two_hours_ago <= e["timestamp"] < one_hour_ago]
    recent_depth = _compute_queue_depth(recent)
    prior_depth = max(_compute_queue_depth(prior), 1)
    if recent_depth >= 5 and recent_depth >= prior_depth * 2:
        anomalies.append(
            AnomalyItem(
                anomaly_type="BILLING_QUEUE_SPIKE",
                severity=AnomalySeverity.CRITICAL if recent_depth >= 10 else AnomalySeverity.WARN,
                message=f"Billing queue depth is {recent_depth}, up from {prior_depth} in the prior hour.",
                suggested_action="Open an additional billing counter and redirect floor staff.",
            )
        )

    cutoff = as_of - timedelta(minutes=30)
    active_zones = {
        e["zone_id"]
        for e in events
        if e.get("zone_id")
        and e["event_type"] in {EventType.ZONE_ENTER.value, EventType.ZONE_DWELL.value}
        and e["timestamp"] >= cutoff
        and not e["is_staff"]
    }
    all_zones = set(known_zones) | {
        e["zone_id"] for e in events if e.get("zone_id") and not e["is_staff"]
    }
    dead = sorted(zone for zone in all_zones if zone not in active_zones)
    if dead:
        sample = ", ".join(dead[:3])
        anomalies.append(
            AnomalyItem(
                anomaly_type="DEAD_ZONE",
                severity=AnomalySeverity.WARN,
                message=f"No customer visits in last 30 minutes for zone(s): {sample}.",
                suggested_action="Inspect camera feed, zone merchandising, and in-aisle staffing.",
            )
        )

    return AnomaliesResponse(store_id=store_id, as_of=as_of, anomalies=anomalies)


class AnalyticsService:
    def __init__(self, storage: Storage, store_layout: dict[str, Any] | None = None):
        self.storage = storage
        self.store_layout = store_layout or {}

    def ingest(self, payload: list[dict[str, Any]]) -> IngestResponse:
        if len(payload) > 500:
            raise ValueError("Maximum 500 events per ingest call")
        valid_events: list[EventIn] = []
        errors: list[ErrorItem] = []
        for idx, item in enumerate(payload):
            try:
                valid_events.append(EventIn.model_validate(item))
            except ValidationError as exc:
                errors.append(ErrorItem(index=idx, reason=exc.errors()[0]["msg"]))
        try:
            inserted, duplicates = self.storage.insert_events(valid_events)
        except sqlite3.Error as exc:
            raise ServiceUnavailableError("Database unavailable") from exc
        return IngestResponse(
            received=len(payload),
            inserted=inserted,
            duplicates=duplicates,
            rejected=len(errors),
            errors=errors,
        )

    def _window_today(self, as_of: datetime | None = None) -> tuple[datetime, datetime]:
        current = as_of or utcnow()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        else:
            current = current.astimezone(timezone.utc)
        start = datetime(current.year, current.month, current.day, tzinfo=timezone.utc)
        return start, current

    def _known_zones(self, store_id: str) -> list[str]:
        zones = self.store_layout.get(store_id, {}).get("zones", [])
        return [zone.get("zone_id") for zone in zones if zone.get("zone_id")]

    def metrics(self, store_id: str, as_of: datetime | None = None) -> MetricsResponse:
        start, end = self._window_today(as_of)
        try:
            events = _hydrate_events(self.storage.fetch_events(store_id, start, end))
        except sqlite3.Error as exc:
            raise ServiceUnavailableError("Database unavailable") from exc
        return compute_metrics_payload(store_id, events, end)

    def funnel(self, store_id: str, as_of: datetime | None = None) -> FunnelResponse:
        start, end = self._window_today(as_of)
        try:
            events = _hydrate_events(self.storage.fetch_events(store_id, start, end))
        except sqlite3.Error as exc:
            raise ServiceUnavailableError("Database unavailable") from exc
        return compute_funnel_payload(store_id, events, end)

    def heatmap(self, store_id: str, as_of: datetime | None = None) -> HeatmapResponse:
        start, end = self._window_today(as_of)
        try:
            events = _hydrate_events(self.storage.fetch_events(store_id, start, end))
        except sqlite3.Error as exc:
            raise ServiceUnavailableError("Database unavailable") from exc
        return compute_heatmap_payload(store_id, events, end)

    def anomalies(self, store_id: str, as_of: datetime | None = None) -> AnomaliesResponse:
        end = as_of or utcnow()
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        else:
            end = end.astimezone(timezone.utc)
        start = end - timedelta(days=8)
        try:
            events = _hydrate_events(self.storage.fetch_events(store_id, start, end))
        except sqlite3.Error as exc:
            raise ServiceUnavailableError("Database unavailable") from exc
        return compute_anomalies_payload(
            store_id=store_id,
            events=events,
            known_zones=self._known_zones(store_id),
            as_of=end,
        )

    def health(self) -> HealthResponse:
        now = utcnow()
        try:
            per_store = self.storage.last_event_timestamp_per_store()
        except sqlite3.Error as exc:
            raise ServiceUnavailableError("Database unavailable") from exc
        stale = [
            store_id
            for store_id, timestamp in per_store.items()
            if timestamp and now - timestamp > timedelta(minutes=10)
        ]
        return HealthResponse(
            status="ok",
            generated_at=now,
            last_event_timestamp_per_store=per_store,
            stale_feeds=stale,
        )
