# PROMPT: Produce tests for anomaly endpoint and health endpoint including stale feeds and graceful database failure behavior.
# CHANGES MADE: Added realistic multi-day conversion baseline setup and explicit 503 assertion by simulating storage outage. Added empty-database health test.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.errors import ServiceUnavailableError
from tests.helpers import build_event


def test_anomalies_detect_queue_spike_and_dead_zone(client):
    now = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2)

    historical_events = []
    for days_back in [3, 2, 1]:
        base = now - timedelta(days=days_back)
        visitor = f"VIS_H_{days_back}"
        historical_events.extend(
            [
                build_event(event_type="ENTRY", visitor_id=visitor, timestamp=base),
                build_event(
                    event_type="BILLING_QUEUE_JOIN",
                    visitor_id=visitor,
                    zone_id="BILLING",
                    timestamp=base + timedelta(minutes=1),
                    metadata={"queue_depth": 1, "session_seq": 2},
                ),
            ]
        )

    current_events = []
    for idx in range(7):
        ts = now - timedelta(minutes=idx * 3)
        current_events.append(build_event(event_type="ENTRY", visitor_id=f"VIS_C_{idx}", timestamp=ts))
        current_events.append(
            build_event(
                event_type="BILLING_QUEUE_JOIN",
                visitor_id=f"VIS_C_{idx}",
                zone_id="BILLING",
                timestamp=ts + timedelta(seconds=20),
                metadata={"queue_depth": idx + 1, "session_seq": 2},
            )
        )

    ingest = client.post("/events/ingest", json=historical_events + current_events)
    assert ingest.status_code == 200

    response = client.get("/stores/STORE_PURPLLE_001/anomalies")
    assert response.status_code == 200
    anomaly_types = {item["anomaly_type"] for item in response.json()["anomalies"]}
    assert "BILLING_QUEUE_SPIKE" in anomaly_types

    assert "DEAD_ZONE" in anomaly_types


def test_health_marks_stale_feed(client):
    stale_ts = datetime.now(timezone.utc) - timedelta(minutes=25)
    ingest = client.post(
        "/events/ingest",
        json=[build_event(event_type="ENTRY", visitor_id="VIS_STALE", timestamp=stale_ts)],
    )
    assert ingest.status_code == 200
    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert "STORE_PURPLLE_001" in payload["stale_feeds"]


def test_returns_503_when_database_unavailable(client, monkeypatch):
    def fail_health():
        raise ServiceUnavailableError("Database unavailable")

    monkeypatch.setattr(client.app.state.analytics_service, "health", fail_health)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


def test_health_no_events(client):
    """Health endpoint with empty database should return ok, no stale feeds."""
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["stale_feeds"] == []
    assert payload["last_event_timestamp_per_store"] == {}

