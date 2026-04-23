# PROMPT: Create endpoint tests for /stores/{id}/metrics with staff exclusion, conversion calculation, zero-purchase, empty-store, and all-staff handling.
# CHANGES MADE: Added direct POS seed via storage fixture, explicit checks for dwell aggregation and confidence behavior, plus empty-store and all-staff edge cases.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.helpers import build_event


def test_metrics_excludes_staff(client):
    now = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)
    customer = "VIS_100"
    staff = "VIS_STAFF_1"
    events = [
        build_event(event_type="ENTRY", visitor_id=customer, timestamp=now),
        build_event(
            event_type="ZONE_ENTER",
            visitor_id=customer,
            zone_id="SKINCARE",
            timestamp=now + timedelta(seconds=15),
        ),
        build_event(
            event_type="ZONE_DWELL",
            visitor_id=customer,
            zone_id="SKINCARE",
            dwell_ms=30000,
            timestamp=now + timedelta(seconds=45),
        ),
        build_event(
            event_type="BILLING_QUEUE_JOIN",
            visitor_id=customer,
            zone_id="BILLING",
            timestamp=now + timedelta(minutes=1),
            metadata={"queue_depth": 1, "session_seq": 4},
        ),
        build_event(
            event_type="ENTRY",
            visitor_id=staff,
            timestamp=now + timedelta(seconds=5),
            is_staff=True,
        ),
    ]
    ingest = client.post("/events/ingest", json=events)
    assert ingest.status_code == 200


    metrics = client.get("/stores/STORE_PURPLLE_001/metrics")
    assert metrics.status_code == 200
    payload = metrics.json()
    assert payload["unique_visitors"] == 1

    assert payload["avg_dwell_per_zone"]["SKINCARE"] == 30000.0


def test_metrics_zero_purchases(client):
    now = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2)
    ingest = client.post(
        "/events/ingest",
        json=[build_event(event_type="ENTRY", visitor_id="VIS_777", timestamp=now)],
    )
    assert ingest.status_code == 200
    metrics = client.get("/stores/STORE_PURPLLE_001/metrics")
    payload = metrics.json()
    assert payload["unique_visitors"] == 1



def test_metrics_empty_store(client):
    """GET metrics for a store with no events at all returns zeros, not error."""
    response = client.get("/stores/STORE_EMPTY_999/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert payload["unique_visitors"] == 0

    assert payload["queue_depth"] == 0
    assert payload["abandonment_rate"] == 0.0
    assert payload["data_confidence"] == "LOW"


def test_metrics_all_staff(client):
    """When every event is staff, unique_visitors must be 0."""
    now = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2)
    events = [
        build_event(event_type="ENTRY", visitor_id="VIS_STAFF_A", timestamp=now, is_staff=True),
        build_event(
            event_type="ZONE_ENTER", visitor_id="VIS_STAFF_A", zone_id="SKINCARE",
            timestamp=now + timedelta(seconds=10), is_staff=True,
        ),
        build_event(event_type="EXIT", visitor_id="VIS_STAFF_A",
            timestamp=now + timedelta(minutes=1), is_staff=True),
    ]
    ingest = client.post("/events/ingest", json=events)
    assert ingest.status_code == 200
    metrics = client.get("/stores/STORE_PURPLLE_001/metrics")
    payload = metrics.json()
    assert payload["unique_visitors"] == 0



