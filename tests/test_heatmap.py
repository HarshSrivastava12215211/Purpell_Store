# PROMPT: Create heatmap endpoint tests for low-confidence flagging and score normalization.
# CHANGES MADE: Added explicit zone score normalization assertions and data_confidence check for small session counts.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.helpers import build_event


def test_heatmap_low_confidence_with_few_sessions(client):
    """When fewer than 20 sessions exist, data_confidence must be LOW."""
    now = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)
    events = [
        build_event(event_type="ENTRY", visitor_id="VIS_H1", timestamp=now),
        build_event(
            event_type="ZONE_ENTER",
            visitor_id="VIS_H1",
            zone_id="SKINCARE",
            timestamp=now + timedelta(seconds=10),
        ),
        build_event(
            event_type="ZONE_DWELL",
            visitor_id="VIS_H1",
            zone_id="SKINCARE",
            dwell_ms=45000,
            timestamp=now + timedelta(seconds=55),
        ),
    ]
    ingest = client.post("/events/ingest", json=events)
    assert ingest.status_code == 200

    response = client.get("/stores/STORE_PURPLLE_001/heatmap")
    assert response.status_code == 200
    payload = response.json()
    assert payload["data_confidence"] == "LOW"
    assert len(payload["zones"]) >= 1
    skincare = next(z for z in payload["zones"] if z["zone_id"] == "SKINCARE")
    assert skincare["score"] == 100.0  # only zone  max normalised score
    assert skincare["avg_dwell_ms"] == 45000.0


def test_heatmap_normalization_multiple_zones(client):
    """Zone scores normalised 0-100 where busiest zone = 100."""
    now = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)
    events = []
    # SKINCARE: 3 visits
    for i in range(3):
        events.append(
            build_event(
                event_type="ZONE_ENTER",
                visitor_id=f"VIS_N{i}",
                zone_id="SKINCARE",
                timestamp=now + timedelta(seconds=i * 10),
            )
        )
    # MAKEUP: 1 visit
    events.append(
        build_event(
            event_type="ZONE_ENTER",
            visitor_id="VIS_N10",
            zone_id="MAKEUP",
            timestamp=now + timedelta(seconds=40),
        )
    )
    ingest = client.post("/events/ingest", json=events)
    assert ingest.status_code == 200

    response = client.get("/stores/STORE_PURPLLE_001/heatmap")
    payload = response.json()
    zones = {z["zone_id"]: z for z in payload["zones"]}
    assert zones["SKINCARE"]["score"] == 100.0
    assert 0 < zones["MAKEUP"]["score"] < 100.0
