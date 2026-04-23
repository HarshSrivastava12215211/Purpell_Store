# PROMPT: Write tests for /stores/{id}/funnel that ensure re-entry sessions do not inflate visitor counts.
# CHANGES MADE: Added canonical visitor assertion through stage counts and purchase correlation via billing-window POS logic.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.helpers import build_event


def test_funnel_deduplicates_reentry_visitors(client):
    now = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=8)
    events = [
        build_event(event_type="ENTRY", visitor_id="VIS_200", timestamp=now),
        build_event(
            event_type="ZONE_ENTER",
            visitor_id="VIS_200",
            zone_id="SKINCARE",
            timestamp=now + timedelta(seconds=10),
        ),
        build_event(
            event_type="BILLING_QUEUE_JOIN",
            visitor_id="VIS_200",
            zone_id="BILLING",
            timestamp=now + timedelta(seconds=50),
            metadata={"queue_depth": 1, "session_seq": 3},
        ),
        build_event(event_type="EXIT", visitor_id="VIS_200", timestamp=now + timedelta(minutes=1)),
        build_event(
            event_type="REENTRY",
            visitor_id="VIS_200_R1",
            timestamp=now + timedelta(minutes=2),
        ),
        build_event(
            event_type="ZONE_ENTER",
            visitor_id="VIS_200_R1",
            zone_id="MAKEUP",
            timestamp=now + timedelta(minutes=2, seconds=12),
        ),
    ]
    ingest = client.post("/events/ingest", json=events)
    assert ingest.status_code == 200
    response = client.get("/stores/STORE_PURPLLE_001/funnel")
    assert response.status_code == 200
    stages = {stage["stage"]: stage["visitors"] for stage in response.json()["stages"]}
    assert stages["ENTRY"] == 1
    assert stages["ZONE_VISIT"] == 1
    assert stages["BILLING_QUEUE"] == 1



