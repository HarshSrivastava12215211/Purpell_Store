# PROMPT: Generate tests for FastAPI ingest endpoint covering schema validation, idempotency, and batch limit behavior.
# CHANGES MADE: Added strict assertions for partial success structure and duplicate accounting to match challenge requirements.

from __future__ import annotations

from tests.helpers import build_event


def test_ingest_partial_success_and_idempotency(client):
    valid = build_event(event_type="ENTRY")
    invalid = {"event_type": "ENTRY"}

    first = client.post("/events/ingest", json=[valid, invalid])
    assert first.status_code == 200
    payload = first.json()
    assert payload["received"] == 2
    assert payload["inserted"] == 1
    assert payload["duplicates"] == 0
    assert payload["rejected"] == 1
    assert len(payload["errors"]) == 1

    second = client.post("/events/ingest", json=[valid])
    assert second.status_code == 200
    payload2 = second.json()
    assert payload2["inserted"] == 0
    assert payload2["duplicates"] == 1
    assert payload2["rejected"] == 0


def test_ingest_batch_limit(client):
    payload = [build_event(visitor_id=f"VIS_{idx:04d}") for idx in range(501)]
    response = client.post("/events/ingest", json=payload)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "BAD_REQUEST"

