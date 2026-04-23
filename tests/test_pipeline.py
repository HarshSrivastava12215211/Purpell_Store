# PROMPT: Create tests for the detection pipeline generator ensuring schema-compliant JSONL output, deterministic behavior, and CentroidTracker correctness.
# CHANGES MADE: Added CentroidTracker unit tests for registration, matching, deregistration, and IoU computation. Added robust pipeline schema validation.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.detect import generate_events
from pipeline.tracker import CentroidTracker, TrackedPerson, _iou, _centroid_distance


def test_pipeline_generates_schema_compliant_jsonl(tmp_path):
    layout = {
        "STORE_PURPLLE_001": {
            "cameras": [{"camera_id": "CAM_3", "type": "entry"}],
            "zones": [{"zone_id": "SKINCARE", "camera_id": "CAM_1"}, {"zone_id": "BILLING", "camera_id": "CAM_5"}],
        }
    }
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    output_path = tmp_path / "events.jsonl"

    args = argparse.Namespace(
        clips_dir=tmp_path.as_posix(),
        store_layout=layout_path.as_posix(),
        output=output_path.as_posix(),
        start_time="2026-03-03T14:00:00Z",
        seed=11,
        visitors_per_source=5,
        reentry_prob=0.2,
        ingest_url="",
        batch_size=100,
        realtime=False,
        speed=10.0,
    )
    generated_count = generate_events(args)
    assert generated_count > 0
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == generated_count
    first = json.loads(lines[0])
    required = {
        "event_id",
        "store_id",
        "camera_id",
        "visitor_id",
        "event_type",
        "timestamp",
        "zone_id",
        "dwell_ms",
        "is_staff",
        "confidence",
        "metadata",
    }
    assert required.issubset(first.keys())


def test_pipeline_event_types_present(tmp_path):
    """Pipeline generates all major event types including ENTRY, EXIT, ZONE_ENTER, ZONE_DWELL."""
    layout = {
        "STORE_BLR_002": {
            "cameras": [{"camera_id": "CAM_ENTRY_01"}],
            "zones": [{"zone_id": "SKINCARE"}, {"zone_id": "BILLING"}],
        }
    }
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout), encoding="utf-8")
    output_path = tmp_path / "events.jsonl"

    args = argparse.Namespace(
        clips_dir=tmp_path.as_posix(),
        store_layout=layout_path.as_posix(),
        output=output_path.as_posix(),
        start_time="2026-03-03T14:00:00Z",
        seed=42,
        visitors_per_source=10,
        reentry_prob=0.3,
        ingest_url="",
        batch_size=100,
        realtime=False,
        speed=10.0,
    )
    generate_events(args)
    events = [json.loads(line) for line in output_path.read_text("utf-8").strip().splitlines()]
    event_types = {e["event_type"] for e in events}
    assert "ENTRY" in event_types
    assert "EXIT" in event_types
    assert "ZONE_ENTER" in event_types
    assert "ZONE_DWELL" in event_types


# --- CentroidTracker unit tests ---

def test_centroid_tracker_registers_new_detections():
    tracker = CentroidTracker(max_disappeared=5)
    detections = [
        {"bbox": (100, 100, 200, 300), "confidence": 0.9},
        {"bbox": (400, 100, 500, 300), "confidence": 0.85},
    ]
    active = tracker.update(detections, frame_idx=1)
    assert len(active) == 2
    assert all(isinstance(t, TrackedPerson) for t in active)
    assert active[0].track_id != active[1].track_id


def test_centroid_tracker_matches_close_detections():
    tracker = CentroidTracker(max_disappeared=5, max_distance=200)
    # Frame 1: person at (100, 100)
    tracker.update([{"bbox": (80, 80, 120, 200), "confidence": 0.9}], frame_idx=1)
    initial_id = list(tracker.tracks.keys())[0]

    # Frame 2: same person moved slightly
    tracker.update([{"bbox": (90, 85, 130, 205), "confidence": 0.88}], frame_idx=2)
    assert initial_id in tracker.tracks
    assert tracker.tracks[initial_id].total_frames == 2


def test_centroid_tracker_deregisters_disappeared_tracks():
    tracker = CentroidTracker(max_disappeared=2)
    tracker.update([{"bbox": (100, 100, 200, 300), "confidence": 0.9}], frame_idx=1)
    assert len(tracker.tracks) == 1

    # No detections for 3 frames  should deregister
    tracker.update([], frame_idx=2)
    tracker.update([], frame_idx=3)
    tracker.update([], frame_idx=4)
    assert len(tracker.tracks) == 0
    assert len(tracker.deregistered) == 1


def test_centroid_tracker_handles_multiple_detections():
    tracker = CentroidTracker(max_disappeared=5, max_distance=300)
    # 3 people enter at once (group)
    detections = [
        {"bbox": (50, 100, 120, 300), "confidence": 0.92},
        {"bbox": (200, 100, 270, 300), "confidence": 0.87},
        {"bbox": (350, 100, 420, 300), "confidence": 0.91},
    ]
    active = tracker.update(detections, frame_idx=1)
    assert len(active) == 3  # Each person gets own track


def test_iou_computation():
    # Identical boxes  IoU = 1.0
    assert _iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0
    # No overlap  IoU = 0.0
    assert _iou((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0
    # Partial overlap
    iou = _iou((0, 0, 100, 100), (50, 50, 150, 150))
    assert 0.1 < iou < 0.3  # ~14.3%


def test_centroid_distance():
    assert _centroid_distance((0, 0), (3, 4)) == 5.0
    assert _centroid_distance((10, 10), (10, 10)) == 0.0


def test_centroid_tracker_empty_frames():
    """Tracker handles empty frames without crashing."""
    tracker = CentroidTracker(max_disappeared=10)
    active = tracker.update([], frame_idx=1)
    assert len(active) == 0
    active = tracker.update([], frame_idx=2)
    assert len(active) == 0
