"""Purpell Store detection pipeline.

Runs YOLOv8 person detection on CCTV clips, tracks individuals across frames,
emits structured behavioral events.  Falls back to simulation when no video
files are available (useful for CI/test environments).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import random
import re
import time
from typing import Any

import requests

from app.models import EventType
from pipeline.emit import EventEmitter
from pipeline.tracker import (
    CentroidTracker,
    TrackedPerson,
    TrackIdentity,
    VisitorTracker,
)

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CLIP_RE = re.compile(
    r"(?P<store>STORE_[A-Z]{3}_\d{3}).*(?P<camera>CAM_[A-Z_0-9]+)", re.IGNORECASE
)

# Camera-to-store/zone mapping for STORE_PURPLLE_001 (Mumbai, 5-camera layout)
GENERIC_CAM_MAP: dict[str, dict[str, str]] = {
    "CAM 1": {"store_id": "STORE_PURPLLE_001", "camera_id": "CAM_1", "type": "floor"},
    "CAM 2": {"store_id": "STORE_PURPLLE_001", "camera_id": "CAM_2", "type": "floor"},
    "CAM 3": {"store_id": "STORE_PURPLLE_001", "camera_id": "CAM_3", "type": "entry"},
    "CAM 4": {"store_id": "STORE_PURPLLE_001", "camera_id": "CAM_4", "type": "staff"},
    "CAM 5": {"store_id": "STORE_PURPLLE_001", "camera_id": "CAM_5", "type": "billing"},
}


# ---------------------------------------------------------------------------
# Real CV pipeline: YOLOv8 + CentroidTracker
# ---------------------------------------------------------------------------

def _load_yolo_model(model_name: str = "yolov8n.pt"):
    """Load YOLOv8 model — downloads weights on first run."""
    from ultralytics import YOLO
    model = YOLO(model_name)
    return model


def _determine_entry_line(
    frame_h: int, frame_w: int, cam_type: str,
    entry_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Define the entry/exit crossing line for direction detection.

    Reads ``entry_rules`` from the store layout when available.  The layout
    specifies a ``cross_line`` (two points) and ``entry_direction``.

    Supported entry_direction values:
      - ``right_to_left``    cross line is vertical (x-axis), entry = cx goes left of x_line
      - ``left_to_right``    cross line is vertical (x-axis), entry = cx goes right of x_line
      - ``top_to_bottom``    cross line is horizontal (y-axis), entry = cy goes down
      - ``bottom_to_top``    cross line is horizontal (y-axis), entry = cy goes up

    Falls back to a sensible default when no rules are provided.
    """
    if entry_rules and "cross_line" in entry_rules:
        cl = entry_rules["cross_line"]   # [x1, y1, x2, y2]
        direction = entry_rules.get("entry_direction", "right_to_left")
        # Vertical cross-line (x1 == x2 or near): use x-axis
        if abs(cl[0] - cl[2]) < abs(cl[1] - cl[3]) or direction in ("right_to_left", "left_to_right"):
            x_line = (cl[0] + cl[2]) // 2
            in_dir = "left" if direction == "right_to_left" else "right"
            out_dir = "right" if direction == "right_to_left" else "left"
            return {"x_line": x_line, "axis": "x", "in_direction": in_dir, "out_direction": out_dir}
        # Horizontal cross-line: use y-axis
        y_line = (cl[1] + cl[3]) // 2
        in_dir = "down" if direction == "top_to_bottom" else "up"
        out_dir = "up" if direction == "top_to_bottom" else "down"
        return {"y_line": y_line, "axis": "y", "in_direction": in_dir, "out_direction": out_dir}

    # Default fallback
    if cam_type == "entry":
        return {"y_line": int(frame_h * 0.45), "axis": "y",
                "in_direction": "down", "out_direction": "up"}
    return {"y_line": int(frame_h * 0.5), "axis": "y",
            "in_direction": "down", "out_direction": "up"}


def _build_roi_index(
    layout: dict[str, Any], store_id: str, camera_id: str
) -> list[dict[str, Any]]:
    """Build a list of {zone_id, roi} dicts for a given camera from the layout.

    ROI is [x1, y1, x2, y2].  Returns an empty list when no layout is provided.
    """
    zones = layout.get(store_id, {}).get("zones", [])
    return [
        {"zone_id": z["zone_id"], "roi": z["roi"]}
        for z in zones
        if z.get("camera_id") == camera_id and "roi" in z and "zone_id" in z
    ]


def _assign_zone(
    cam_type: str,
    bbox: tuple,
    frame_w: int,
    frame_h: int,
    zones: list[str],
    roi_index: list[dict[str, Any]] | None = None,
) -> str | None:
    """Assign a zone to a detection.

    Priority:
    1. ROI-based matching from the layout (accurate, uses actual bounding boxes).
    2. Legacy heuristic fallback (frame position splits) when no ROI data.

    Entry cameras return None (no zone — only entry/exit events matter).
    Staff cameras return the first matched zone (typically STOCKROOM).
    """
    if cam_type == "entry":
        return None

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0

    # --- ROI-based assignment (preferred) ---
    if roi_index:
        for zone_def in roi_index:
            x1, y1, x2, y2 = zone_def["roi"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return zone_def["zone_id"]
        return None  # centroid not inside any defined ROI

    # --- Legacy heuristic fallback ---
    if cam_type == "billing":
        return "BILLING"
    if not zones:
        return None
    non_billing = [z for z in zones if z.upper() != "BILLING"]
    if not non_billing:
        return None
    zone_idx = min(int(cx / frame_w * len(non_billing)), len(non_billing) - 1)
    return non_billing[zone_idx]


def _detect_staff(track: TrackedPerson, total_clip_frames: int,
                  staff_threshold: float = 0.65) -> bool:
    """Heuristic: a track visible for >65% of the clip is likely staff.

    Staff are typically present throughout most of the footage while
    customers visit temporarily.
    """
    if total_clip_frames < 30:
        return False
    presence_ratio = track.total_frames / total_clip_frames
    return presence_ratio > staff_threshold


def process_video_clip(
    video_path: Path,
    store_id: str,
    camera_id: str,
    cam_type: str,
    emitter: EventEmitter,
    zones: list[str],
    start_timestamp: datetime,
    model: Any,
    frame_skip: int = 2,
    confidence_threshold: float = 0.30,
    entry_rules: dict[str, Any] | None = None,
    roi_index: list[dict[str, Any]] | None = None,
) -> int:
    """Process a single video clip through YOLOv8  tracker  event emission.

    Returns the number of events generated.
    """
    import cv2

    # Staff cameras: all detections are auto-flagged as staff (e.g. stockroom)
    is_staff_cam = cam_type == "staff"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    line_config = _determine_entry_line(frame_h, frame_w, cam_type, entry_rules)
    tracker = CentroidTracker(max_disappeared=int(fps * 2), max_distance=frame_w * 0.15)

    frame_idx = 0
    events_before = len(emitter._buffer)

    # Per-track state for event logic
    track_visitor_map: dict[int, str] = {}      # track_id  visitor_id
    track_entered: set[int] = set()              # tracks that emitted ENTRY
    track_in_zone: dict[int, tuple[str, datetime]] = {}  # track_id  (zone, enter_time)
    track_dwell_emitted: dict[int, datetime] = {}  # last dwell emit time
    track_exited: set[int] = set()               # tracks that emitted EXIT
    track_crossed: dict[int, str] = {}           # track_id  crossing direction
    visitor_counter = 1

    logger.info("Processing %s (%s) — %d frames @ %.1f fps",
                video_path.name, cam_type, total_frames, fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_skip != 0:
            continue

        current_time = start_timestamp + timedelta(seconds=frame_idx / fps)

        # YOLOv8 inference — person class only (class 0)
        results = model(frame, verbose=False, classes=[0],
                        conf=confidence_threshold, imgsz=640)

        detections: list[dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                detections.append({
                    "bbox": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    "confidence": conf,
                })

        # Update tracker
        active_tracks = tracker.update(detections, frame_idx)

        # Process each active track
        for track in active_tracks:
            if track.frames_since_seen > 0:
                continue  # Only process tracks seen in this frame

            tid = track.track_id
            conf = track.confidence_history[-1] if track.confidence_history else 0.85

            # Assign visitor_id
            if tid not in track_visitor_map:
                visitor_id = f"VIS_{visitor_counter:06d}"
                track_visitor_map[tid] = visitor_id
                visitor_counter += 1

            visitor_id = track_visitor_map[tid]

            # --- Entry camera logic: line crossing detection ---
            if cam_type == "entry":
                axis = line_config.get("axis", "y")

                if track.first_centroid and track.total_frames >= 3:
                    if axis == "x":
                        # Vertical cross-line (x-axis): right_to_left entry
                        x_line = line_config["x_line"]
                        cx = track.centroid[0]
                        first_x = track.first_centroid[0]
                        in_dir = line_config["in_direction"]   # "left"

                        crossed_in = (
                            in_dir == "left"
                            and first_x > x_line and cx <= x_line
                        ) or (
                            in_dir == "right"
                            and first_x < x_line and cx >= x_line
                        )
                        crossed_out = (
                            in_dir == "left"
                            and first_x <= x_line and cx > x_line
                        ) or (
                            in_dir == "right"
                            and first_x >= x_line and cx < x_line
                        )
                    else:
                        # Horizontal cross-line (y-axis)
                        y_line = line_config["y_line"]
                        cy = track.centroid[1]
                        first_y = track.first_centroid[1]
                        in_dir = line_config["in_direction"]   # "down"

                        crossed_in = (
                            in_dir == "down"
                            and first_y < y_line and cy >= y_line
                        ) or (
                            in_dir == "up"
                            and first_y > y_line and cy <= y_line
                        )
                        crossed_out = (
                            in_dir == "down"
                            and first_y >= y_line and cy < y_line
                        ) or (
                            in_dir == "up"
                            and first_y <= y_line and cy > y_line
                        )

                    if crossed_in and tid not in track_crossed:
                        track_crossed[tid] = "IN"
                        if tid not in track_entered:
                            emitter.emit(
                                store_id=store_id, camera_id=camera_id,
                                visitor_id=visitor_id, event_type=EventType.ENTRY,
                                timestamp=current_time, confidence=conf,
                                is_staff=False,
                                metadata={"session_seq": 1},
                            )
                            track_entered.add(tid)

                    elif crossed_out and tid not in track_crossed:
                        track_crossed[tid] = "OUT"
                        if tid in track_entered and tid not in track_exited:
                            emitter.emit(
                                store_id=store_id, camera_id=camera_id,
                                visitor_id=visitor_id, event_type=EventType.EXIT,
                                timestamp=current_time, confidence=conf,
                                is_staff=False,
                                metadata={"session_seq": 99},
                            )
                            track_exited.add(tid)

            # --- Staff camera logic: auto-flag all detections as staff ---
            elif is_staff_cam:
                zone = _assign_zone(cam_type, track.bbox, frame_w, frame_h, zones, roi_index)
                if zone and (tid not in track_in_zone or track_in_zone[tid][0] != zone):
                    emitter.emit(
                        store_id=store_id, camera_id=camera_id,
                        visitor_id=visitor_id, event_type=EventType.ZONE_ENTER,
                        timestamp=current_time, zone_id=zone,
                        confidence=conf, is_staff=True,
                        metadata={"sku_zone": zone, "session_seq": 0},
                    )
                    track_in_zone[tid] = (zone, current_time)

            # --- Floor / billing camera logic: zone tracking ---
            else:
                zone = _assign_zone(cam_type, track.bbox, frame_w, frame_h, zones, roi_index)
                if zone:
                    prev = track_in_zone.get(tid)
                    if prev is None or prev[0] != zone:
                        # Emit ZONE_EXIT for previous zone
                        if prev is not None:
                            emitter.emit(
                                store_id=store_id, camera_id=camera_id,
                                visitor_id=visitor_id, event_type=EventType.ZONE_EXIT,
                                timestamp=current_time, zone_id=prev[0],
                                confidence=conf, is_staff=False,
                                metadata={"sku_zone": prev[0], "session_seq": 0},
                            )
                        # Emit ZONE_ENTER for new zone
                        emitter.emit(
                            store_id=store_id, camera_id=camera_id,
                            visitor_id=visitor_id, event_type=EventType.ZONE_ENTER,
                            timestamp=current_time, zone_id=zone,
                            confidence=conf, is_staff=False,
                            metadata={"sku_zone": zone, "session_seq": 0},
                        )
                        track_in_zone[tid] = (zone, current_time)
                        track_dwell_emitted[tid] = current_time

                        # Billing zone logic
                        if zone.upper() == "BILLING":
                            # Count people currently in billing
                            billing_count = sum(
                                1 for t, (z, _) in track_in_zone.items()
                                if z.upper() == "BILLING" and t != tid
                            )
                            emitter.emit(
                                store_id=store_id, camera_id=camera_id,
                                visitor_id=visitor_id,
                                event_type=EventType.BILLING_QUEUE_JOIN,
                                timestamp=current_time, zone_id=zone,
                                confidence=conf, is_staff=False,
                                metadata={"queue_depth": billing_count + 1,
                                          "session_seq": 0},
                            )
                    else:
                        # Still in same zone — check dwell
                        last_dwell = track_dwell_emitted.get(tid, current_time)
                        if (current_time - last_dwell).total_seconds() >= 30:
                            dwell_ms = int((current_time - last_dwell).total_seconds() * 1000)
                            emitter.emit(
                                store_id=store_id, camera_id=camera_id,
                                visitor_id=visitor_id, event_type=EventType.ZONE_DWELL,
                                timestamp=current_time, zone_id=zone,
                                dwell_ms=dwell_ms, confidence=conf, is_staff=False,
                                metadata={"sku_zone": zone, "session_seq": 0},
                            )
                            track_dwell_emitted[tid] = current_time

        # Log progress periodically
        if frame_idx % (int(fps) * 60) == 0:
            elapsed_min = frame_idx / fps / 60
            logger.info("  %.1f min processed — %d active tracks, %d events so far",
                        elapsed_min, len(tracker.tracks),
                        len(emitter._buffer) - events_before)

    cap.release()

    # --- Post-processing: staff detection + final zone exits ---
    effective_frames = max(total_frames // frame_skip, 1)
    all_tracks: dict[int, TrackedPerson] = {t.track_id: t for t in tracker.deregistered}
    all_tracks.update(tracker.tracks)
    for tid, track in all_tracks.items():
        visitor_id = track_visitor_map.get(tid)
        if not visitor_id:
            continue

        is_staff = _detect_staff(track, effective_frames)

        # Retroactively flag staff events in buffer
        if is_staff:
            for ev in emitter._buffer[events_before:]:
                if ev.get("visitor_id") == visitor_id:
                    ev["is_staff"] = True

    events_generated = len(emitter._buffer) - events_before
    logger.info("Completed %s — %d events generated", video_path.name, events_generated)
    return events_generated


def _discover_clips(clips_dir: Path) -> list[tuple[Path, dict[str, str]]]:
    """Find video files and map them to store/camera metadata."""
    clips: list[tuple[Path, dict[str, str]]] = []
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        for file_path in sorted(clips_dir.rglob(ext)):
            stem = file_path.stem
            # Try structured naming first (STORE_PURPLLE_001_CAM_3.mp4)
            match = CLIP_RE.search(stem.upper())
            if match:
                clips.append((file_path, {
                    "store_id": match.group("store"),
                    "camera_id": match.group("camera"),
                    "type": _infer_cam_type(match.group("camera")),
                }))
            else:
                # Try generic naming (CAM 1.mp4, CAM 2.mp4, etc.)
                generic_key = stem.upper().replace("_", " ").strip()
                # Also try without spaces
                for key, meta in GENERIC_CAM_MAP.items():
                    if key.upper() in generic_key or key.replace(" ", "").upper() in generic_key.replace(" ", ""):
                        clips.append((file_path, dict(meta)))
                        break
    return clips


def _infer_cam_type(camera_id: str) -> str:
    """Infer camera type from camera ID."""
    cid = camera_id.upper()
    if "ENTRY" in cid:
        return "entry"
    if "BILLING" in cid:
        return "billing"
    return "floor"


def _load_layout(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def _iter_zones(layout: dict[str, Any], store_id: str) -> list[str]:
    zone_payload = layout.get(store_id, {}).get("zones", [])
    zones = [zone.get("zone_id") for zone in zone_payload if zone.get("zone_id")]
    if not zones:
        zones = ["ENTRY_ZONE", "SKINCARE", "MAKEUP", "BILLING"]
    return zones


# ---------------------------------------------------------------------------
# Video pipeline entry point
# ---------------------------------------------------------------------------

def process_clips(args: argparse.Namespace) -> int:
    """Process all discovered video clips through YOLOv8 detection pipeline."""
    layout = _load_layout(Path(args.store_layout))
    clips = _discover_clips(Path(args.clips_dir))

    if not clips:
        logger.warning("No video clips found in %s — falling back to simulation", args.clips_dir)
        return generate_events_simulation(args)

    logger.info("Found %d video clips to process", len(clips))
    model = _load_yolo_model(getattr(args, "model", "yolov8n.pt"))
    emitter = EventEmitter(Path(args.output))

    if args.start_time == "now":
        start_at = datetime.now(timezone.utc)
    else:
        start_at = datetime.fromisoformat(
            args.start_time.replace("Z", "+00:00")
        ).astimezone(timezone.utc)

    total_events = 0
    for idx, (clip_path, cam_meta) in enumerate(clips):
        store_id = cam_meta["store_id"]
        camera_id = cam_meta["camera_id"]
        cam_type = cam_meta["type"]
        zones = _iter_zones(layout, store_id)

        # Load ROI index and entry rules from layout for this camera
        roi_index = _build_roi_index(layout, store_id, camera_id)
        entry_rules = layout.get(store_id, {}).get("entry_rules") if cam_type == "entry" else None

        # Offset start time per clip to simulate sequential camera feeds
        clip_start = start_at + timedelta(minutes=idx * 20)

        events = process_video_clip(
            video_path=clip_path,
            store_id=store_id,
            camera_id=camera_id,
            cam_type=cam_type,
            emitter=emitter,
            zones=zones,
            start_timestamp=clip_start,
            model=model,
            frame_skip=getattr(args, "frame_skip", 2),
            confidence_threshold=getattr(args, "confidence_threshold", 0.30),
            entry_rules=entry_rules,
            roi_index=roi_index,
        )
        total_events += events

    flushed = emitter.flush()
    logger.info("Total: %d events written to %s", flushed, args.output)
    return flushed


# ---------------------------------------------------------------------------
# Simulation fallback (original logic, used when no clips available)
# ---------------------------------------------------------------------------

def _discover_sources_sim(clips_dir: Path, layout: dict[str, Any]) -> list[tuple[str, str]]:
    sources: list[tuple[str, str]] = []
    for store_id, payload in layout.items():
        for camera in payload.get("cameras", []):
            camera_id = camera.get("camera_id")
            if camera_id:
                sources.append((store_id, camera_id))
    if not sources:
        sources = [("STORE_PURPLLE_001", "CAM_3")]
    return sorted(set(sources))


def _emit_session(
    emitter: EventEmitter,
    rng: random.Random,
    store_id: str,
    camera_id: str,
    identity: TrackIdentity,
    zones: list[str],
    started_at: datetime,
    queue_depth: int,
    is_reentry: bool = False,
) -> tuple[datetime, int]:
    timestamp = started_at
    seq = 1
    is_staff = rng.random() < 0.08
    confidence = round(rng.uniform(0.62, 0.98), 2)
    entry_type = EventType.REENTRY if is_reentry else EventType.ENTRY
    emitter.emit(
        store_id=store_id,
        camera_id=camera_id,
        visitor_id=identity.visitor_id,
        event_type=entry_type,
        timestamp=timestamp,
        confidence=confidence,
        is_staff=is_staff,
        metadata={"session_seq": seq},
    )
    seq += 1

    visit_count = rng.randint(1, min(4, len(zones)))
    selected_zones = [zone for zone in rng.sample(zones, k=visit_count) if zone != "ENTRY_ZONE"]
    for zone in selected_zones:
        timestamp += timedelta(seconds=rng.randint(10, 45))
        emitter.emit(
            store_id=store_id,
            camera_id=camera_id,
            visitor_id=identity.visitor_id,
            event_type=EventType.ZONE_ENTER,
            timestamp=timestamp,
            zone_id=zone,
            confidence=confidence,
            is_staff=is_staff,
            metadata={"sku_zone": zone, "session_seq": seq},
        )
        seq += 1

        dwell_chunks = rng.randint(1, 4)
        for _ in range(dwell_chunks):
            timestamp += timedelta(seconds=30)
            emitter.emit(
                store_id=store_id,
                camera_id=camera_id,
                visitor_id=identity.visitor_id,
                event_type=EventType.ZONE_DWELL,
                timestamp=timestamp,
                zone_id=zone,
                dwell_ms=30000,
                confidence=confidence,
                is_staff=is_staff,
                metadata={"sku_zone": zone, "session_seq": seq},
            )
            seq += 1

        timestamp += timedelta(seconds=rng.randint(5, 25))
        emitter.emit(
            store_id=store_id,
            camera_id=camera_id,
            visitor_id=identity.visitor_id,
            event_type=EventType.ZONE_EXIT,
            timestamp=timestamp,
            zone_id=zone,
            confidence=confidence,
            is_staff=is_staff,
            metadata={"sku_zone": zone, "session_seq": seq},
        )
        seq += 1

        if zone.upper().startswith("BILLING") and not is_staff:
            queue_depth = max(queue_depth + rng.choice([0, 1]), 1)
            timestamp += timedelta(seconds=rng.randint(2, 8))
            emitter.emit(
                store_id=store_id,
                camera_id=camera_id,
                visitor_id=identity.visitor_id,
                event_type=EventType.BILLING_QUEUE_JOIN,
                timestamp=timestamp,
                zone_id=zone,
                confidence=confidence,
                is_staff=False,
                metadata={"queue_depth": queue_depth, "session_seq": seq},
            )
            seq += 1
            if rng.random() < 0.2:
                timestamp += timedelta(seconds=rng.randint(20, 70))
                emitter.emit(
                    store_id=store_id,
                    camera_id=camera_id,
                    visitor_id=identity.visitor_id,
                    event_type=EventType.BILLING_QUEUE_ABANDON,
                    timestamp=timestamp,
                    zone_id=zone,
                    confidence=confidence,
                    is_staff=False,
                    metadata={"queue_depth": queue_depth, "session_seq": seq},
                )
                queue_depth = max(queue_depth - 1, 0)
                seq += 1

    timestamp += timedelta(seconds=rng.randint(15, 60))
    emitter.emit(
        store_id=store_id,
        camera_id=camera_id,
        visitor_id=identity.visitor_id,
        event_type=EventType.EXIT,
        timestamp=timestamp,
        confidence=confidence,
        is_staff=is_staff,
        metadata={"session_seq": seq},
    )
    queue_depth = max(queue_depth - rng.choice([0, 1]), 0)
    return timestamp, queue_depth


def generate_events_simulation(args: argparse.Namespace) -> int:
    """Simulation-based event generation (fallback when no clips)."""
    rng = random.Random(args.seed)
    tracker = VisitorTracker(seed=args.seed)
    emitter = EventEmitter(Path(args.output))
    layout = _load_layout(Path(args.store_layout))
    sources = _discover_sources_sim(Path(args.clips_dir), layout)
    if args.start_time == "now":
        start_at = datetime.now(timezone.utc) - timedelta(hours=3)
    else:
        start_at = datetime.fromisoformat(args.start_time.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )

    for source_idx, (store_id, camera_id) in enumerate(sources):
        zones = _iter_zones(layout, store_id)
        current_time = start_at + timedelta(minutes=source_idx * 3)
        queue_depth = 0
        visitor_count = args.visitors_per_source
        for _ in range(visitor_count):
            identity = tracker.new_visit()
            current_time += timedelta(seconds=rng.randint(15, 80))
            current_time, queue_depth = _emit_session(
                emitter=emitter, rng=rng, store_id=store_id, camera_id=camera_id,
                identity=identity, zones=zones, started_at=current_time,
                queue_depth=queue_depth, is_reentry=False,
            )
            tracker.mark_exit(identity)
            if tracker.should_reenter(chance=args.reentry_prob):
                current_time += timedelta(seconds=rng.randint(30, 180))
                reentry_identity = tracker.reenter(identity)
                current_time, queue_depth = _emit_session(
                    emitter=emitter, rng=rng, store_id=store_id, camera_id=camera_id,
                    identity=reentry_identity, zones=zones, started_at=current_time,
                    queue_depth=queue_depth, is_reentry=True,
                )
                tracker.mark_exit(reentry_identity)
    return emitter.flush()


# Alias for backward compat with tests
generate_events = generate_events_simulation


# ---------------------------------------------------------------------------
# Replay events to ingest API
# ---------------------------------------------------------------------------

def replay_to_ingest(
    ingest_url: str, events_path: Path, batch_size: int, realtime: bool, speed: float
) -> None:
    with events_path.open("r", encoding="utf-8") as fh:
        batch: list[dict[str, Any]] = []
        previous_ts: datetime | None = None
        for line in fh:
            payload = json.loads(line)
            event_ts = datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
            if realtime and previous_ts is not None:
                delta = (event_ts - previous_ts).total_seconds() / max(speed, 0.1)
                if delta > 0:
                    time.sleep(min(delta, 2.0))
            previous_ts = event_ts
            batch.append(payload)
            if len(batch) >= batch_size:
                response = requests.post(ingest_url, json=batch, timeout=15)
                response.raise_for_status()
                batch = []
        if batch:
            response = requests.post(ingest_url, json=batch, timeout=15)
            response.raise_for_status()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Purpell Store detection pipeline")
    parser.add_argument("--clips-dir", default="data/clips", help="Path to raw CCTV clips")
    parser.add_argument(
        "--store-layout", default="data/sample/store_layout.json", help="Store layout JSON"
    )
    parser.add_argument("--output", default="data/events.jsonl", help="Output event JSONL path")
    parser.add_argument(
        "--start-time",
        default="now",
        help="Base timestamp for generated events in ISO-8601, or 'now'",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--visitors-per-source", type=int, default=18)
    parser.add_argument("--reentry-prob", type=float, default=0.2)
    parser.add_argument("--ingest-url", default="", help="Optional API ingest endpoint URL")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--speed", type=float, default=12.0, help="Realtime replay acceleration")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 model to use")
    parser.add_argument("--frame-skip", type=int, default=2,
                        help="Process every Nth frame (higher = faster, lower = more accurate)")
    parser.add_argument("--confidence-threshold", type=float, default=0.30,
                        help="Minimum detection confidence")
    parser.add_argument("--mode", choices=["auto", "cv", "sim"], default="auto",
                        help="Pipeline mode: auto (try CV, fall back to sim), cv (require clips), sim (simulation)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "sim":
        generated = generate_events_simulation(args)
        print(f"[sim] Generated {generated} events at {args.output}")
    elif args.mode == "cv":
        generated = process_clips(args)
        print(f"[cv] Generated {generated} events at {args.output}")
    else:  # auto
        clips = _discover_clips(Path(args.clips_dir))
        if clips:
            generated = process_clips(args)
            print(f"[cv] Generated {generated} events at {args.output}")
        else:
            generated = generate_events_simulation(args)
            print(f"[sim] Generated {generated} events at {args.output}")

    if args.ingest_url:
        replay_to_ingest(
            ingest_url=args.ingest_url,
            events_path=Path(args.output),
            batch_size=args.batch_size,
            realtime=args.realtime,
            speed=args.speed,
        )
        print(f"Replayed events to {args.ingest_url}")


if __name__ == "__main__":
    main()


