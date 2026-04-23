"""Visitor tracking with centroid-based IoU tracker and Re-ID.

Provides both:
- CentroidTracker for frame-to-frame person tracking (real CV path)
- VisitorTracker for session-level identity management (shared by both CV and sim paths)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import count
import random
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Frame-level centroid / IoU tracker  (used by the real CV pipeline)
# ---------------------------------------------------------------------------

@dataclass
class TrackedPerson:
    """State for a tracked individual across frames."""
    track_id: int
    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]          # x1,y1,x2,y2
    frames_since_seen: int = 0
    total_frames: int = 1
    first_centroid: tuple[float, float] | None = None
    last_centroid: tuple[float, float] | None = None
    entry_frame: int = 0
    last_frame: int = 0
    direction: str | None = None             # "IN" or "OUT"
    crossed_line: bool = False
    zone_history: list[str] = field(default_factory=list)
    is_staff_candidate: bool = False
    confidence_history: list[float] = field(default_factory=list)


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Compute intersection-over-union between two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class CentroidTracker:
    """Frame-to-frame person tracker using IoU + centroid distance matching.

    Assigns persistent track IDs across frames.  When a detection can't be
    matched to an existing track it gets a new ID.  Tracks that haven't been
    seen for ``max_disappeared`` frames are deregistered.
    """

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.25,
                 max_distance: float = 120.0):
        self._next_id = 0
        self.tracks: OrderedDict[int, TrackedPerson] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        self._deregistered: list[TrackedPerson] = []

    # ------------------------------------------------------------------
    def _register(self, centroid: tuple[float, float], bbox: tuple,
                  frame_idx: int, confidence: float) -> TrackedPerson:
        track = TrackedPerson(
            track_id=self._next_id,
            centroid=centroid,
            bbox=bbox,
            first_centroid=centroid,
            last_centroid=centroid,
            entry_frame=frame_idx,
            last_frame=frame_idx,
            confidence_history=[confidence],
        )
        self.tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _deregister(self, track_id: int) -> None:
        track = self.tracks.pop(track_id, None)
        if track is not None:
            self._deregistered.append(track)

    @property
    def deregistered(self) -> list[TrackedPerson]:
        return list(self._deregistered)

    # ------------------------------------------------------------------
    def update(self, detections: list[dict[str, Any]], frame_idx: int) -> list[TrackedPerson]:
        """Update tracker with new detections for a single frame.

        Each detection dict must have keys: ``bbox`` (x1,y1,x2,y2),
        ``confidence`` (float).

        Returns list of currently active TrackedPerson objects.
        """
        if len(detections) == 0:
            for tid in list(self.tracks):
                self.tracks[tid].frames_since_seen += 1
                if self.tracks[tid].frames_since_seen > self.max_disappeared:
                    self._deregister(tid)
            return list(self.tracks.values())

        # Compute centroids for incoming detections
        input_centroids: list[tuple[float, float]] = []
        input_bboxes: list[tuple] = []
        input_confs: list[float] = []
        for det in detections:
            bx = det["bbox"]
            cx = (bx[0] + bx[2]) / 2.0
            cy = (bx[1] + bx[3]) / 2.0
            input_centroids.append((cx, cy))
            input_bboxes.append(tuple(bx))
            input_confs.append(det["confidence"])

        if len(self.tracks) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_bboxes[i], frame_idx, input_confs[i])
            return list(self.tracks.values())

        # Build cost matrix (IoU for close boxes, centroid distance as tiebreaker)
        track_ids = list(self.tracks.keys())
        track_list = [self.tracks[tid] for tid in track_ids]

        n_tracks = len(track_list)
        n_dets = len(input_centroids)
        cost = np.zeros((n_tracks, n_dets), dtype=np.float64)

        for i, track in enumerate(track_list):
            for j in range(n_dets):
                iou_score = _iou(track.bbox, input_bboxes[j])
                dist = _centroid_distance(track.centroid, input_centroids[j])
                # Combined cost: lower is better
                cost[i, j] = 1.0 - iou_score + dist / max(self.max_distance, 1)

        # Greedy matching (Hungarian is better but greedy is fine for this scale)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        for _ in range(min(n_tracks, n_dets)):
            min_val = float("inf")
            min_i, min_j = -1, -1
            for i in range(n_tracks):
                if i in matched_tracks:
                    continue
                for j in range(n_dets):
                    if j in matched_dets:
                        continue
                    if cost[i, j] < min_val:
                        min_val = cost[i, j]
                        min_i, min_j = i, j
            if min_i < 0 or min_val > 2.0:
                break
            # Verify IoU or distance constraint
            iou_ok = _iou(track_list[min_i].bbox, input_bboxes[min_j]) >= self.iou_threshold
            dist_ok = _centroid_distance(track_list[min_i].centroid, input_centroids[min_j]) < self.max_distance
            if iou_ok or dist_ok:
                matched_tracks.add(min_i)
                matched_dets.add(min_j)
                tid = track_ids[min_i]
                self.tracks[tid].centroid = input_centroids[min_j]
                self.tracks[tid].bbox = input_bboxes[min_j]
                self.tracks[tid].last_centroid = input_centroids[min_j]
                self.tracks[tid].last_frame = frame_idx
                self.tracks[tid].frames_since_seen = 0
                self.tracks[tid].total_frames += 1
                self.tracks[tid].confidence_history.append(input_confs[min_j])
            else:
                break

        # Handle unmatched tracks
        for i in range(n_tracks):
            if i not in matched_tracks:
                tid = track_ids[i]
                self.tracks[tid].frames_since_seen += 1
                if self.tracks[tid].frames_since_seen > self.max_disappeared:
                    self._deregister(tid)

        # Register new detections
        for j in range(n_dets):
            if j not in matched_dets:
                self._register(input_centroids[j], input_bboxes[j], frame_idx, input_confs[j])

        return list(self.tracks.values())


# ---------------------------------------------------------------------------
# Session-level identity tracker  (shared by both CV and simulation paths)
# ---------------------------------------------------------------------------

@dataclass
class TrackIdentity:
    logical_person_id: str
    visit_index: int

    @property
    def visitor_id(self) -> str:
        suffix = f"_R{self.visit_index}" if self.visit_index > 0 else ""
        return f"{self.logical_person_id}{suffix}"


class VisitorTracker:
    """Simple token manager for visitor sessions and re-entries."""

    def __init__(self, seed: int = 7):
        self._rng = random.Random(seed)
        self._counter = count(1)
        self._active: dict[str, TrackIdentity] = {}
        self._reentry_count: dict[str, int] = {}

    def new_visit(self) -> TrackIdentity:
        person_id = f"VIS_{next(self._counter):06d}"
        identity = TrackIdentity(logical_person_id=person_id, visit_index=0)
        self._active[person_id] = identity
        return identity

    def new_visit_from_track(self, track_id: int) -> TrackIdentity:
        """Create a visitor identity from a CV track ID."""
        person_id = f"VIS_{track_id:06d}"
        identity = TrackIdentity(logical_person_id=person_id, visit_index=0)
        self._active[person_id] = identity
        return identity

    def mark_exit(self, identity: TrackIdentity) -> None:
        self._active.pop(identity.logical_person_id, None)

    def reenter(self, identity: TrackIdentity) -> TrackIdentity:
        new_visit_index = self._reentry_count.get(identity.logical_person_id, 0) + 1
        self._reentry_count[identity.logical_person_id] = new_visit_index
        reentry_identity = TrackIdentity(
            logical_person_id=identity.logical_person_id, visit_index=new_visit_index
        )
        self._active[identity.logical_person_id] = reentry_identity
        return reentry_identity

    def should_reenter(self, chance: float = 0.15) -> bool:
        return self._rng.random() < chance
