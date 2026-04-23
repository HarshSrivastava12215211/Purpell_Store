# DESIGN.md

## Architecture Overview

This system is a full pipeline from raw CCTV footage to operational store analytics API, with explicit separation between event production, intelligence serving, and live observability. The architecture has four layers:

1. **Detection and tracking pipeline** (`pipeline/`) — YOLOv8 person detection + centroid-IoU tracker
2. **Event stream contract** — JSONL + REST ingest payload with structured behavioral events
3. **Real-time intelligence API** (`app/`) — FastAPI with SQLite-backed analytics
4. **Live dashboard** (`dashboard/`) — Web UI with Chart.js and terminal fallback

### Detection Pipeline Architecture

The detection layer uses a **dual-mode design**: a real computer vision path and a simulation fallback.

**CV Path** (`--mode cv` or `--mode auto` with clips present):
- **YOLOv8n** (`yolov8n.pt`) runs inference on each frame (class 0 = person only), with configurable frame skip and confidence threshold
- **CentroidTracker** performs frame-to-frame tracking using a combined IoU + centroid-distance matching cost matrix, with greedy Hungarian-style assignment
- **Entry/exit detection** uses line-crossing on entry cameras — the crossing line is positioned at ~45% frame height; downward crossing = ENTRY, upward = EXIT
- **Zone assignment** on floor cameras splits the frame horizontally into zones from `store_layout.json`; billing cameras always assign to BILLING zone
- **Staff detection** uses a presence heuristic: tracks visible for >65% of the clip duration are classified as staff (retail staff are persistently visible; customers are transient)
- **Dwell tracking** emits ZONE_DWELL events every 30 seconds of continuous zone presence
- **Queue depth** is computed as the count of concurrent tracks in the billing zone

**Simulation Path** (`--mode sim` or auto-fallback when no clips found):
- Generates schema-compliant events with realistic session structure, including re-entry, queue join/abandon, staff flags, and dwell intervals
- Uses seeded RNG for deterministic output — useful for tests and CI

Both paths produce identical event schemas, so the API layer is completely agnostic to the upstream source.

### API Layer Architecture

The API is implemented in FastAPI with a layered service architecture:

- **`app/main.py`** — HTTP transport, middleware (structured logging, trace IDs), exception handlers, CORS, and static file mounts
- **`app/analytics.py`** — Business logic, metric computation, funnel construction, anomaly detection. Session is the first-class unit — re-entry IDs are canonicalized via regex suffix stripping to prevent double-counting
- **`app/storage.py`** — SQLite persistence with thread-safe transactions, idempotent insert (INSERT OR IGNORE), and indexed queries on `(store_id, timestamp)`
- **`app/models.py`** — Pydantic v2 schemas for strict validation, including UUID, timezone, and enum enforcement

Anomaly detection includes three detectors:
1. **Billing queue spike** — current-hour queue depth 5 and 2 prior-hour depth

3. **Dead zone** — zones with no customer visits in the last 30 minutes

### Live Dashboard

Two dashboard options:
- **Web UI** (`/dashboard`) — served as static HTML from the API, uses Chart.js for visitor funnel and zone heatmap visualizations, auto-refreshes every 3 seconds
- **Terminal** (`dashboard/live_dashboard.py`) — uses `rich` library for tabular display, useful for headless environments

### Operational Design

- Structured JSON request logging with `trace_id`, `store_id`, `endpoint`, `latency_ms`, `event_count`, `status_code`
- `/health` reports per-store last event timestamps and STALE_FEED warnings (>10 min lag)
- Graceful degradation: storage failures  structured 503 with error code, no leaked stack traces
- Docker HEALTHCHECK configured for container orchestration liveness probes

## AI-Assisted Decisions

### 1) Detection model selection and tracker architecture

I used Claude to evaluate YOLOv8 vs RT-DETR vs MediaPipe for person detection in retail CCTV footage. The prompt was: *"Compare YOLOv8n, RT-DETR-L, and MediaPipe Pose for detecting people in 1080p retail CCTV footage at 15fps on CPU. Consider: accuracy on crowded scenes, inference speed, handling of partial occlusion, and setup complexity for a 48-hour challenge."*

Claude recommended YOLOv8n as the best trade-off: fastest inference (~15ms/frame on CPU), good crowded-scene performance, and single-line API. RT-DETR was rated better for occlusion but 3-5 slower. I agreed and used YOLOv8n. For tracking, I evaluated ByteTrack vs a custom centroid-IoU tracker. I chose custom because ByteTrack's Kalman filter adds complexity without proportional benefit at 15fps retail footage, and a centroid-IoU approach is easier to debug and tune for the specific line-crossing use case.

### 2) Partial-success ingest contract

An AI suggestion proposed rejecting the entire batch on the first malformed record for simplicity. I rejected that and intentionally implemented per-item validation with indexed errors. Production event streams are inherently noisy — a single malformed event from a camera glitch should not block 499 valid events. The structured `IngestResponse` with `errors[].index` and `errors[].reason` lets operators debug specific failures without replaying entire batches.

### 3) Staff detection approach

I evaluated three approaches with AI assistance: (a) uniform color detection using HSV histograms, (b) VLM-based classification using GPT-4V/Claude Vision, (c) temporal presence heuristic. The VLM approach was promising — I tested a prompt: *"Is this person wearing a retail store uniform? Reply YES or NO."* — but inference cost and latency made it impractical for frame-by-frame processing. The HSV approach breaks when uniform colors overlap with customer clothing. I chose the temporal heuristic (staff visible >65% of clip duration) because it requires zero additional model inference and is robust across lighting conditions. This could be upgraded to VLM-based spot-checks in production.


## Store Layout Design (store_layout.json)

To bridge the physical space with digital detection, I manually designed a custom store_layout.json schema. This file acts as the configuration bedrock for spatial awareness:

1. **Camera Metadata Mapping**: It maps generic camera IDs (CAM_1, CAM_2, etc.) to logical store sections (SKINCARE_SECTION, ENTRY_THRESHOLD). This allows the pipeline to know *what* it is looking at.
2. **Custom Regions of Interest (ROIs)**: I designed absolute bounding-box coordinate arrays [x_min, y_min, x_max, y_max] for distinct sub-zones (e.g., SKINCARE, DOOR_MAT, BILLING). 
   - When a tracked individual's centroid falls within these ROI coordinates, a ZONE_ENTER or ZONE_DWELL event is emitted.
   - I explicitly separated overlapping views, such as defining QUEUE_WAIT adjacent to the BILLING zone, enabling precise queue-depth analytics.
3. **Entry Rule Crossing Lines**: For entry cameras, I defined an explicit cross_line vector along with entry_direction (e.g., ight_to_left). This enables the system to differentiate between a customer entering the store vs. just walking past the door, triggering a distinct funnel event.

By designing this layout schema, the computer vision pipeline remains completely agnostic of the physical store geometry. Adding a new store or camera just requires updating the JSON configuration without changing any python logic.
