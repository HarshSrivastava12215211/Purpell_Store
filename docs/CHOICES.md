# CHOICES.md

## 1) Detection Model and Pipeline Choice

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **YOLOv8n + CentroidTracker** | Fast inference (~15ms/frame CPU), battle-tested on person detection, simple API, small model (6MB) | Lower accuracy than larger models on occlusion |
| **RT-DETR-L + ByteTrack** | Best occlusion handling via transformer attention, Kalman-filtered tracks | 3-5 slower, complex setup, overkill for 15fps retail footage |
| **MediaPipe Pose + DeepSORT** | Skeleton-based tracking, good for direction detection | Poor in crowded scenes, requires pose estimation overhead |
| **Simulation-first with model-pluggable architecture** | Guarantees schema fidelity and API correctness | No real CV — sacrifices Part A detection accuracy |

### What AI Suggested

I prompted Claude: *"For a 48-hour take-home challenge processing retail CCTV at 1080p/15fps, which detection model gives the best accuracy-to-setup-time ratio? I need person detection, not pose estimation. CPU-only is likely."*

Claude recommended **YOLOv8n** as the practical default, citing documentation quality, one-line inference API, and reliable person detection on retail footage. It also suggested a centroid-based tracker over ByteTrack for simplicity given the time constraint.

### What I Chose and Why

I implemented a **dual-mode pipeline**: YOLOv8n + CentroidTracker for real video processing, with simulation fallback for CI and testing.

**Why YOLOv8n specifically:**
- Nano variant runs at ~15ms/frame on CPU at 640px inference size — fast enough for batch processing 20-minute clips
- Person class (class 0) detection is strong even in crowded retail scenes
- Confidence scores are well-calibrated — I set threshold at 0.30 to capture partial occlusions rather than miss them
- Weights auto-download on first run — no manual model setup

**Why CentroidTracker over ByteTrack/DeepSORT:**
- Greedy IoU + centroid-distance matching is sufficient at 15fps (objects don't move far between frames)
- Line-crossing direction detection only needs position history — no Kalman prediction needed
- Max-disappeared parameter handles brief occlusions (person behind display for <2 seconds)
- ~60 lines of code vs importing heavy tracking libraries with complex configuration

**Staff detection via temporal presence heuristic (>65% clip duration):**
- I evaluated using a VLM for uniform detection. Prompt tested: *"Is this person wearing a retail store uniform? Reply YES or NO with confidence."* 
- VLM worked in spot-checks but was too slow for frame-level classification (>1s per query vs <20ms for our heuristic)
- The temporal approach is robust: staff are persistently visible, customers are transient. False positive rate is low in 20-minute clips.

**Runtime benchmarks (single CAM clip, ~180MB, 20 min @ 15fps):**
- Frame skip=2: ~9,000 frames processed, ~5 min wall time on CPU
- Frame skip=4: ~4,500 frames, ~2.5 min wall time (recommended for full dataset processing)

## 2) Event Schema Design Rationale

### Options Considered

- **Minimal schema**: `event_id`, `visitor_id`, `event_type`, `timestamp` only — lightweight but requires separate lookups for zone and confidence data
- **Rich schema with operational metadata**: include zone, dwell, confidence, staff flag, queue depth, and extensible metadata object — heavier but self-contained for analytics

### What AI Suggested

I prompted: *"Design an event schema for retail store visitor tracking that supports: conversion funnel analysis, zone heatmaps, queue monitoring, and staff exclusion. Should I embed metadata or use separate tables?"*

Claude recommended a rich embedded schema to avoid JOIN overhead in real-time analytics queries, with optional metadata fields to prevent schema migrations. It specifically suggested keeping `is_staff` as a top-level boolean rather than burying it in metadata, and using `confidence` to support uncertainty-aware filtering.

### What I Chose and Why

I implemented the **rich schema** exactly as the challenge specifies, with these design decisions:

- **`event_id` as UUID v4**: Enables idempotent ingest via primary key deduplication (`INSERT OR IGNORE`). UUIDs are generated at emission time, not ingest time, so the pipeline owns identity.
- **`is_staff` as top-level boolean**: Not metadata — staff exclusion is a first-class analytics concern. Every metric query filters on `WHERE is_staff = 0`.
- **`confidence` preserved without suppression**: Low-confidence detections (0.300.60) are emitted and stored, not silently dropped. This supports transparency and post-hoc analysis. The API could filter by confidence threshold if needed.
- **`visitor_id` with re-entry suffix convention** (`VIS_xxx_R1`): The canonical visitor ID is extracted by stripping the `_R\d+` suffix. This means the pipeline produces distinct `visitor_id` values for each visit while the API can deduplicate the same physical person for funnel analysis.
- **`metadata.queue_depth`**: Populated only for BILLING_QUEUE_JOIN events. Null for all other event types. This sparse approach avoids meaningless zeros.
- **`metadata.session_seq`**: Ordinal position within a visitor session, useful for debugging event ordering and session reconstruction.

## 3) API Architecture Choice: Layered Service with SQLite

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **Single-file FastAPI with inline SQL** | Fast to prototype, minimal files | Untestable, unmaintainable, no separation of concerns |
| **Layered with Storage + AnalyticsService** | Clean test isolation, swappable storage, explicit error boundaries | More files, slightly more boilerplate |
| **Event-stream + async worker (Kafka/Redis)** | Best scalability, true real-time | Overkill for 48-hour scope, complex Docker setup |

### What AI Suggested

Claude recommended the layered architecture as the best balance between challenge speed and production feel. Specific suggestions I adopted:
- Middleware for structured logging with `trace_id` (not per-endpoint manual logging)
- Exception handler registration for `ServiceUnavailableError`  503
- `asynccontextmanager` lifespan for clean startup/shutdown

### What I Chose and Why

**Layered architecture with SQLite**:

- **`app/main.py`** — Transport only: routes, middleware, exception handlers. No business logic.
- **`app/analytics.py`** — Pure business logic: `AnalyticsService` with `ingest()`, `metrics()`, `funnel()`, `heatmap()`, `anomalies()`, `health()`. Can be unit-tested without HTTP.
- **`app/storage.py`** — SQLite with thread-safe lock, `INSERT OR IGNORE` for idempotency, indexed queries. Swappable to PostgreSQL by implementing the same interface.
- **`app/models.py`** — Pydantic v2 models as the schema contract. Input validation happens here, not in route handlers.

**Why SQLite over PostgreSQL:**
- Zero-config: no separate database container in `docker-compose.yml`
- File-based: persists across restarts, trivial to inspect with any SQLite browser
- Sufficient for single-store or few-store deployment (the challenge has 2 stores)
- Production upgrade path: swap `Storage` implementation to asyncpg + PostgreSQL. The `AnalyticsService` doesn't know or care about the storage engine.

**Why not Redis/Kafka for event streaming:**
- At 2 stores with batch ingest, the event volume is ~5002000 events/clip. SQLite handles this trivially.
- Adding Kafka/Redis to `docker-compose.yml` increases the acceptance gate risk (more things to fail on `docker compose up`)
- The challenge evaluates whether the system *works*, not whether it could theoretically scale. The DESIGN.md documents the upgrade path for 40-store deployment.
## 4) Libraries and Tooling Choices

- **Ultralytics (YOLOv8)**: Chosen for its highly optimized, Python-native API. It automatically handles tensor conversions, NMS (Non-Maximum Suppression), and provides a clean bounding-box output matrix.
- **OpenCV (opencv-python)**: The industry standard for video I/O. Used to parse the .mp4 CCTV files frame-by-frame efficiently without loading the entire video into memory.
- **SciPy (scipy.optimize.linear_sum_assignment)**: Used to solve the bipartite matching problem (Hungarian Algorithm) during centroid tracking. It optimally pairs previous object centroids with newly detected centroids based on a distance cost matrix.

## 5) Bounding Box and Centroid-Based Tracking

For retail analytics, tracking individuals across a single camera view is critical. 
1. **Bounding Box Detection**: For every frame, YOLOv8 outputs bounding boxes [x_min, y_min, x_max, y_max] for all detected persons (Class 0).
2. **Centroid Calculation**: We compute the center of mass (centroid) for each bounding box as cx = (x_min + x_max) / 2 and cy = (y_min + y_max) / 2.
3. **Tracking**: We calculate the Euclidean distance between the centroids of existing tracks and new detections. 
4. **Hungarian Algorithm**: SciPy minimizes the total spatial distance across all assignments. If a centroid moves less than the max-distance threshold between frames, it is assigned the same tracking ID.

5. **Zone Assignment (ROI)**: Once a person's centroid is calculated, we check if those (cx, cy) coordinates fall inside any of the custom ROI bounding boxes [x1, y1, x2, y2] defined in your store_layout.json. This instantly tells us which zone the person is standing in! 

This approach is highly lightweight and works perfectly for standard 15fps CCTV footage, eliminating the need for complex, computationally heavy DeepSORT/Kalman filter logic.


## 6) Conversion Ratio Estimation

Initially, the project specification included calculating a store-wide Conversion Ratio. However, due to the unavailability of the required pos_transactions.csv file (Point-of-Sale data), exact conversion correlation was mathematically impossible to calculate with high confidence.

**Future Calculation Method**:
If POS transaction data becomes available in the future, the Conversion Ratio can be calculated by:
1. Counting the total number of unique visitors detected by the ENTRY_THRESHOLD camera.
2. Counting the total number of unique purchase transactions in the POS feed for the identical time window.
3. Ratio = (Total Transactions / Total Unique Visitors) * 100

**Current Heuristic Estimation (CV Only)**:
In the absence of external transaction data, we can estimate purchases purely through computer vision behavior logic. If a tracked person's centroid is registered inside the BILLING zone (the checkout counter) and their continuous dwell time in that specific zone exceeds **5 minutes**, we can confidently infer that a transaction occurred. We can treat these long-dwell billing events as surrogate "Purchases" to calculate an estimated conversion funnel without needing external CSV files.
