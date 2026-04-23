#!/usr/bin/env bash
set -euo pipefail

echo "=== Purpell Store Detection Pipeline ==="
echo "Processing CCTV clips with YOLOv8..."

python -m pipeline.detect \
  --clips-dir "CCTV Footage" \
  --store-layout data/sample/store_layout.json \
  --output data/events.jsonl \
  --mode auto \
  --frame-skip 2 \
  --confidence-threshold 0.30

echo ""
echo "Replaying events into API..."

python -m scripts.replay_events \
  --events data/events.jsonl \
  --ingest-url "http://localhost:8000/events/ingest" \
  --realtime \
  --speed 15

