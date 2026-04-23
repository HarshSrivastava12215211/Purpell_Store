Write-Host "=== Purpell Store Detection Pipeline ==="
Write-Host "Processing CCTV clips with YOLOv8..."

py -m pipeline.detect `
  --clips-dir "CCTV Footage" `
  --store-layout data/sample/store_layout.json `
  --output data/events.jsonl `
  --mode auto `
  --frame-skip 2 `
  --confidence-threshold 0.30

Write-Host ""
Write-Host "Replaying events into API..."

py -m scripts.replay_events `
  --events data/events.jsonl `
  --ingest-url "http://localhost:8000/events/ingest" `
  --realtime `
  --speed 15

