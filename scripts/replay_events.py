from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.detect import replay_to_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay JSONL events to ingest API")
    parser.add_argument("--events", default="data/events.jsonl")
    parser.add_argument("--ingest-url", default="http://localhost:8000/events/ingest")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--speed", type=float, default=10.0)
    args = parser.parse_args()
    replay_to_ingest(
        ingest_url=args.ingest_url,
        events_path=Path(args.events),
        batch_size=args.batch_size,
        realtime=args.realtime,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()

