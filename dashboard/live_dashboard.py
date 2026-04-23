from __future__ import annotations

import argparse
import time
from urllib.parse import urlencode

import requests
from rich.console import Console
from rich.live import Live
from rich.table import Table


def _fetch_json(url: str, params: dict | None = None) -> dict:
    if params:
        url = f"{url}?{urlencode(params)}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def render_table(api_base: str, store_id: str, as_of: str | None) -> Table:
    params = {"as_of": as_of} if as_of else None
    metrics = _fetch_json(f"{api_base}/stores/{store_id}/metrics", params=params)
    anomalies = _fetch_json(f"{api_base}/stores/{store_id}/anomalies", params=params)

    table = Table(title=f"Purpell Store Live: {store_id}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Unique Visitors", str(metrics["unique_visitors"]))

    table.add_row("Queue Depth", str(metrics["queue_depth"]))
    table.add_row("Abandonment Rate", f'{metrics["abandonment_rate"]:.2f}%')
    table.add_row("Data Confidence", metrics["data_confidence"])
    table.add_row("Active Anomalies", str(len(anomalies.get("anomalies", []))))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal live dashboard")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--store-id", default="STORE_PURPLLE_001")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument(
        "--as-of",
        default="",
        help="Optional ISO-8601 UTC timestamp (e.g. 2026-03-03T16:00:00Z) to view that day's window",
    )
    args = parser.parse_args()
    as_of = args.as_of.strip() or None

    console = Console()
    with Live(
        render_table(args.api_base, args.store_id, as_of), refresh_per_second=2, console=console
    ) as live:
        while True:
            live.update(render_table(args.api_base, args.store_id, as_of))
            time.sleep(args.interval)


if __name__ == "__main__":
    main()

