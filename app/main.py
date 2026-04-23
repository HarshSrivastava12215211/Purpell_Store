from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.analytics import AnalyticsService
from app.config import DB_PATH, STORE_LAYOUT_PATH
from app.errors import ServiceUnavailableError
from app.logging_utils import configure_logging, log_structured
from app.reference_data import load_store_layout
from app.storage import Storage


logger = configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage = Storage(DB_PATH)
    layout = load_store_layout(STORE_LAYOUT_PATH)

    app.state.storage = storage
    app.state.analytics_service = AnalyticsService(storage=storage, store_layout=layout)
    yield


app = FastAPI(title="Purpell Store API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount web dashboard static files
_dashboard_dir = Path(__file__).resolve().parent.parent / "dashboard" / "web"
if _dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_dir), html=True), name="dashboard")


@app.middleware("http")
async def request_logger(request: Request, call_next: Callable):
    start = perf_counter()
    trace_id = request.headers.get("x-trace-id", str(uuid4()))
    request.state.trace_id = trace_id
    request.state.event_count = getattr(request.state, "event_count", 0)
    try:
        response = await call_next(request)
    except Exception:
        response = JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "Unexpected server error",
                }
            },
        )
    latency_ms = round((perf_counter() - start) * 1000, 2)
    path_parts = [part for part in request.url.path.strip("/").split("/") if part]
    store_id = None
    if len(path_parts) >= 2 and path_parts[0] == "stores":
        store_id = path_parts[1]
    log_structured(
        logger,
        {
            "trace_id": trace_id,
            "store_id": store_id,
            "endpoint": request.url.path,
            "latency_ms": latency_ms,
            "event_count": request.state.event_count,
            "status_code": response.status_code,
        },
    )
    response.headers["x-trace-id"] = trace_id
    return response


@app.exception_handler(ServiceUnavailableError)
async def handle_service_unavailable(_: Request, exc: ServiceUnavailableError):
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "code": "SERVICE_UNAVAILABLE",
                "message": str(exc),
            }
        },
    )


@app.exception_handler(ValueError)
async def handle_value_error(_: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "BAD_REQUEST",
                "message": str(exc),
            }
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "store-intelligence", "status": "ok"}


@app.post("/events/ingest")
async def ingest_events(request: Request, payload: list[dict[str, Any]]):
    request.state.event_count = len(payload)
    service: AnalyticsService = request.app.state.analytics_service
    result = service.ingest(payload)
    return result.model_dump()


@app.get("/stores/{store_id}/metrics")
async def store_metrics(request: Request, store_id: str, as_of: datetime | None = None):
    service: AnalyticsService = request.app.state.analytics_service
    return service.metrics(store_id=store_id, as_of=as_of).model_dump()


@app.get("/stores/{store_id}/funnel")
async def store_funnel(request: Request, store_id: str, as_of: datetime | None = None):
    service: AnalyticsService = request.app.state.analytics_service
    return service.funnel(store_id=store_id, as_of=as_of).model_dump()


@app.get("/stores/{store_id}/heatmap")
async def store_heatmap(request: Request, store_id: str, as_of: datetime | None = None):
    service: AnalyticsService = request.app.state.analytics_service
    return service.heatmap(store_id=store_id, as_of=as_of).model_dump()


@app.get("/stores/{store_id}/anomalies")
async def store_anomalies(request: Request, store_id: str, as_of: datetime | None = None):
    service: AnalyticsService = request.app.state.analytics_service
    return service.anomalies(store_id=store_id, as_of=as_of).model_dump()


@app.get("/health")
async def health(request: Request):
    service: AnalyticsService = request.app.state.analytics_service
    return service.health().model_dump()



