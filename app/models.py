from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    queue_depth: int | None = None
    sku_zone: str | None = None
    session_seq: int | None = None


class EventIn(BaseModel):
    event_id: str
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: EventType
    timestamp: datetime
    zone_id: str | None = None
    dwell_ms: int = Field(ge=0)
    is_staff: bool = False
    confidence: float = Field(ge=0, le=1)
    metadata: EventMetadata = Field(default_factory=EventMetadata)

    @field_validator("event_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        UUID(value)
        return value

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must include timezone")
        return value.astimezone(timezone.utc)


class EventStored(EventIn):
    metadata: dict[str, Any]


class ErrorItem(BaseModel):
    index: int
    reason: str


class IngestResponse(BaseModel):
    received: int
    inserted: int
    duplicates: int
    rejected: int
    errors: list[ErrorItem]


class MetricsResponse(BaseModel):
    store_id: str
    as_of: datetime
    unique_visitors: int
    avg_dwell_per_zone: dict[str, float]
    queue_depth: int
    abandonment_rate: float
    data_confidence: str


class FunnelStage(BaseModel):
    stage: str
    visitors: int
    drop_off_pct: float


class FunnelResponse(BaseModel):
    store_id: str
    as_of: datetime
    stages: list[FunnelStage]


class HeatmapZone(BaseModel):
    zone_id: str
    visits: int
    avg_dwell_ms: float
    score: float


class HeatmapResponse(BaseModel):
    store_id: str
    as_of: datetime
    data_confidence: str
    zones: list[HeatmapZone]


class AnomalySeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class AnomalyItem(BaseModel):
    anomaly_type: str
    severity: AnomalySeverity
    message: str
    suggested_action: str


class AnomaliesResponse(BaseModel):
    store_id: str
    as_of: datetime
    anomalies: list[AnomalyItem]


class HealthResponse(BaseModel):
    status: str
    generated_at: datetime
    last_event_timestamp_per_store: dict[str, datetime | None]
    stale_feeds: list[str]

