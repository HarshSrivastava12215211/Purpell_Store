from __future__ import annotations

import json
import logging
import sys
from typing import Any


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("store_intelligence")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def log_structured(logger: logging.Logger, payload: dict[str, Any]) -> None:
    logger.info(json.dumps(payload, default=str))

