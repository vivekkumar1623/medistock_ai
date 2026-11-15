"""Structured logging configuration.

Two output modes, one call site:
* ``log_json=True``  -> newline-delimited JSON (container/aggregator friendly)
* ``log_json=False`` -> human-readable aligned text (local development)

Every log record is tagged with ``service=medistock-ai`` and the active
environment so mixed log streams stay attributable.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_CONFIGURED = False
_SERVICE_NAME = "medistock-ai"

# Attributes present on every LogRecord; anything else was supplied via extra=.
_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single-line JSON object."""

    def __init__(self, environment: str) -> None:
        super().__init__()
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": _SERVICE_NAME,
            "environment": self.environment,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Compact, aligned text format for local development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )


def configure_logging(
    level: str = "INFO", json_logs: bool = False, environment: str = "development"
) -> None:
    """Install the root handler. Idempotent: safe to call on every startup."""
    global _CONFIGURED
    root = logging.getLogger()

    if _CONFIGURED:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter(environment) if json_logs else TextFormatter())

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Third-party loggers that are too chatty at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``medistock`` hierarchy."""
    if not name.startswith(_SERVICE_NAME):
        name = f"{_SERVICE_NAME}.{name}"
    return logging.getLogger(name)
