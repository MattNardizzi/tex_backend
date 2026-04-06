from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from typing import Any, Iterator, Mapping
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send


_REQUEST_ID_CTX: ContextVar[str | None] = ContextVar("tex_request_id", default=None)
_DECISION_ID_CTX: ContextVar[str | None] = ContextVar("tex_decision_id", default=None)
_POLICY_VERSION_CTX: ContextVar[str | None] = ContextVar(
    "tex_policy_version",
    default=None,
)

_DEFAULT_SERVICE_NAME = "tex"
_DEFAULT_LOGGER_NAME = "tex"
_DEFAULT_LOG_LEVEL = logging.INFO
_DEFAULT_MAX_JSON_DEPTH = 8


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """
    Lightweight in-process telemetry state for local visibility.

    Important limitation:
    - this state is process-local only
    - it resets on restart
    - it does not aggregate across multiple workers / containers

    That is acceptable for local development and early product stages, but it
    must not be mistaken for durable metrics infrastructure.
    """

    process_started_at: datetime
    requests_total: int
    requests_failed: int
    evaluations_total: int
    outcomes_total: int

    @property
    def uptime_seconds(self) -> float:
        return max(
            0.0,
            (datetime.now(UTC) - self.process_started_at).total_seconds(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "process_started_at": self.process_started_at.isoformat(),
            "uptime_seconds": round(self.uptime_seconds, 3),
            "requests_total": self.requests_total,
            "requests_failed": self.requests_failed,
            "evaluations_total": self.evaluations_total,
            "outcomes_total": self.outcomes_total,
        }


class TelemetryState:
    """
    Thread-safe mutable in-process counters.

    This is intentionally simple and process-local. It is not a replacement for
    a real metrics backend.
    """

    __slots__ = (
        "_lock",
        "_process_started_at",
        "_requests_total",
        "_requests_failed",
        "_evaluations_total",
        "_outcomes_total",
    )

    def __init__(self) -> None:
        self._lock = RLock()
        self._process_started_at = datetime.now(UTC)
        self._requests_total = 0
        self._requests_failed = 0
        self._evaluations_total = 0
        self._outcomes_total = 0

    def record_request(self, *, failed: bool) -> None:
        with self._lock:
            self._requests_total += 1
            if failed:
                self._requests_failed += 1

    def record_evaluation(self) -> None:
        with self._lock:
            self._evaluations_total += 1

    def record_outcome(self) -> None:
        with self._lock:
            self._outcomes_total += 1

    def snapshot(self) -> TelemetrySnapshot:
        with self._lock:
            return TelemetrySnapshot(
                process_started_at=self._process_started_at,
                requests_total=self._requests_total,
                requests_failed=self._requests_failed,
                evaluations_total=self._evaluations_total,
                outcomes_total=self._outcomes_total,
            )


_STATE = TelemetryState()


class JsonLogFormatter(logging.Formatter):
    """
    Minimal structured JSON log formatter.

    This keeps Tex logs machine-readable without adding another dependency.
    If Tex later adopts structlog, this formatter can be removed cleanly.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": _DEFAULT_SERVICE_NAME,
            "message": record.getMessage(),
            "request_id": get_request_id(),
            "decision_id": get_decision_id(),
            "policy_version": get_policy_version(),
        }

        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event

        fields = getattr(record, "fields", None)
        if isinstance(fields, Mapping):
            payload.update(_coerce_jsonable_mapping(fields))

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(_drop_none_values(payload), ensure_ascii=False, sort_keys=True)


def configure_logging(
    *,
    logger_name: str = _DEFAULT_LOGGER_NAME,
    level: int = _DEFAULT_LOG_LEVEL,
) -> logging.Logger:
    """
    Configure Tex structured logging.

    Safe to call repeatedly. Existing handlers are replaced to avoid duplicate
    log output during reload-heavy local development.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonLogFormatter())

    logger.handlers.clear()
    logger.addHandler(handler)

    return logger


def get_logger(name: str = _DEFAULT_LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        return configure_logging(logger_name=name)
    return logger


def get_request_id() -> str | None:
    return _REQUEST_ID_CTX.get()


def get_decision_id() -> str | None:
    return _DECISION_ID_CTX.get()


def get_policy_version() -> str | None:
    return _POLICY_VERSION_CTX.get()


@contextmanager
def bind_request_id(request_id: str | None) -> Iterator[None]:
    token: Token[str | None] = _REQUEST_ID_CTX.set(_normalize_optional_string(request_id))
    try:
        yield
    finally:
        _REQUEST_ID_CTX.reset(token)


@contextmanager
def bind_decision_id(decision_id: str | None) -> Iterator[None]:
    token: Token[str | None] = _DECISION_ID_CTX.set(
        _normalize_optional_string(decision_id)
    )
    try:
        yield
    finally:
        _DECISION_ID_CTX.reset(token)


@contextmanager
def bind_policy_version(policy_version: str | None) -> Iterator[None]:
    token: Token[str | None] = _POLICY_VERSION_CTX.set(
        _normalize_optional_string(policy_version)
    )
    try:
        yield
    finally:
        _POLICY_VERSION_CTX.reset(token)


@contextmanager
def bind_telemetry_context(
    *,
    request_id: str | None = None,
    decision_id: str | None = None,
    policy_version: str | None = None,
) -> Iterator[None]:
    request_token: Token[str | None] = _REQUEST_ID_CTX.set(
        _normalize_optional_string(request_id)
    )
    decision_token: Token[str | None] = _DECISION_ID_CTX.set(
        _normalize_optional_string(decision_id)
    )
    policy_token: Token[str | None] = _POLICY_VERSION_CTX.set(
        _normalize_optional_string(policy_version)
    )
    try:
        yield
    finally:
        _POLICY_VERSION_CTX.reset(policy_token)
        _DECISION_ID_CTX.reset(decision_token)
        _REQUEST_ID_CTX.reset(request_token)


def emit_event(
    event: str,
    *,
    level: int = logging.INFO,
    logger: logging.Logger | None = None,
    message: str | None = None,
    **fields: Any,
) -> None:
    """
    Emit one structured telemetry event.
    """
    resolved_event = _normalize_required_string(event, field_name="event")
    resolved_message = (
        _normalize_optional_string(message) or resolved_event.replace(".", " ")
    )
    resolved_logger = logger or get_logger()

    resolved_logger.log(
        level,
        resolved_message,
        extra={
            "event": resolved_event,
            "fields": _coerce_jsonable_mapping(fields),
        },
    )


def mark_evaluation_recorded() -> None:
    _STATE.record_evaluation()


def mark_outcome_recorded() -> None:
    _STATE.record_outcome()


def telemetry_snapshot() -> dict[str, Any]:
    return _STATE.snapshot().to_dict()


class TelemetryMiddleware:
    """
    ASGI middleware for request-scoped telemetry.

    Responsibilities:
    - bind / propagate request_id
    - measure request duration
    - emit one completion event per request
    - increment in-process request counters
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        logger: logging.Logger | None = None,
        request_id_header: str = "x-request-id",
    ) -> None:
        self._app = app
        self._logger = logger or get_logger()
        self._request_id_header = request_id_header.lower().encode("latin-1")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = self._extract_request_id(scope) or self._generate_request_id()
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "")
        start = time.perf_counter()

        status_code: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code

            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = list(message.get("headers", []))
                headers.append((self._request_id_header, request_id.encode("latin-1")))
                message["headers"] = headers

            await send(message)

        failed = False

        with bind_request_id(request_id):
            try:
                await self._app(scope, receive, send_wrapper)
            except Exception:
                failed = True
                duration_ms = round((time.perf_counter() - start) * 1000.0, 3)
                _STATE.record_request(failed=True)
                emit_event(
                    "http.request.failed",
                    level=logging.ERROR,
                    logger=self._logger,
                    method=method,
                    path=path,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )
                raise

            duration_ms = round((time.perf_counter() - start) * 1000.0, 3)
            failed = bool(status_code is not None and status_code >= 500)
            _STATE.record_request(failed=failed)

            emit_event(
                "http.request.completed",
                level=logging.INFO if not failed else logging.ERROR,
                logger=self._logger,
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
            )

    def _extract_request_id(self, scope: Scope) -> str | None:
        for key, value in scope.get("headers", []):
            if key.lower() == self._request_id_header:
                decoded = value.decode("latin-1").strip()
                return decoded or None
        return None

    @staticmethod
    def _generate_request_id() -> str:
        return str(uuid4())


def instrument_app(app: Any, *, logger: logging.Logger | None = None) -> Any:
    """
    Attach telemetry middleware and publish telemetry helpers onto app.state.
    """
    resolved_logger = logger or get_logger()
    app.add_middleware(TelemetryMiddleware, logger=resolved_logger)

    if not hasattr(app, "state"):
        return app

    app.state.telemetry_logger = resolved_logger
    app.state.telemetry_snapshot = telemetry_snapshot
    app.state.emit_telemetry_event = emit_event

    return app


def _normalize_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be blank")
    return normalized


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    normalized = value.strip()
    return normalized or None


def _coerce_jsonable_mapping(
    value: Mapping[str, Any],
    *,
    max_depth: int = _DEFAULT_MAX_JSON_DEPTH,
) -> dict[str, Any]:
    return {
        str(key): _coerce_jsonable(item, max_depth=max_depth, _depth=0)
        for key, item in value.items()
    }


def _coerce_jsonable(
    value: Any,
    *,
    max_depth: int = _DEFAULT_MAX_JSON_DEPTH,
    _depth: int = 0,
) -> Any:
    if _depth > max_depth:
        return "<max_depth_exceeded>"

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.astimezone(UTC).isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): _coerce_jsonable(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _coerce_jsonable(
                item,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value
        ]

    return str(value)


def _drop_none_values(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}