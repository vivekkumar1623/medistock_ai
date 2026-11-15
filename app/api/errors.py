"""Exception handlers: translate domain errors into the one JSON envelope."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import MedistockError
from app.core.logging import get_logger

logger = get_logger("api.errors")


def _envelope(code: str, message: str, details: dict | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so *every* failure has the same shape."""

    @app.exception_handler(MedistockError)
    async def _handle_domain(request: Request, exc: MedistockError) -> JSONResponse:
        logger.warning(
            "Domain error",
            extra={"path": request.url.path, "code": exc.error_code, "detail_message": exc.message},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.error_code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.info("Request validation failed", extra={"path": request.url.path})
        return JSONResponse(
            status_code=422,
            content=_envelope(
                "validation_error",
                "Request body or parameters failed validation.",
                {"errors": [dict(err) for err in exc.errors()]},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope("http_error", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content=_envelope("internal_error", "An unexpected error occurred."),
        )
