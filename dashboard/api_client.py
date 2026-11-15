"""Thin HTTP client for the dashboard.

Rules:
* The dashboard never imports the ML pipeline. It reads the API, so the API and
  the UI cannot disagree about a number.
* Every failure becomes an ``ApiError`` with an actionable message. A dashboard
  that shows a blank chart when the backend is down is worse than one that says
  what is wrong.
* Responses are cached briefly (``ttl``) so page reruns do not hammer the API.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import streamlit as st

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def base_url() -> str:
    return os.environ.get("MEDITSTOCK_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


class ApiError(RuntimeError):
    """Raised for transport failures and non-2xx responses."""

    def __init__(self, message: str, *, status_code: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


def _params_key(params: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not params:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in params.items()))


def _request(path: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> dict:
    url = f"{base_url()}{path}"
    try:
        response = httpx.get(url, params=params, timeout=timeout)
    except httpx.HTTPError as exc:
        raise ApiError(
            f"Cannot reach the MediStock API at {base_url()}. "
            "Start it with  .\\scripts\\run_api.ps1  and reload this page.",
            payload=str(exc),
        ) from exc

    if response.status_code >= 400:
        try:
            body = response.json()
            detail = body.get("error", {})
            message = detail.get("message") or response.text
            code = detail.get("code", "error")
        except ValueError:
            message = response.text or response.reason_phrase
            code = "http_error"
        raise ApiError(
            f"{code}: {message}",
            status_code=response.status_code,
            payload=response.text,
        )
    try:
        return response.json()
    except ValueError as exc:
        raise ApiError("API returned a non-JSON response", payload=response.text) from exc


@st.cache_data(ttl=120, show_spinner=False)
def _cached_get(path: str, params_key: tuple[tuple[str, str], ...]) -> dict:
    params = dict(params_key) if params_key else None
    return _request(path, params)


def get(path: str, params: dict[str, Any] | None = None) -> dict:
    return _cached_get(path, _params_key(params))


@st.cache_data(ttl=15, show_spinner=False)
def health() -> dict:
    return _request("/health", timeout=5.0)


def clear_cache() -> None:
    _cached_get.clear()
    health.clear()
