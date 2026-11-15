"""Engine and session management.

SQLite specifics handled here so no other module needs to know:
* ``check_same_thread=False`` is required because FastAPI serves requests on a
  thread pool; SQLAlchemy's pool still serialises per-connection use.
* Foreign keys are OFF by default in SQLite and must be enabled per connection,
  otherwise declared FK constraints are silently unenforced.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


def _engine_kwargs(settings: Settings) -> dict:
    if settings.db_url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {}


def build_engine(settings: Settings | None = None) -> Engine:
    settings = settings or get_settings()
    engine = create_engine(settings.db_url, future=True, **_engine_kwargs(settings))

    if settings.db_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, class_=Session)


def ensure_sqlite_directory(settings: Settings) -> None:
    """SQLite will not create missing parent directories; do it explicitly."""
    if not settings.db_url.startswith("sqlite"):
        return
    raw = settings.db_url.removeprefix("sqlite:///")
    path = Path(raw)
    if not path.is_absolute():
        path = (settings.data_dir.parent / raw).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(settings: Settings | None = None) -> Engine:
    global _engine
    if _engine is None:
        _engine = build_engine(settings)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = build_session_factory(get_engine())
    return _session_factory


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
