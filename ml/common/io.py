"""Deterministic I/O and hashing helpers.

Two rules enforced here, because both are easy to get wrong once and then carry
for the rest of the project:

1. **Writes are canonical.** CSVs are written with ``\\n`` line endings, a fixed
   float format, and a stable column order. Two runs with the same seed must
   produce byte-identical files, otherwise the reproducibility test is theatre.
2. **Hashes describe content, not paths.** `sha256_file` is the same function the
   artifact store uses, so "the dataset I trained on" and "the bundle I serve"
   are verified with one implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

FLOAT_FORMAT = "%.6f"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> Path:
    """Write a canonical CSV: ordered columns, LF endings, fixed float format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[columns] if columns else frame
    out.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format=FLOAT_FORMAT,
        na_rep="",
        encoding="utf-8",
    )
    return path


def write_parquet(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[columns] if columns else frame
    out.to_parquet(path, index=False, compression="snappy")
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Expected dataset not found: {path}")
    return pd.read_parquet(path)


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, default=str), encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_fingerprints(paths: list[Path], root: Path) -> dict[str, dict[str, Any]]:
    """Content fingerprint for every file, relative to ``root``."""
    return {
        str(path.relative_to(root)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    }
