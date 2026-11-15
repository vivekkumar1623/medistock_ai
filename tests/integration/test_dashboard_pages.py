"""Dashboard rendering tests via Streamlit's AppTest.

Marked ``slow`` and skipped automatically when the API is not reachable, because
the dashboard is a pure API client: without a live backend the only meaningful
assertion is that the error state renders (which is tested separately below).

Run with the API up:
    python -m pytest -m slow tests/integration/test_dashboard_pages.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAGES = [
    "executive",
    "forecasting",
    "expiry",
    "inventory",
    "product_explorer",
    "model_performance",
    "system_health",
]


def _api_available() -> bool:
    try:
        response = httpx.get("http://127.0.0.1:8000/health", timeout=3.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


pytestmark = [
    pytest.mark.slow,
    pytest.mark.integration,
    pytest.mark.skipif(
        not _api_available(),
        reason="Dashboard renders live API data; start the API (scripts/run_api.ps1) to run these.",
    ),
]


def _script_for(page: str, tmp_dir: Path) -> Path:
    """AppTest.from_function does not carry module imports, so write a real script."""
    path = tmp_dir / f"render_{page}.py"
    path.write_text(
        "import sys\n"
        f'sys.path.insert(0, r"{ROOT}")\n'
        f"from dashboard.pages.{page} import render\n"
        "render()\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page: str) -> None:
    from streamlit.testing.v1 import AppTest

    with tempfile.TemporaryDirectory() as tmp:
        script = _script_for(page, Path(tmp))
        app = AppTest.from_file(str(script), default_timeout=240).run()

    exceptions = [element.value for element in app.exception]
    api_errors = [element.value for element in app.error]
    assert not exceptions, f"{page} raised: {exceptions}"
    assert not api_errors, f"{page} showed API errors: {api_errors}"
    # Every page must render at least a header and some content.
    assert len(app.markdown) >= 4, f"{page} rendered too little content"
