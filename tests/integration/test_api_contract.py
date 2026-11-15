"""API contract tests.

These assert the *published contract*, not implementation details: status
codes, the error envelope, and the OpenAPI surface. A stub today must have the
same route + shape as the real endpoint in a later phase, which is what makes
the stub-first strategy safe.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.deps import store_dep
from app.main import create_app
from app.services.artifact_store import ArtifactStore
from tests.factories import build_fake_bundle


def _client(tmp_path, monkeypatch, *, with_bundle: bool) -> Iterator[TestClient]:
    artifact_dir = tmp_path / "artifacts"
    if with_bundle:
        build_fake_bundle(artifact_dir / "v1")

    monkeypatch.setenv("MEDITSTOCK_ARTIFACT_DIR", str(artifact_dir))
    monkeypatch.setenv("MEDITSTOCK_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEDITSTOCK_ENVIRONMENT", "test")
    # Materialise the SQLite file so the database component can report "ok";
    # otherwise /health is legitimately "degraded" and this fixture would be
    # testing a half-initialised deployment.
    db_path = tmp_path / "medistock.db"
    db_path.touch()
    monkeypatch.setenv("MEDITSTOCK_DB_URL", f"sqlite:///{db_path.as_posix()}")
    from app.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    app = create_app()
    store = ArtifactStore(settings=settings, version="v1")
    app.dependency_overrides[store_dep] = lambda: store
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def client_without_bundle(tmp_path, monkeypatch) -> Iterator[TestClient]:
    yield from _client(tmp_path, monkeypatch, with_bundle=False)


@pytest.fixture
def client_with_bundle(tmp_path, monkeypatch) -> Iterator[TestClient]:
    yield from _client(tmp_path, monkeypatch, with_bundle=True)


def test_health_is_200_and_degraded_without_artifacts(client_without_bundle: TestClient) -> None:
    response = client_without_bundle.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    names = {component["name"] for component in body["components"]}
    assert {"api", "artifacts", "database"} <= names


def test_health_is_ok_with_verified_bundle(client_with_bundle: TestClient) -> None:
    body = client_with_bundle.get("/health").json()
    assert body["status"] == "ok"
    artifacts = next(c for c in body["components"] if c["name"] == "artifacts")
    assert artifacts["status"] == "ok"


def test_error_envelope_shape(client_without_bundle: TestClient) -> None:
    response = client_without_bundle.get("/api/v1/model/info")
    assert response.status_code == 503
    body = response.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == "artifact_not_ready"
    assert body["error"]["message"]
    assert isinstance(body["error"]["details"], dict)


def test_model_info_serves_manifest_when_bundle_present(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.get("/api/v1/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["selected_model"] == "hist_gradient_boosting"
    assert body["horizon_days"] == 30
    assert body["n_features"] == 6
    assert body["dataset"]["is_synthetic"] is True


def test_model_metrics_serves_demand_expiry_and_importance(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.get("/api/v1/model/metrics")
    assert response.status_code == 200
    body = response.json()
    assert len(body["demand_metrics"]) == 2
    assert body["expiry_metrics"]["confusion"]["true_positive"] == 34
    assert [item["rank"] for item in body["feature_importance"]] == [1, 2, 3]


def test_products_endpoint_serves_the_artifact_catalogue(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.get("/api/v1/products")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert {item["product_id"] for item in body["items"]} == {"MED-0001", "MED-0002"}
    assert body["items"][0]["category"] in {"Antibiotics", "Antihistamines"}


def test_forecast_endpoint_runs_the_loaded_model(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.get("/api/v1/forecast/MED-0001")
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_demand"] > 0
    assert body["model_name"] == "hist_gradient_boosting"
    assert body["horizon_days"] == 30
    assert body["artifact_version"] == "v1"
    assert len(body["recent_history"]) == 60
    assert body["prediction_interval"]["method"] == "empirical_test_residual_quantiles_10_90"
    assert (
        body["prediction_interval"]["lower"]
        <= body["predicted_demand"]
        <= body["prediction_interval"]["upper"]
    )


def test_horizon_mismatch_is_rejected_with_an_explanation(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.get("/api/v1/forecast/MED-0001?horizon_days=14")
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["details"]["trained_horizon_days"] == 30


def test_unknown_product_is_404(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.get("/api/v1/forecast/MED-9999")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_inventory_overview_computes_coverage(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.get("/api/v1/inventory/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["total_products"] == 2
    assert body["total_units_on_hand"] == 165
    assert len(body["items"]) == 2
    for item in body["items"]:
        assert item["stock_status"] in {"critical", "below_reorder", "adequate", "excess"}


def test_expiry_risk_endpoint_returns_ranked_batches(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.get("/api/v1/expiry-risk")
    assert response.status_code == 200
    body = response.json()
    assert body["total_batches"] == 3
    assert body["as_of_date"] == "2025-11-01"
    scores = [item["risk_score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    first = body["items"][0]
    assert {"batch_id", "days_to_expiry", "estimated_unsold_units", "reasons"} <= set(first)


def test_recommendations_cover_every_product(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.get("/api/v1/recommendations")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    for item in body["items"]:
        assert item["action"] in {
            "reorder",
            "sufficient",
            "overstock",
            "expiry_exposure",
            "urgent_rotation",
            "monitor",
        }
        assert item["reasons"], "every recommendation must carry a reason"
        assert "safety_stock" in item["inputs"]


def test_endpoints_requiring_a_model_are_503_without_a_bundle(
    client_without_bundle: TestClient,
) -> None:
    for path in (
        "/api/v1/forecast/MED-0001",
        "/api/v1/inventory/overview",
        "/api/v1/expiry-risk",
        "/api/v1/recommendations",
        "/api/v1/products",
    ):
        response = client_without_bundle.get(path)
        assert response.status_code == 503, path
        assert response.json()["error"]["code"] == "artifact_not_ready"


def test_request_validation_uses_same_envelope(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.post(
        "/api/v1/forecast", json={"product_id": "MED-0001", "horizon_days": 999}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"]["errors"]


def test_openapi_documents_the_full_contract(client_with_bundle: TestClient) -> None:
    schema = client_with_bundle.get("/openapi.json").json()
    paths = set(schema["paths"])
    expected = {
        "/health",
        "/api/v1/products",
        "/api/v1/inventory/overview",
        "/api/v1/forecast/{product_id}",
        "/api/v1/forecast",
        "/api/v1/expiry-risk",
        "/api/v1/expiry-risk/{batch_id}",
        "/api/v1/recommendations",
        "/api/v1/model/info",
        "/api/v1/model/metrics",
    }
    assert expected <= paths


def test_unknown_route_uses_envelope(client_with_bundle: TestClient) -> None:
    response = client_with_bundle.get("/api/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_error"
