from __future__ import annotations

from fastapi.testclient import TestClient

from atomicvision_env.server.app import app


def test_openenv_health_route_for_space() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "ok"}
