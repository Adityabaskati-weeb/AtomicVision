from __future__ import annotations

from fastapi.testclient import TestClient

from atomicvision_env.server.app import app


def test_space_root_route_has_browser_landing_page() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AtomicVision | AI Defect Mapping Platform" in response.text
    assert "ATOMICVISION" in response.text
    assert "Autonomous AI for Non-Destructive Multi-Defect Mapping" in response.text
    assert "/static/space-ui.css" in response.text
    assert "/static/space-ui.js" in response.text
    assert "/static/media/Atoms_Move_Slowly_Around_Circle.mp4" in response.text
    assert "/reset" in response.text
    assert "/step" in response.text
    assert "/ws" in response.text


def test_openenv_health_route_for_space() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"healthy", "ok"}


def test_space_static_assets_are_served() -> None:
    client = TestClient(app)

    css_response = client.get("/static/space-ui.css")
    js_response = client.get("/static/space-ui.js")

    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert "Neue Machina" in css_response.text

    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]
    assert "setupHeroVideoScrub" in js_response.text
    assert "openDemoSocket" in js_response.text


def test_space_live_demo_websocket_reaches_prior_prediction() -> None:
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "reset", "data": {"difficulty": "medium"}})
        reset_payload = websocket.receive_json()
        assert reset_payload["type"] == "observation"
        assert reset_payload["data"]["observation"]["step_count"] == 0

        websocket.send_json({"type": "step", "data": {"action_type": "ask_prior"}})
        prior_payload = websocket.receive_json()
        assert prior_payload["type"] == "observation"
        prior = prior_payload["data"]["observation"]["prior_prediction"]
        assert prior is not None
        assert prior["predicted_defects"]
