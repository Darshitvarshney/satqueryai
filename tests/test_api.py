from __future__ import annotations

import time

from fastapi.testclient import TestClient

from satquery.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


def test_health():
    with _client() as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"


def test_status_reports_config():
    with _client() as client:
        body = client.get("/api/v1/status").json()
        assert body["llm"]["configured"] is True
        assert "image_analysis" in body["graph_nodes"]


def test_analyze_multipart(png_bytes):
    with _client() as client:
        response = client.post(
            "/api/v1/analyze",
            data={"query": "Describe the scene."},
            files={"optical": ("o.png", png_bytes(), "image/png")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["final_answer"]
        assert body["current_task"] == "image_analysis"
        assert body["evidence"]


def test_analyze_requires_an_image():
    with _client() as client:
        response = client.post("/api/v1/analyze", data={"query": "hi"})
        assert response.status_code == 422


def test_analyze_json_base64(png_bytes):
    import base64

    payload = {
        "query": "Describe the scene.",
        "optical_image_b64": base64.b64encode(png_bytes()).decode(),
    }
    with _client() as client:
        response = client.post("/api/v1/analyze/json", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["final_answer"]


def test_jobs_flow(png_bytes):
    with _client() as client:
        created = client.post(
            "/api/v1/jobs",
            data={"query": "Describe."},
            files={"optical": ("o.png", png_bytes(), "image/png")},
        )
        assert created.status_code == 202, created.text
        job_id = created.json()["job_id"]

        detail = {}
        for _ in range(100):
            detail = client.get(f"/api/v1/jobs/{job_id}").json()
            if detail["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.05)

        assert detail["status"] == "succeeded", detail
        assert detail["result"]["final_answer"]

        listing = client.get("/api/v1/jobs").json()
        assert any(item["job_id"] == job_id for item in listing)


def test_api_key_enforced(monkeypatch, png_bytes):
    monkeypatch.setenv("API_KEYS", "secret-a,secret-b")
    from satquery.config import reload_settings

    reload_settings()
    with _client() as client:
        denied = client.post(
            "/api/v1/analyze",
            data={"query": "x"},
            files={"optical": ("o.png", png_bytes(), "image/png")},
        )
        assert denied.status_code == 401

        allowed = client.post(
            "/api/v1/analyze",
            data={"query": "Describe the scene."},
            files={"optical": ("o.png", png_bytes(), "image/png")},
            headers={"X-API-Key": "secret-a"},
        )
        assert allowed.status_code == 200
