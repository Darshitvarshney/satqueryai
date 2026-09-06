from __future__ import annotations

import json

from fastapi.testclient import TestClient

from satquery.graph import stream_analysis
from satquery.main import create_app


def _parse_sse(text: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    event = None
    data: list[str] = []
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data.append(line.split(":", 1)[1].strip())
        elif line == "":
            if event is not None:
                events.append((event, "\n".join(data)))
            event, data = None, []
    return events


def test_stream_analysis_generator(png_file):
    events = list(stream_analysis(query="Describe the scene.", optical_image=png_file("o.png")))
    types = [e["type"] for e in events]

    assert types[0] == "start"
    assert types[-1] == "result"
    assert "progress" in types

    nodes = [e["entry"].get("node") for e in events if e["type"] == "progress"]
    assert "answer_synthesis" in nodes

    final = events[-1]["state"]
    assert final["final_answer"]
    assert "duration_seconds" in final


def test_stream_endpoint(png_bytes):
    with TestClient(create_app()) as client:
        with client.stream(
            "POST",
            "/api/v1/analyze/stream",
            data={"query": "Describe the scene."},
            files={"optical": ("o.png", png_bytes(), "image/png")},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())

    events = _parse_sse(body)
    names = [name for name, _ in events]
    assert names[0] == "start"
    assert "progress" in names
    assert names[-1] == "result"

    result_payload = json.loads(dict((n, d) for n, d in events)["result"])
    assert result_payload["final_answer"]
    assert result_payload["current_task"] == "image_analysis"


def test_stream_endpoint_requires_image():
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/analyze/stream", data={"query": "hi"})
        assert response.status_code == 422
