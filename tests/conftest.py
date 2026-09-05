"""Test wiring: stub the vision model and the Groq LLM so the full graph runs
offline and deterministically."""

from __future__ import annotations

import io

import pytest
from PIL import Image


class FakeVLM:
    name = "fake-vlm"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def caption(self, image, prompt, *, max_new_tokens=None) -> str:
        self.calls.append(("caption", prompt[:60]))
        if "SAR" in prompt:
            return (
                "SAR_OBSERVATIONS:\n- bright angular returns in the north\n"
                "- large smooth low-backscatter area\nUNCERTAINTY:\nSpeckle limits detail.\n"
                "CONFIDENCE:\n0.6"
            )
        return (
            "FINDING:\nAgricultural fields, scattered built-up areas and a river are visible.\n"
            "VISUAL_EVIDENCE:\n- rectangular field parcels\n- bright rooftops\n- dark sinuous channel\n"
            "UNCERTAINTY:\nExact land-cover boundaries are approximate.\nCONFIDENCE:\n0.7"
        )

    def compare(self, a, b, prompt, *, label_a="IMAGE 1", label_b="IMAGE 2", max_new_tokens=None) -> str:
        self.calls.append(("compare", prompt[:60]))
        return (
            "CHANGE_SUMMARY:\nNew construction has appeared in the north.\n"
            "DETECTED_CHANGES:\n- new built-up area\n- cropland converted to bare soil\n"
            "UNCHANGED_FEATURES:\n- river course\nUNCERTAINTY:\nCo-registration differences.\n"
            "CONFIDENCE:\n0.6"
        )

    def health(self) -> dict:
        return {"backend": self.name, "loaded": True}


def fake_llm_text(prompt: str, system: str | None = None) -> str:
    upper = prompt.upper()
    if "RAW ANALYSIS:" in upper or "EVIDENCE VERIFIER" in upper:
        return (
            "VALID_FINDING:\nFields, built-up areas and a watercourse are visible.\n"
            "REJECTED_CLAIMS:\n- none\nCONFIDENCE:\n0.82"
        )
    if "SELF-REFLECTION AGENT" in upper or "DECISION:" in upper:
        return (
            "DECISION: VALIDATED\nREQUIRED_ACTION: NONE\n"
            "REASON: Evidence is sufficient.\nCONFIDENCE: 0.88"
        )
    if "SUPERVISOR AGENT" in upper:
        return "IMAGE_ANALYSIS"
    if "FINAL ANSWER GENERATOR" in upper:
        return "The scene shows farmland with some built-up areas beside a river. Boundaries are approximate."
    if "SENIOR REMOTE-SENSING ANALYST" in upper:
        return (
            "OPTICAL_FINDINGS:\n- fields\nSAR_FINDINGS:\n- bright returns\n"
            "CROSS_MODAL_FINDINGS:\n- built-up areas agree in both\nCONFIDENCE:\n0.7"
        )
    if "KNOWLEDGE ASSISTANT" in upper:
        return "DOMAIN_KNOWLEDGE:\nSAR backscatter rises with surface roughness and moisture.\nCAVEATS:\nGeneral only."
    return "OK"


@pytest.fixture(autouse=True)
def wire_stubs(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("VLM_BACKEND", "local")
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("DEFAULT_MAX_RETRIES", "2")

    from satquery.config import reload_settings
    from satquery.graph import reset_graph
    from satquery.graph.llm import set_llm_text_override
    from satquery.vlm import reset_vlm, set_vlm

    reload_settings()
    reset_graph()
    set_llm_text_override(fake_llm_text)
    set_vlm(FakeVLM())

    yield

    set_llm_text_override(None)
    reset_vlm()
    reset_graph()
    reload_settings()


@pytest.fixture
def png_bytes():
    def _make(color=(20, 120, 40), size=(48, 40)):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format="PNG")
        return buf.getvalue()

    return _make


@pytest.fixture
def png_file(tmp_path, png_bytes):
    def _make(name="img.png", color=(20, 120, 40)):
        path = tmp_path / name
        path.write_bytes(png_bytes(color=color))
        return str(path)

    return _make
