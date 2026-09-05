"""Run one analysis end to end.

By default the vision model and the Groq LLM are stubbed so this works with no
credentials and no GPU:

    python scripts/smoke_test.py

Use the real models by exporting GROQ_API_KEY (and installing requirements-ml)
then:

    SATQUERY_SMOKE_REAL=1 python scripts/smoke_test.py path/to/optical.png
"""

from __future__ import annotations

import json
import os
import sys
import tempfile


def _install_stubs() -> None:
    from satquery.graph.llm import set_llm_text_override
    from satquery.vlm import set_vlm

    class _VLM:
        name = "smoke-stub"

        def caption(self, *_a, **_k):
            return "FINDING:\nFields and a road are visible.\nCONFIDENCE:\n0.7"

        def compare(self, *_a, **_k):
            return "CHANGE_SUMMARY:\nMinor vegetation change.\nCONFIDENCE:\n0.6"

        def health(self):
            return {"backend": self.name, "loaded": True}

    def _text(prompt: str, _system=None) -> str:
        up = prompt.upper()
        if "DECISION:" in up or "SELF-REFLECTION" in up:
            return "DECISION: VALIDATED\nREQUIRED_ACTION: NONE\nCONFIDENCE: 0.9"
        if "RAW ANALYSIS" in up or "VERIFIER" in up:
            return "VALID_FINDING:\nFields and a road are visible.\nCONFIDENCE:\n0.8"
        if "FINAL ANSWER GENERATOR" in up:
            return "The image shows farmland crossed by a road."
        return "IMAGE_ANALYSIS"

    set_vlm(_VLM())
    set_llm_text_override(_text)


def _sample_image() -> str:
    from PIL import Image

    path = os.path.join(tempfile.gettempdir(), "satquery_smoke.png")
    Image.new("RGB", (64, 64), (30, 120, 40)).save(path)
    return path


def main() -> None:
    real = os.getenv("SATQUERY_SMOKE_REAL") == "1"
    if not real:
        os.environ.setdefault("GROQ_API_KEY", "smoke")
        _install_stubs()

    from satquery.graph import run_analysis

    image = sys.argv[1] if len(sys.argv) > 1 else _sample_image()
    state = run_analysis(query="Describe the land cover in this image.", optical_image=image)

    print("=" * 70)
    print("FINAL ANSWER:\n", state["final_answer"])
    print("-" * 70)
    print("task:", state.get("current_task"), "| confidence:", state.get("confidence"))
    print("trace:")
    for entry in state["execution_trace"]:
        print("  ", json.dumps(entry, default=str) if isinstance(entry, dict) else entry)


if __name__ == "__main__":
    main()
