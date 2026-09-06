from __future__ import annotations

from satquery.graph import get_graph, run_analysis


def _trace_nodes(state) -> list[str]:
    return [t.get("node") for t in state["execution_trace"] if isinstance(t, dict)]


def test_image_analysis_flow(png_file):
    state = run_analysis(query="Describe the land cover.", optical_image=png_file("o.png"))

    assert state["input_valid"] is True
    assert state["current_task"] == "image_analysis"
    assert state["final_answer"]
    nodes = _trace_nodes(state)
    assert "context_manager" in nodes
    assert "verification" in nodes
    assert "answer_synthesis" in nodes
    assert state["evidence"]
    assert state["evidence"][-1].get("verified_finding")


def test_validation_short_circuits_without_images():
    state = run_analysis(query="What is in this image?")

    assert state["input_valid"] is False
    assert "could not be analyzed" in state["final_answer"].lower()
    assert state["validation_errors"]
    assert "supervisor" not in _trace_nodes(state)


def test_change_detection_flow(png_file):
    state = run_analysis(
        query="What changed between the two dates?",
        image_t1=png_file("t1.png", color=(10, 100, 30)),
        image_t2=png_file("t2.png", color=(120, 90, 60)),
    )

    assert state["current_task"] == "change_detection"
    assert state["temporal_mode"] == "bi_temporal"
    assert state["final_answer"]
    assert state["evidence"][-1]["task"] == "change_detection"


def test_cross_modal_flow(png_file):
    state = run_analysis(
        query="Compare the optical and SAR images.",
        optical_image=png_file("opt.png"),
        sar_image=png_file("sar.png", color=(80, 80, 80)),
    )

    assert state["current_task"] == "cross_modal"
    assert state["final_answer"]
    latest = state["evidence"][-1]
    assert latest["task"] == "cross_modal_analysis"
    assert latest.get("raw_sar_analysis")


def test_geo_spatial_flow(png_file):
    state = run_analysis(
        query="What are the coordinates and resolution of this raster?",
        optical_image=png_file("geo.png"),
    )

    assert state["current_task"] == "geo_spatial"
    assert state["final_answer"]
    assert state["evidence"][-1]["task"] == "geo_spatial_analysis"


def test_generic_images_route_to_change_detection(png_file):
    """Two images via the unnamed `images` pool + a change query -> change_detection."""
    state = run_analysis(
        query="What changed between these two images?",
        images=[png_file("a.png"), png_file("b.png", color=(120, 90, 60))],
    )
    assert state["image_count"] == 2
    assert state["current_task"] == "change_detection"
    assert state["evidence"][-1]["task"] == "change_detection"
    assert state["final_answer"]


def test_generic_images_route_to_cross_modal(png_file):
    state = run_analysis(
        query="Compare the optical and SAR views of this area.",
        images=[png_file("a.png"), png_file("b.png", color=(80, 80, 80))],
    )
    assert state["current_task"] == "cross_modal"
    assert state["evidence"][-1]["task"] == "cross_modal_analysis"


def test_single_generic_image_routes_to_image_analysis(png_file):
    state = run_analysis(query="Describe the land cover.", images=[png_file("a.png")])
    assert state["current_task"] == "image_analysis"
    assert state["final_answer"]


def test_knowledge_question_without_image_routes_to_retrieval():
    state = run_analysis(query="Explain in general how SAR backscatter relates to surface roughness.")
    assert state["input_valid"] is True
    assert state["current_task"] == "retrieval"
    assert state["final_answer"]


def test_trace_node_names_are_graph_nodes(png_file):
    from satquery.vlm import set_vlm

    class BrokenVLM:
        name = "broken"

        def caption(self, *a, **k):
            raise RuntimeError("boom")

        def compare(self, *a, **k):
            raise RuntimeError("boom")

        def health(self):
            return {}

    set_vlm(BrokenVLM())
    state = run_analysis(
        query="Compare the optical and SAR images.",
        images=[png_file("a.png"), png_file("b.png", color=(80, 80, 80))],
    )
    trace_nodes = {t["node"] for t in state["execution_trace"] if isinstance(t, dict)}
    assert trace_nodes <= set(get_graph().get_graph().nodes), trace_nodes
    assert "cross_modal" in trace_nodes  # not "cross_modal_analysis"


def test_retry_is_bounded(png_file, monkeypatch):
    # Force reflection to always demand more analysis.
    import satquery.graph.nodes.reflection as reflection

    monkeypatch.setattr(
        reflection,
        "invoke_text",
        lambda *a, **k: "DECISION: NEEDS_ANALYSIS\nREQUIRED_ACTION: IMAGE_ANALYSIS\nCONFIDENCE: 0.10",
    )
    state = run_analysis(query="Describe the scene.", optical_image=png_file("o.png"), max_retries=1)

    assert state["retry_count"] <= 1
    assert state["final_answer"]
