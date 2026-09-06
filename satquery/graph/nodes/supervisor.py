"""Supervisor node: pick the specialist for this turn.

Decision order:
1. If a retry targeted a specific specialist, honour it.
2. Deterministic keyword + modality rules (fast, predictable).
3. LLM classification with ``SUPERVISOR_PROMPT`` as a fallback.
Every choice is checked for feasibility against the images actually provided.
"""

from __future__ import annotations

import logging

from satquery.graph.llm import invoke_text
from satquery.graph.nodes._common import append_result, has_any_image, trace
from satquery.graph.prompts import SUPERVISOR_PROMPT
from satquery.graph.state import SatQueryState

logger = logging.getLogger(__name__)

SPECIALISTS = ("image_analysis", "change_detection", "cross_modal", "geo_spatial", "retrieval")

_RETRY_MAP = {
    "IMAGE_ANALYSIS": "image_analysis",
    "CHANGE_DETECTION": "change_detection",
    "CROSS_MODAL": "cross_modal",
    "GEO_SPATIAL": "geo_spatial",
    "RETRIEVAL": "retrieval",
    "NONE": "image_analysis",
}

_CHANGE_KW = (
    "change", "changed", "difference", "differ", "before and after", "before/after",
    "temporal", "time series", "time-series", "what happened",
)
_CROSS_KW = (
    "optical and sar", "sar and optical", "optical + sar", "optical/sar",
    "cross modal", "cross-modal", "both modalities", "compare modalities",
    "multimodal", "multi-modal",
)
_GEO_KW = (
    "coordinate", "crs", "spatial bound", "geographic bound", "bounding box",
    "footprint", "distance", "resolution", "geospatial", "geo-spatial",
    "how large", "how big", "square", "hectare", "area of",
)


def _has_t1_t2(state: SatQueryState) -> bool:
    return state.get("image_t1") is not None and state.get("image_t2") is not None


def _has_optical_sar(state: SatQueryState) -> bool:
    return state.get("optical_image") is not None and state.get("sar_image") is not None


def _feasible(task: str, state: SatQueryState) -> str:
    if task == "change_detection" and not _has_t1_t2(state):
        return "cross_modal" if _has_optical_sar(state) else "image_analysis"
    if task == "cross_modal" and not _has_optical_sar(state):
        return "image_analysis"
    if task == "geo_spatial" and not has_any_image(state):
        return "retrieval"
    if task not in SPECIALISTS:
        return "image_analysis"
    return task


def _llm_classify(state: SatQueryState) -> str:
    prompt = (
        f"{SUPERVISOR_PROMPT}\n\n"
        f"User query:\n{state.get('query', '')}\n\n"
        f"Image count: {state.get('image_count', 0)}\n"
        f"Modalities: {state.get('modalities', [])}\n"
        f"Temporal mode: {state.get('temporal_mode', 'single')}\n\n"
        "Return ONLY the task name."
    )
    try:
        text = invoke_text(prompt).upper()
    except Exception as exc:  # noqa: BLE001 - classification must never crash the graph
        logger.warning("Supervisor LLM classification failed: %s", exc)
        return "image_analysis"

    for key, task in (
        ("CHANGE_DETECTION", "change_detection"),
        ("CROSS_MODAL", "cross_modal"),
        ("GEO_SPATIAL", "geo_spatial"),
        ("RETRIEVAL", "retrieval"),
        ("IMAGE_ANALYSIS", "image_analysis"),
    ):
        if key in text:
            return task
    return "image_analysis"


def classify_task(state: SatQueryState) -> str:
    query = (state.get("query") or "").lower()

    if _has_t1_t2(state) and any(k in query for k in _CHANGE_KW):
        return "change_detection"
    if _has_optical_sar(state) and (
        any(k in query for k in _CROSS_KW) or state.get("temporal_mode") == "cross_modal"
    ):
        return "cross_modal"
    if any(k in query for k in _GEO_KW):
        return _feasible("geo_spatial", state)
    return _feasible(_llm_classify(state), state)


def supervisor_node(state: SatQueryState) -> dict:
    retry_task = state.get("retry_task")
    if retry_task:
        task = _feasible(_RETRY_MAP.get(str(retry_task).upper(), "image_analysis"), state)
        source = "retry"
    else:
        task = classify_task(state)
        source = "classifier"

    return {
        "current_task": task,
        "agent_results": append_result(
            state, {"agent": "supervisor", "task": task, "raw_output": task, "source": source}
        ),
        "execution_trace": trace(
            state,
            {"node": "supervisor", "status": "completed", "selected_task": task, "source": source},
        ),
    }
