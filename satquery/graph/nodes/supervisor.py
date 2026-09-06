"""Supervisor node: pick the specialist for this turn.

Routing is driven by the user's **query intent** and the **number of images**
supplied — never by which upload field (`optical` / `sar` / `image_t1` /
`image_t2`) was used. Those labels are optional hints that only tell a
specialist which image is which; a client can just send `images[]` and let the
supervisor work out what to run.

Decision order:
1. A retry that targeted a specific specialist wins.
2. Unambiguous query intent + enough images (e.g. "what changed" + >= 2 images).
3. Structural hints from labelled uploads (t1+t2, or optical+sar).
4. LLM classification, then bounded to what the image count allows.
"""

from __future__ import annotations

import logging

from satquery.graph.llm import invoke_text
from satquery.graph.nodes._common import append_result, trace
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
    "temporal", "time series", "time-series", "what happened", "grew", "expansion",
    "deforestation", "construction over time",
)
_CROSS_KW = (
    "optical and sar", "sar and optical", "optical + sar", "optical/sar",
    "cross modal", "cross-modal", "both modalities", "compare modalities",
    "multimodal", "multi-modal", "radar and optical", "optical vs sar",
)
_GEO_KW = (
    "coordinate", "crs", "spatial bound", "geographic bound", "bounding box",
    "footprint", "distance", "resolution", "geospatial", "geo-spatial",
    "how large", "how big", "square", "hectare", "area of", "pixel size",
    "ground sample", "extent", "projection",
)


def _n_images(state: SatQueryState) -> int:
    return int(state.get("image_count", 0) or 0)


def _has_t1_t2(state: SatQueryState) -> bool:
    return state.get("image_t1") is not None and state.get("image_t2") is not None


def _has_optical_sar(state: SatQueryState) -> bool:
    return state.get("optical_image") is not None and state.get("sar_image") is not None


def _feasible(task: str, state: SatQueryState) -> str:
    """Downgrade a task only when the *image count* makes it impossible."""
    n = _n_images(state)
    if task in ("change_detection", "cross_modal") and n < 2:
        return "image_analysis" if n >= 1 else "retrieval"
    if task in ("image_analysis", "geo_spatial") and n < 1:
        return "retrieval"
    if task not in SPECIALISTS:
        return "image_analysis" if n >= 1 else "retrieval"
    return task


def _image_summary(state: SatQueryState) -> str:
    n = _n_images(state)
    if n == 0:
        return "no images supplied"
    labels = []
    if state.get("optical_image") is not None:
        labels.append("optical")
    if state.get("sar_image") is not None:
        labels.append("SAR")
    if state.get("image_t1") is not None:
        labels.append("earlier/T1")
    if state.get("image_t2") is not None:
        labels.append("later/T2")
    if labels:
        return f"{n} image(s) supplied; labelled: {', '.join(labels)}"
    return f"{n} image(s) supplied (unlabelled)"


def _llm_classify(state: SatQueryState) -> str:
    prompt = (
        f"{SUPERVISOR_PROMPT}\n\n"
        f"User query:\n{state.get('query', '')}\n\n"
        f"Images: {_image_summary(state)}\n\n"
        "Choose the single most appropriate task for this query and this set of "
        "images. Return ONLY the task name."
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
    n = _n_images(state)

    wants_change = any(k in query for k in _CHANGE_KW)
    wants_cross = any(k in query for k in _CROSS_KW)
    wants_geo = any(k in query for k in _GEO_KW)

    # 1. Explicit, unambiguous intent — only if the images can support it.
    if wants_change and n >= 2:
        return "change_detection"
    if wants_cross and n >= 2:
        return "cross_modal"
    if wants_geo and n >= 1:
        return "geo_spatial"

    # 2. Structural hint from labelled uploads (client told us what the pair is).
    if _has_t1_t2(state) and not wants_cross and not wants_geo:
        return "change_detection"
    if _has_optical_sar(state) and not wants_change and not wants_geo:
        return "cross_modal"

    # 3. Let the model decide, then bound it to what the image count allows.
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
