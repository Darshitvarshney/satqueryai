"""Specialist analysis nodes.

Each produces one entry in ``agent_results`` with a ``finding``. Vision-model
and text-model failures are caught and turned into failed-evidence records so
the pipeline always reaches ``answer_synthesis`` with something to say.
"""

from __future__ import annotations

import logging

from satquery.config import get_settings
from satquery.geospatial.raster import (
    analyze_spatial_properties,
    get_pixel_space_info,
    get_raster_metadata,
)
from satquery.graph.llm import invoke_text
from satquery.graph.nodes._common import append_result, as_text, first_image, next_evidence_id, trace
from satquery.graph.prompts import (
    CROSS_MODAL_REASONING_PROMPT,
    CHANGE_DETECTION_PROMPT,
    IMAGE_ANALYSIS_PROMPT,
    RETRIEVAL_PROMPT,
    SAR_ANALYSIS_PROMPT,
)
from satquery.graph.state import SatQueryState
from satquery.vlm import get_vlm
from satquery.vlm.base import VLMUnavailableError

logger = logging.getLogger(__name__)


def _failed(state: SatQueryState, agent: str, task: str, message: str, reason: str | None = None) -> dict:
    evidence = {
        "evidence_id": f"{task}_error",
        "agent": agent,
        "task": task,
        "status": "failed",
        "finding": message,
        "confidence": 0.0,
        "visual_evidence": [],
    }
    return {
        "agent_results": append_result(state, evidence),
        "execution_trace": trace(
            state, {"node": task, "status": "failed", "reason": reason or message}
        ),
    }


def _completed(state: SatQueryState, node: str, evidence: dict, **trace_extra) -> dict:
    return {
        "agent_results": append_result(state, evidence),
        "execution_trace": trace(
            state, {"node": node, "status": "completed", **trace_extra}
        ),
    }


# --------------------------------------------------------------------------- #
# Single image                                                                 #
# --------------------------------------------------------------------------- #
def image_analysis_node(state: SatQueryState) -> dict:
    image = state.get("optical_image") or state.get("sar_image") or first_image(state)
    if image is None:
        return _failed(state, "image_analysis_agent", "image_analysis", "No image was provided.")

    prompt = f"{IMAGE_ANALYSIS_PROMPT}\n\nUSER QUESTION:\n{state.get('query', '')}"
    try:
        finding = get_vlm().caption(image, prompt, max_new_tokens=300)
    except VLMUnavailableError as exc:
        return _failed(state, "image_analysis_agent", "image_analysis", str(exc), "vlm unavailable")
    except Exception as exc:  # noqa: BLE001
        logger.exception("image_analysis VLM error")
        return _failed(
            state, "image_analysis_agent", "image_analysis",
            f"Vision model error: {exc}", "vlm error",
        )

    evidence = {
        "evidence_id": next_evidence_id(state, "image_analysis"),
        "agent": "image_analysis_agent",
        "task": "image_analysis",
        "model": get_settings().vlm_model_id,
        "finding": finding,
        "confidence": 0.75,
        "visual_evidence": ["source_image"],
        "parameters": {"max_new_tokens": 300},
    }
    return _completed(state, "image_analysis", evidence)


# --------------------------------------------------------------------------- #
# Bi-temporal change                                                           #
# --------------------------------------------------------------------------- #
def change_detection_node(state: SatQueryState) -> dict:
    t1, t2 = state.get("image_t1"), state.get("image_t2")
    if t1 is None or t2 is None:
        return _failed(
            state, "change_detection_agent", "change_detection",
            "Two bi-temporal images (T1 and T2) are required.",
        )
    try:
        finding = get_vlm().compare(
            t1, t2, CHANGE_DETECTION_PROMPT,
            label_a="IMAGE 1 - EARLIER OBSERVATION",
            label_b="IMAGE 2 - LATER OBSERVATION",
            max_new_tokens=220,
        )
    except VLMUnavailableError as exc:
        return _failed(state, "change_detection_agent", "change_detection", str(exc), "vlm unavailable")
    except Exception as exc:  # noqa: BLE001
        logger.exception("change_detection VLM error")
        return _failed(
            state, "change_detection_agent", "change_detection",
            f"Vision model error: {exc}", "vlm error",
        )

    evidence = {
        "evidence_id": next_evidence_id(state, "change_detection"),
        "agent": "change_detection_agent",
        "task": "change_detection",
        "model": get_settings().vlm_model_id,
        "finding": finding,
        "confidence": 0.75,
        "visual_evidence": ["image_t1", "image_t2"],
        "parameters": {"max_new_tokens": 220},
    }
    return _completed(state, "change_detection", evidence)


# --------------------------------------------------------------------------- #
# Optical + SAR                                                                #
# --------------------------------------------------------------------------- #
def cross_modal_agent_node(state: SatQueryState) -> dict:
    optical, sar = state.get("optical_image"), state.get("sar_image")
    if optical is None or sar is None:
        return _failed(
            state, "cross_modal_agent", "cross_modal_analysis",
            "Cross-modal analysis requires both an optical and a SAR image.",
        )

    vlm = get_vlm()
    try:
        optical_result = vlm.caption(optical, IMAGE_ANALYSIS_PROMPT, max_new_tokens=180)
        sar_result = vlm.caption(sar, SAR_ANALYSIS_PROMPT, max_new_tokens=180)
    except VLMUnavailableError as exc:
        return _failed(state, "cross_modal_agent", "cross_modal_analysis", str(exc), "vlm unavailable")
    except Exception as exc:  # noqa: BLE001
        logger.exception("cross_modal VLM error")
        return _failed(
            state, "cross_modal_agent", "cross_modal_analysis",
            f"Vision model error: {exc}", "vlm error",
        )

    reasoning_prompt = CROSS_MODAL_REASONING_PROMPT.format(
        optical_observations=optical_result,
        sar_observations=sar_result,
    )
    try:
        finding = invoke_text(reasoning_prompt)
    except Exception as exc:  # noqa: BLE001
        logger.exception("cross_modal reasoning error")
        finding = (
            f"OPTICAL_FINDINGS:\n{optical_result}\n\n"
            f"SAR_FINDINGS:\n{sar_result}\n\n"
            f"(Cross-modal reasoning model unavailable: {exc})"
        )

    settings = get_settings()
    evidence = {
        "evidence_id": next_evidence_id(state, "cross_modal"),
        "agent": "cross_modal_agent",
        "task": "cross_modal_analysis",
        "model": {"vision": settings.vlm_model_id, "reasoning": settings.groq_model},
        "finding": finding,
        "raw_optical_analysis": optical_result,
        "raw_sar_analysis": sar_result,
        "confidence": 0.75,
        "visual_evidence": ["optical_image", "sar_image"],
        "parameters": {"optical_max_new_tokens": 180, "sar_max_new_tokens": 180},
    }
    return _completed(state, "cross_modal", evidence)


# --------------------------------------------------------------------------- #
# Geospatial                                                                   #
# --------------------------------------------------------------------------- #
def geo_spatial_agent_node(state: SatQueryState) -> dict:
    image = first_image(state)
    if image is None:
        return _failed(
            state, "geo_spatial_agent", "geo_spatial_analysis",
            "No image available for geospatial analysis.",
        )

    metadata = get_raster_metadata(image)
    spatial = analyze_spatial_properties(image)
    pixels = get_pixel_space_info(image)
    georeferenced = bool(spatial.get("georeferenced"))

    evidence = {
        "evidence_id": next_evidence_id(state, "geo_spatial"),
        "agent": "geo_spatial_agent",
        "task": "geo_spatial_analysis",
        "finding": {
            "raster_metadata": metadata,
            "spatial_properties": spatial,
            "pixel_space": pixels,
        },
        "confidence": 1.0 if georeferenced else 0.95,
        "visual_evidence": ["source_raster"],
        "parameters": {"tool": "rasterio"},
    }
    return _completed(state, "geo_spatial", evidence, georeferenced=georeferenced)


# --------------------------------------------------------------------------- #
# Retrieval / domain knowledge                                                 #
# --------------------------------------------------------------------------- #
def retrieval_node(state: SatQueryState) -> dict:
    gathered = [e.get("verified_finding") or e.get("finding") for e in state.get("evidence", [])]
    prompt = (
        f"{RETRIEVAL_PROMPT}\n\n"
        f"USER QUESTION:\n{state.get('query', '')}\n\n"
        f"EVIDENCE GATHERED SO FAR:\n{as_text(gathered)}"
    )
    try:
        text = invoke_text(prompt)
    except Exception as exc:  # noqa: BLE001
        return _failed(
            state, "retrieval_agent", "retrieval",
            f"Knowledge model unavailable: {exc}", "llm error",
        )

    evidence = {
        "evidence_id": next_evidence_id(state, "retrieval"),
        "agent": "retrieval_agent",
        "task": "retrieval",
        "finding": text,
        "confidence": 0.6,
        "visual_evidence": [],
        "parameters": {"source": "domain_knowledge"},
    }
    result = _completed(state, "retrieval", evidence)
    result["rag_context"] = text
    return result
