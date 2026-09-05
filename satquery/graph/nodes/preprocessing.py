"""Context detection + input validation (graph entry nodes)."""

from __future__ import annotations

from satquery.config import get_settings
from satquery.graph.nodes._common import trace
from satquery.graph.state import SatQueryState

_IMAGE_SLOTS = ("optical_image", "sar_image", "image_t1", "image_t2")


def context_manager(state: SatQueryState) -> dict:
    images = state.get("images") or []

    modalities: list[str] = []
    if state.get("optical_image") is not None:
        modalities.append("optical")
    if state.get("sar_image") is not None:
        modalities.append("sar")

    if state.get("image_t1") is not None and state.get("image_t2") is not None:
        temporal_mode = "bi_temporal"
    elif state.get("optical_image") is not None and state.get("sar_image") is not None:
        temporal_mode = "cross_modal"
    else:
        temporal_mode = "single"

    image_count = len(images) or sum(1 for key in _IMAGE_SLOTS if state.get(key) is not None)

    settings = get_settings()
    return {
        "image_count": image_count,
        "modalities": modalities,
        "temporal_mode": temporal_mode,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", settings.default_max_retries),
        "execution_trace": trace(
            state,
            {
                "node": "context_manager",
                "status": "completed",
                "image_count": image_count,
                "modalities": modalities,
                "temporal_mode": temporal_mode,
            },
        ),
    }


def input_validation(state: SatQueryState) -> dict:
    errors: list[str] = []
    image_count = state.get("image_count", 0)
    modalities = state.get("modalities", [])
    temporal_mode = state.get("temporal_mode", "single")

    if image_count == 0:
        errors.append("No satellite image supplied.")

    if temporal_mode == "bi_temporal":
        if state.get("image_t1") is None:
            errors.append("T1 image is missing.")
        if state.get("image_t2") is None:
            errors.append("T2 image is missing.")

    if "optical" in modalities and "sar" in modalities:
        if state.get("optical_image") is None:
            errors.append("Optical image is missing.")
        if state.get("sar_image") is None:
            errors.append("SAR image is missing.")

    valid = not errors
    return {
        "validation_errors": errors,
        "input_valid": valid,
        "execution_trace": trace(
            state,
            {
                "node": "input_validation",
                "status": "completed",
                "valid": valid,
                "errors": errors,
            },
        ),
    }
