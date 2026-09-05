"""Conditional-edge functions."""

from __future__ import annotations

from satquery.config import get_settings
from satquery.graph.nodes.supervisor import SPECIALISTS
from satquery.graph.state import SatQueryState


def validation_router(state: SatQueryState) -> str:
    return "ok" if state.get("input_valid", False) else "invalid"


def specialist_router(state: SatQueryState) -> str:
    task = state.get("current_task") or "image_analysis"
    return task if task in SPECIALISTS else "image_analysis"


def reflection_router(state: SatQueryState) -> str:
    reflection = state.get("reflection") or {}
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries")
    if max_retries is None:
        max_retries = get_settings().default_max_retries

    if retry_count >= max_retries:
        return "validated"
    if reflection.get("decision") == "NEEDS_ANALYSIS":
        return "retry"
    return "validated"
