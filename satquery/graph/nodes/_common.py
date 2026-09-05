"""Small helpers shared by the graph nodes."""

from __future__ import annotations

import json
from typing import Any

from satquery.graph.state import SatQueryState

_IMAGE_SLOTS = ("optical_image", "sar_image", "image_t1", "image_t2")


def trace(state: SatQueryState, entry: dict[str, Any]) -> list[Any]:
    """Return the execution trace with ``entry`` appended (non-mutating)."""
    return list(state.get("execution_trace", [])) + [entry]


def append_result(state: SatQueryState, result: dict[str, Any]) -> list[Any]:
    return list(state.get("agent_results", [])) + [result]


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, default=str)
    return str(value)


def first_image(state: SatQueryState) -> Any:
    for key in _IMAGE_SLOTS:
        if state.get(key):
            return state[key]
    images = state.get("images") or []
    return images[0] if images else None


def has_any_image(state: SatQueryState) -> bool:
    return first_image(state) is not None


def next_evidence_id(state: SatQueryState, prefix: str) -> str:
    return f"{prefix}_{len(state.get('agent_results', [])) + 1}"
