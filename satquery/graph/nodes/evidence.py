"""Evidence pooling + verification.

``evidence_pool_node`` normalises every specialist result. ``verification_node``
runs a strict LLM verifier over the most recent finding (dispatching to the
change / cross-modal / generic verifier prompt) and records a cleaned
``verified_finding`` plus a parsed confidence. Deterministic tool output
(geo-spatial) skips the LLM.
"""

from __future__ import annotations

import logging
import re

from satquery.graph.llm import invoke_text
from satquery.graph.nodes._common import as_text, trace
from satquery.graph.prompts import (
    CHANGE_VERIFICATION_PROMPT,
    CROSS_MODAL_VERIFICATION_PROMPT,
    VERIFICATION_PROMPT,
)
from satquery.graph.state import SatQueryState

logger = logging.getLogger(__name__)

_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE)

# task -> (verifier prompt, default confidence when the model omits one)
_VERIFIERS: dict[str, tuple[str, float]] = {
    "change_detection": (CHANGE_VERIFICATION_PROMPT, 0.5),
    "cross_modal_analysis": (CROSS_MODAL_VERIFICATION_PROMPT, 0.8),
    "cross_modal": (CROSS_MODAL_VERIFICATION_PROMPT, 0.8),
}
_DEFAULT_VERIFIER = (VERIFICATION_PROMPT, 0.75)


def _parse_confidence(text: str, default: float) -> float:
    match = _CONFIDENCE_RE.search(text or "")
    if not match:
        return default
    try:
        return max(0.0, min(1.0, float(match.group(1))))
    except ValueError:  # pragma: no cover - regex already constrains this
        return default


def evidence_pool_node(state: SatQueryState) -> dict:
    items = [r for r in state.get("agent_results", []) if r.get("agent") != "supervisor"]
    evidence = [
        {
            "evidence_id": item.get("evidence_id"),
            "agent": item.get("agent"),
            "task": item.get("task"),
            "finding": item.get("finding"),
            "verified_finding": item.get("verified_finding"),
            "confidence": item.get("confidence", 0.0),
            "status": item.get("status", "ok"),
            "visual_evidence": item.get("visual_evidence", []),
            "parameters": item.get("parameters", {}),
            "raw_optical_analysis": item.get("raw_optical_analysis"),
            "raw_sar_analysis": item.get("raw_sar_analysis"),
        }
        for item in items
    ]
    return {
        "evidence": evidence,
        "execution_trace": trace(
            state, {"node": "evidence_pool", "status": "completed", "count": len(evidence)}
        ),
    }


def verification_node(state: SatQueryState) -> dict:
    evidence = list(state.get("evidence", []))
    if not evidence:
        return {
            "confidence": 0.0,
            "execution_trace": trace(
                state, {"node": "verification", "status": "skipped", "reason": "no evidence"}
            ),
        }

    latest = evidence[-1]
    task = latest.get("task", "image_analysis")

    # Deterministic tool output — trust as-is.
    if task == "geo_spatial_analysis":
        confidence = float(latest.get("confidence", 0.95))
        verified = evidence[:-1] + [
            {
                **latest,
                "verified_finding": as_text(latest.get("finding")),
                "verification_status": "tool_output",
            }
        ]
        return {
            "evidence": verified,
            "confidence": confidence,
            "execution_trace": trace(
                state,
                {
                    "node": "verification",
                    "status": "completed",
                    "task": task,
                    "mode": "tool_output",
                    "confidence": confidence,
                },
            ),
        }

    if latest.get("status") == "failed":
        return {
            "confidence": 0.0,
            "execution_trace": trace(
                state,
                {"node": "verification", "status": "skipped", "reason": "latest evidence failed"},
            ),
        }

    prompt_template, default_conf = _VERIFIERS.get(task, _DEFAULT_VERIFIER)
    raw = as_text(latest.get("finding"))
    try:
        verified_text = invoke_text(f"{prompt_template}\n\nRAW ANALYSIS:\n{raw}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Verification LLM unavailable: %s", exc)
        return {
            "evidence": evidence,
            "confidence": float(latest.get("confidence", default_conf)),
            "execution_trace": trace(
                state, {"node": "verification", "status": "error", "error": str(exc)}
            ),
        }

    confidence = _parse_confidence(verified_text, default_conf)
    verified = evidence[:-1] + [
        {
            **latest,
            "verified_finding": verified_text,
            "verification_status": "verified",
            "confidence": confidence,
        }
    ]
    return {
        "evidence": verified,
        "confidence": confidence,
        "execution_trace": trace(
            state,
            {"node": "verification", "status": "completed", "task": task, "confidence": confidence},
        ),
    }
