"""Self-reflection / quality control + retry bookkeeping (ported from notebook)."""

from __future__ import annotations

import logging
import re

from satquery.config import get_settings
from satquery.graph.llm import invoke_text
from satquery.graph.nodes._common import as_text, trace
from satquery.graph.prompts import REFLECTION_PROMPT
from satquery.graph.state import SatQueryState

logger = logging.getLogger(__name__)

_DECISION_RE = re.compile(r"DECISION:\s*(VALIDATED|NEEDS_ANALYSIS)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(0(?:\.\d+)?|1(?:\.0+)?)", re.IGNORECASE)
_ACTION_RE = re.compile(
    r"REQUIRED_ACTION:\s*(IMAGE_ANALYSIS|CHANGE_DETECTION|CROSS_MODAL|GEO_SPATIAL|RETRIEVAL|NONE)",
    re.IGNORECASE,
)

_CHANGE_WORDS = ("change", "changed", "difference", "differences", "before and after", "temporal", "time series")
_GEO_WORDS = (
    "coordinates", "coordinate", "crs", "spatial bounds", "geographic bounds", "area",
    "distance", "resolution", "geospatial", "geo-spatial",
)


def determine_required_action(state: SatQueryState, reflection_text: str) -> str:
    query = (state.get("query") or "").lower()
    n_images = int(state.get("image_count", 0) or 0)
    all_results = list(state.get("evidence", [])) + list(state.get("agent_results", []))

    optical_sar_requested = (
        ("optical" in query and "sar" in query)
        or "cross-modal" in query
        or "cross modal" in query
        or "both modalities" in query
        or "compare modalities" in query
    )
    cross_modal_present = any(
        item.get("agent") == "cross_modal_agent"
        or item.get("task") in ("cross_modal_analysis", "cross_modal")
        for item in all_results
    )
    if optical_sar_requested and not cross_modal_present and n_images >= 2:
        return "CROSS_MODAL"

    if any(word in query for word in _CHANGE_WORDS) and n_images >= 2:
        return "CHANGE_DETECTION"

    if any(word in query for word in _GEO_WORDS) and n_images >= 1:
        return "GEO_SPATIAL"

    match = _ACTION_RE.search(reflection_text)
    if match:
        return match.group(1).upper()
    return "IMAGE_ANALYSIS" if n_images >= 1 else "RETRIEVAL"


def reflection_node(state: SatQueryState) -> dict:
    settings = get_settings()
    min_confidence = settings.min_confidence
    evidence = state.get("evidence", [])
    agent_results = state.get("agent_results", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.default_max_retries)

    if not evidence and not agent_results:
        return {
            "reflection": {
                "decision": "NEEDS_ANALYSIS",
                "required_action": "IMAGE_ANALYSIS",
                "reason": "No evidence is available.",
                "confidence": 0.0,
                "retry_count": retry_count,
            },
            "execution_trace": trace(
                state, {"node": "reflection", "status": "no_evidence"}
            ),
        }

    latest = evidence[-1] if evidence else agent_results[-1]

    # Repeated specialist failure (e.g. VLM disabled): stop burning retries.
    non_failed = [e for e in evidence if e.get("status") != "failed"]
    if not non_failed and retry_count >= 1:
        reflection = {
            "decision": "VALIDATED",
            "required_action": "NONE",
            "reason": "Specialist analysis failed repeatedly; proceeding with what is available.",
            "confidence": 0.0,
            "retry_count": retry_count,
        }
        return {
            "reflection": reflection,
            "needs_reanalysis": False,
            "execution_trace": trace(
                state,
                {
                    "node": "reflection",
                    "status": "completed",
                    "decision": "VALIDATED",
                    "required_action": "NONE",
                    "reason": "repeated_failure",
                },
            ),
        }

    verified_finding = as_text(latest.get("verified_finding") or latest.get("finding") or "")
    evidence_confidence = latest.get("confidence", state.get("confidence", 0.0))

    prompt = (
        f"{REFLECTION_PROMPT}\n\n"
        f"USER QUESTION:\n{state.get('query', '')}\n\n"
        f"CURRENT VERIFIED EVIDENCE:\n{verified_finding}\n\n"
        f"CURRENT EVIDENCE CONFIDENCE:\n{evidence_confidence}\n\n"
        f"RETRY COUNT:\n{retry_count}\n\n"
        f"MAXIMUM RETRIES:\n{max_retries}\n"
    )
    try:
        reflection_text = invoke_text(prompt).strip()
    except Exception as exc:  # noqa: BLE001
        # Can't judge sufficiency without the model — accept the current evidence
        # rather than burning retries on an arbitrary specialist.
        logger.warning("Reflection LLM unavailable: %s", exc)
        reflection = {
            "decision": "VALIDATED",
            "required_action": "NONE",
            "reason": f"Reflection model unavailable ({exc}); proceeding with current evidence.",
            "confidence": evidence_confidence,
            "retry_count": retry_count,
        }
        return {
            "reflection": reflection,
            "needs_reanalysis": False,
            "execution_trace": trace(
                state,
                {
                    "node": "reflection",
                    "status": "completed",
                    "decision": "VALIDATED",
                    "required_action": "NONE",
                    "reason": "llm_unavailable",
                },
            ),
        }

    decision_match = _DECISION_RE.search(reflection_text)
    decision = decision_match.group(1).upper() if decision_match else "NEEDS_ANALYSIS"

    confidence_match = _CONFIDENCE_RE.search(reflection_text)
    reflection_confidence = float(confidence_match.group(1)) if confidence_match else 0.0

    required_action = determine_required_action(state, reflection_text)

    if (
        decision == "VALIDATED"
        and reflection_confidence < min_confidence
        and retry_count < max_retries
    ):
        # Only retry if there is a concrete specialist to hand off to.
        candidate = required_action
        if candidate == "NONE":
            candidate = determine_required_action(state, reflection_text)
        if candidate != "NONE":
            decision = "NEEDS_ANALYSIS"
            required_action = candidate
            reflection_text += (
                f"\nConfidence {reflection_confidence:.2f} is below minimum threshold "
                f"{min_confidence:.2f}. Additional analysis required."
            )

    if retry_count >= max_retries:
        decision = "VALIDATED"
        required_action = "NONE"
        reflection_text += "\nMaximum retry limit reached. Proceeding with available evidence."

    reflection = {
        "decision": decision,
        "required_action": required_action,
        "reason": reflection_text,
        "confidence": reflection_confidence,
        "evidence_confidence": evidence_confidence,
        "min_confidence": min_confidence,
        "retry_count": retry_count,
    }
    return {
        "reflection": reflection,
        "needs_reanalysis": decision == "NEEDS_ANALYSIS",
        "execution_trace": trace(
            state,
            {
                "node": "reflection",
                "status": "completed",
                "decision": decision,
                "required_action": required_action,
                "confidence": reflection_confidence,
                "retry": f"{retry_count}/{max_retries}",
            },
        ),
    }


def retry_node(state: SatQueryState) -> dict:
    retry_count = state.get("retry_count", 0)
    reflection = state.get("reflection", {})
    required_action = reflection.get("required_action", "IMAGE_ANALYSIS")
    return {
        "retry_count": retry_count + 1,
        "retry_task": required_action,
        "execution_trace": trace(
            state,
            {
                "node": "retry",
                "status": "retrying",
                "target": required_action,
                "attempt": retry_count + 1,
            },
        ),
    }
