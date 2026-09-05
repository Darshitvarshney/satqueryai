"""Final answer generation."""

from __future__ import annotations

import logging

from satquery.graph.llm import invoke_text
from satquery.graph.nodes._common import as_text, trace
from satquery.graph.prompts import ANSWER_SYNTHESIS_PROMPT
from satquery.graph.state import SatQueryState

logger = logging.getLogger(__name__)


def answer_synthesis_node(state: SatQueryState) -> dict:
    # Reached directly from input_validation when the request is unusable.
    if not state.get("input_valid", True):
        errors = state.get("validation_errors") or ["Unknown validation error."]
        message = "The request could not be analyzed. " + " ".join(errors)
        return {
            "final_answer": message,
            "execution_trace": trace(
                state, {"node": "answer_synthesis", "status": "completed", "mode": "validation_error"}
            ),
        }

    evidence = state.get("evidence", [])
    parts: list[str] = []
    for item in evidence:
        if item.get("status") == "failed":
            continue
        text = as_text(item.get("verified_finding") or item.get("finding")).strip()
        if text:
            parts.append(f"[{item.get('agent', 'agent')} · {item.get('task', '')}]\n{text}")
    evidence_text = "\n\n".join(parts)

    if not evidence_text:
        failures = [as_text(e.get("finding")).strip() for e in evidence if e.get("status") == "failed"]
        unique = list(dict.fromkeys(f for f in failures if f))
        reason = " ".join(unique) or "No usable visual evidence was produced."
        return {
            "final_answer": f"SatQuery could not produce a grounded answer. {reason}",
            "execution_trace": trace(
                state, {"node": "answer_synthesis", "status": "completed", "mode": "no_evidence"}
            ),
        }

    try:
        confidence = float(state.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0

    prompt = ANSWER_SYNTHESIS_PROMPT.format(
        query=state.get("query", ""),
        evidence=evidence_text,
        confidence=confidence,
    )
    try:
        answer = invoke_text(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Answer synthesis LLM unavailable: %s", exc)
        answer = (
            "SatQuery gathered the following verified evidence but could not run the "
            f"final summariser ({exc}):\n\n{evidence_text}"
        )

    return {
        "final_answer": answer,
        "execution_trace": trace(
            state, {"node": "answer_synthesis", "status": "completed", "mode": "synthesised"}
        ),
    }
