"""High-level entrypoint that builds the initial state and runs the graph."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

from satquery.config import get_settings
from satquery.graph.builder import get_graph

logger = logging.getLogger(__name__)


def build_initial_state(
    *,
    query: str,
    optical_image: str | None = None,
    sar_image: str | None = None,
    image_t1: str | None = None,
    image_t2: str | None = None,
    images: list[str] | None = None,
    roi: Any = None,
    max_retries: int | None = None,
) -> dict:
    settings = get_settings()

    all_images: list[str] = list(images) if images else []
    for path in (optical_image, sar_image, image_t1, image_t2):
        if path and path not in all_images:
            all_images.append(path)

    return {
        "query": query,
        "images": all_images,
        "optical_image": optical_image,
        "sar_image": sar_image,
        "image_t1": image_t1,
        "image_t2": image_t2,
        "roi": roi,
        "max_retries": settings.default_max_retries if max_retries is None else int(max_retries),
        "retry_count": 0,
        "retry_task": None,
        "agent_results": [],
        "evidence": [],
        "execution_trace": [],
        "confidence": 0.0,
        "final_answer": "",
    }


def run_analysis(**kwargs: Any) -> dict:
    """Run one full analysis. ``kwargs`` are forwarded to :func:`build_initial_state`."""
    state = build_initial_state(**kwargs)
    settings = get_settings()
    started = time.perf_counter()
    result = get_graph().invoke(state, config={"recursion_limit": settings.recursion_limit})
    result["duration_seconds"] = round(time.perf_counter() - started, 3)
    logger.info(
        "analysis done task=%s retries=%s duration=%.2fs",
        result.get("current_task"),
        result.get("retry_count"),
        result["duration_seconds"],
    )
    return result


def stream_analysis(**kwargs: Any) -> Iterator[dict]:
    """Run one analysis, yielding events as the graph progresses.

    Event shapes::

        {"type": "start",    "state": <initial state summary>}
        {"type": "progress", "entry": <execution_trace entry>}
        {"type": "result",   "state": <final state (dict)>}
        {"type": "error",    "detail": "<message>"}

    Synchronous generator — bridge it to async with
    ``satquery.api.streaming.sse_from_sync``.
    """
    state = build_initial_state(**kwargs)
    settings = get_settings()
    started = time.perf_counter()

    yield {
        "type": "start",
        "state": {
            "query": state["query"],
            "image_count": len(state["images"]),
            "max_retries": state["max_retries"],
        },
    }

    final_state: dict = dict(state)
    seen = 0
    try:
        for snapshot in get_graph().stream(
            state,
            config={"recursion_limit": settings.recursion_limit},
            stream_mode="values",
        ):
            final_state = snapshot
            entries = snapshot.get("execution_trace", []) or []
            for entry in entries[seen:]:
                yield {"type": "progress", "entry": entry}
            seen = len(entries)
    except Exception as exc:  # noqa: BLE001
        logger.exception("stream_analysis failed")
        yield {"type": "error", "detail": f"{type(exc).__name__}: {exc}"}
        return

    final_state["duration_seconds"] = round(time.perf_counter() - started, 3)
    yield {"type": "result", "state": final_state}
