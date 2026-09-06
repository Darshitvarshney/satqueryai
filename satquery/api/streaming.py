"""Server-Sent Events plumbing.

Bridges a blocking generator (the LangGraph stream) onto an async SSE response,
running the producer in a worker thread and emitting ``: keepalive`` comments
during long gaps so proxies don't drop the connection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Callable, Iterator

logger = logging.getLogger(__name__)

_KEEPALIVE_SECONDS = 15.0
_SENTINEL = object()

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disable nginx proxy buffering
}


def format_sse(event: str, data) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, default=str)
    body = "".join(f"data: {line}\n" for line in (payload.splitlines() or [""]))
    return f"event: {event}\n{body}\n"


async def sse_from_sync(
    make_iterator: Callable[[], Iterator[dict]],
    *,
    keepalive_seconds: float = _KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    """Consume ``make_iterator()`` in a thread; yield formatted SSE strings.

    Each item yielded by the sync iterator must be ``{"event": str, "data": ...}``.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def produce() -> None:
        try:
            for item in make_iterator():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # noqa: BLE001
            logger.exception("SSE producer crashed")
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"event": "error", "data": {"detail": f"{type(exc).__name__}: {exc}"}},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    task = loop.run_in_executor(None, produce)
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if item is _SENTINEL:
                break
            yield format_sse(item.get("event", "message"), item.get("data"))
    finally:
        task.cancel()
