"""Groq chat model access for the orchestration / verification / synthesis nodes.

Everything goes through :func:`invoke_text`, which returns a plain string and
honours a test override so the graph can run without network access.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable

from satquery.config import get_settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when a text-model call is attempted without a configured key."""


@lru_cache(maxsize=1)
def get_llm():
    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMUnavailableError(
            "GROQ_API_KEY is not configured. Set it in the environment or .env file."
        )
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
        max_retries=settings.llm_max_retries,
        timeout=settings.llm_timeout,
    )


def reset_llm() -> None:
    get_llm.cache_clear()


_text_override: Callable[[str, str | None], str] | None = None


def set_llm_text_override(fn: Callable[[str, str | None], str] | None) -> None:
    """Install a substitute for :func:`invoke_text` (tests / offline mode)."""
    global _text_override
    _text_override = fn


def invoke_text(prompt: str, *, system: str | None = None) -> str:
    """Send ``prompt`` to the chat model and return its text content."""
    if _text_override is not None:
        return _text_override(prompt, system)

    messages: list[tuple[str, str]] = []
    if system:
        messages.append(("system", system))
    messages.append(("human", prompt))

    response = get_llm().invoke(messages)
    content = getattr(response, "content", response)
    if isinstance(content, list):  # some providers return content parts
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content).strip()
