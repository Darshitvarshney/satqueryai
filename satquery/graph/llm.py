"""Groq chat model access with an automatic fallback chain.

The text model used by the orchestration / verification / synthesis nodes is a
chain:

    primary Groq model
      -> each GROQ_FALLBACK_MODELS entry (same Groq key)
      -> OpenRouter model (if OPENROUTER_API_KEY is set)

Any exception from one link (rate limit, outage, decommissioned model, ...)
transparently falls through to the next via LangChain's ``with_fallbacks``.
Everything is consumed through :func:`invoke_text`, which returns a plain string
and honours a test override so the graph can run without network access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from satquery.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when no text model in the chain can be constructed."""


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # "groq" | "openrouter"
    model: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.provider}:{self.model}"


def build_model_chain(settings: Settings | None = None) -> list[ModelSpec]:
    """Ordered list of models that :func:`get_llm` will try."""
    settings = settings or get_settings()
    chain: list[ModelSpec] = []
    if settings.groq_api_key:
        chain.append(ModelSpec("groq", settings.groq_model))
        for model in settings.groq_fallback_model_list:
            if model and model != settings.groq_model:
                chain.append(ModelSpec("groq", model))
    if settings.openrouter_api_key:
        chain.append(ModelSpec("openrouter", settings.openrouter_model))
    return chain


def _build_model(spec: ModelSpec, settings: Settings):
    if spec.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=spec.model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout,
        )
    if spec.provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "OPENROUTER_API_KEY is set but langchain-openai is not installed "
                "(pip install langchain-openai)."
            ) from exc

        return ChatOpenAI(
            model=spec.model,
            temperature=settings.llm_temperature,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout,
        )
    raise ValueError(f"Unknown model provider: {spec.provider!r}")


@lru_cache(maxsize=1)
def get_llm():
    settings = get_settings()
    specs = build_model_chain(settings)
    if not specs:
        raise LLMUnavailableError(
            "No text model configured. Set GROQ_API_KEY (and optionally "
            "OPENROUTER_API_KEY) in the environment or .env file."
        )

    built = []
    for spec in specs:
        try:
            built.append(_build_model(spec, settings))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping text model %s: %s", spec, exc)
    if not built:
        raise LLMUnavailableError("None of the configured text models could be initialised.")

    primary, *fallbacks = built
    if fallbacks:
        logger.info(
            "Text model chain: %s", " -> ".join(str(s) for s in specs[: len(built)])
        )
        return primary.with_fallbacks(fallbacks)
    return primary


def reset_llm() -> None:
    get_llm.cache_clear()


def describe_model_chain() -> list[str]:
    """Human-readable model chain for the status endpoint."""
    try:
        return [str(spec) for spec in build_model_chain()]
    except Exception:  # noqa: BLE001 - never break /status
        return []


_text_override: Callable[[str, str | None], str] | None = None


def set_llm_text_override(fn: Callable[[str, str | None], str] | None) -> None:
    """Install a substitute for :func:`invoke_text` (tests / offline mode)."""
    global _text_override
    _text_override = fn


def invoke_text(prompt: str, *, system: str | None = None) -> str:
    """Send ``prompt`` to the text model chain and return its text content."""
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
