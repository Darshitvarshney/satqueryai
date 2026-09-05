"""Process-wide VLM backend selection."""

from __future__ import annotations

import logging
from typing import Any

from satquery.config import get_settings
from satquery.vlm.base import VLMUnavailableError

logger = logging.getLogger(__name__)

_backend: Any = None


class DisabledVLM:
    """Placeholder backend used when ``VLM_BACKEND=disabled``.

    Image specialists degrade gracefully: instead of crashing the request they
    record a failed-evidence note and the final answer explains that the vision
    model is unavailable.
    """

    name = "disabled"

    def _unavailable(self, *_args, **_kwargs):
        raise VLMUnavailableError(
            "The vision model is disabled (VLM_BACKEND=disabled). Set VLM_BACKEND=local "
            "and install the ML dependencies (pip install -r requirements-ml.txt) to "
            "enable image analysis."
        )

    caption = _unavailable
    compare = _unavailable

    def health(self) -> dict:
        return {"backend": self.name, "loaded": False}


def _build():
    settings = get_settings()
    backend = settings.vlm_backend.strip().lower()
    if backend in ("disabled", "none", "off", ""):
        logger.info("VLM backend: disabled")
        return DisabledVLM()
    if backend == "local":
        from satquery.vlm.local import LocalQwen2VLBackend

        instance = LocalQwen2VLBackend()
        if settings.vlm_load_on_startup:
            instance.load()
        return instance
    raise ValueError(f"Unknown VLM backend: {settings.vlm_backend!r}")


def get_vlm():
    global _backend
    if _backend is None:
        _backend = _build()
    return _backend


def set_vlm(backend) -> None:
    """Install a specific backend (used by tests / custom deployments)."""
    global _backend
    _backend = backend


def reset_vlm() -> None:
    global _backend
    _backend = None
