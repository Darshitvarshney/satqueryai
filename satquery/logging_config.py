"""Minimal logging setup shared by the API and the CLI entrypoint."""

from __future__ import annotations

import logging

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    resolved = level.upper()
    if _CONFIGURED:
        logging.getLogger().setLevel(resolved)
        return
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _CONFIGURED = True
