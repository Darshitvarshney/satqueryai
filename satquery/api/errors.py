"""Translate internal exceptions into HTTP responses."""

from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from satquery.graph.llm import LLMUnavailableError
from satquery.vlm.base import VLMUnavailableError

logger = logging.getLogger(__name__)


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LLMUnavailableError)
    async def _llm_unavailable(_request: Request, exc: LLMUnavailableError):
        return JSONResponse(
            status_code=503, content={"error": "llm_unavailable", "detail": str(exc)}
        )

    @app.exception_handler(VLMUnavailableError)
    async def _vlm_unavailable(_request: Request, exc: VLMUnavailableError):
        return JSONResponse(
            status_code=503, content={"error": "vlm_unavailable", "detail": str(exc)}
        )

    @app.exception_handler(ValueError)
    async def _bad_request(_request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400, content={"error": "bad_request", "detail": str(exc)}
        )

    @app.exception_handler(httpx.HTTPError)
    async def _upstream_error(_request: Request, exc: httpx.HTTPError):
        logger.warning("Upstream HTTP error: %s", exc)
        return JSONResponse(
            status_code=502,
            content={"error": "upstream_error", "detail": str(exc)},
        )
