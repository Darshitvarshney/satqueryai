"""Health / status / graph-introspection endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from satquery import __version__
from satquery.config import get_settings
from satquery.graph.builder import get_graph
from satquery.graph.llm import describe_model_chain
from satquery.vlm import get_vlm

logger = logging.getLogger(__name__)
router = APIRouter(tags=["meta"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/api/v1/status")
async def status() -> dict:
    settings = get_settings()
    try:
        vlm_health = get_vlm().health()
    except Exception as exc:  # noqa: BLE001
        vlm_health = {"error": str(exc)}

    try:
        nodes = sorted(get_graph().get_graph().nodes)
    except Exception:  # noqa: BLE001
        nodes = []

    return {
        "version": __version__,
        "llm": {
            "configured": bool(settings.groq_api_key or settings.openrouter_api_key),
            "model": settings.groq_model,
            "fallback_chain": describe_model_chain(),
        },
        "vlm": vlm_health,
        "graph_nodes": nodes,
        "auth_required": bool(settings.api_key_list),
        "defaults": {
            "max_retries": settings.default_max_retries,
            "min_confidence": settings.min_confidence,
            "max_upload_mb": settings.max_upload_mb,
        },
    }


@router.get("/api/v1/graph", response_class=PlainTextResponse)
async def graph_mermaid() -> str:
    try:
        return get_graph().get_graph().draw_mermaid()
    except Exception as exc:  # noqa: BLE001
        return f"# graph diagram unavailable: {exc}"
