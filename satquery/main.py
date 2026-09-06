"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from satquery import __version__
from satquery.api.errors import install_exception_handlers
from satquery.api.routes import analyze, health, jobs
from satquery.config import get_settings
from satquery.jobs import get_job_manager
from satquery.logging_config import configure_logging

logger = logging.getLogger(__name__)

DESCRIPTION = """
SatQuery AI — a multi-agent LangGraph pipeline over a Qwen2-VL vision model and
a Groq LLM for satellite-image question answering, scene description, bi-temporal
change detection, optical+SAR cross-modal reasoning and geo-spatial inspection.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("SatQuery backend %s starting (vlm_backend=%s)", __version__, settings.vlm_backend)

    if settings.vlm_load_on_startup:
        try:
            from satquery.vlm import get_vlm

            get_vlm().load()
        except Exception:  # noqa: BLE001
            logger.exception("VLM warm-up failed; continuing without a preloaded model")

    yield

    get_job_manager().shutdown()
    logger.info("SatQuery backend stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="SatQuery AI Backend",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(analyze.router)
    app.include_router(jobs.router)
    return app


app = create_app()
