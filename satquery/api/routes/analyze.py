"""Synchronous analysis endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from satquery.api.deps import require_api_key
from satquery.api.inputs import collect_json_inputs, collect_upload_inputs
from satquery.api.schemas import AnalyzeJsonRequest, AnalyzeResult
from satquery.graph import run_analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["analyze"], dependencies=[Depends(require_api_key)])


def _run(query: str, max_retries: int | None, kwargs: dict) -> dict:
    return run_analysis(query=query, max_retries=max_retries, **kwargs)


@router.post("/analyze", response_model=AnalyzeResult)
async def analyze_multipart(
    query: str = Form(..., description="Natural-language question about the imagery"),
    optical: UploadFile | None = File(default=None),
    sar: UploadFile | None = File(default=None),
    image_t1: UploadFile | None = File(default=None, description="Earlier image for change detection"),
    image_t2: UploadFile | None = File(default=None, description="Later image for change detection"),
    images: list[UploadFile] | None = File(default=None, description="Generic image(s)"),
    max_retries: int | None = Form(default=None, ge=0, le=5),
) -> AnalyzeResult:
    kwargs = await collect_upload_inputs(
        optical=optical, sar=sar, image_t1=image_t1, image_t2=image_t2, images=images
    )
    if not kwargs:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one image file (optical, sar, image_t1, image_t2 or images).",
        )
    state = await run_in_threadpool(_run, query, max_retries, kwargs)
    return AnalyzeResult.from_state(state)


@router.post("/analyze/json", response_model=AnalyzeResult)
async def analyze_json(request: AnalyzeJsonRequest) -> AnalyzeResult:
    kwargs = await run_in_threadpool(collect_json_inputs, request)
    if not kwargs:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one image via *_url or *_b64 fields.",
        )
    state = await run_in_threadpool(_run, request.query, request.max_retries, kwargs)
    return AnalyzeResult.from_state(state)
