"""Asynchronous (background) analysis jobs."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from satquery.api.deps import require_api_key
from satquery.api.inputs import collect_upload_inputs
from satquery.api.schemas import AnalyzeResult, JobDetail, JobSummary
from satquery.graph import run_analysis
from satquery.jobs import Job, get_job_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"], dependencies=[Depends(require_api_key)])


def _summary(job: Job) -> JobSummary:
    return JobSummary(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
    )


def _detail(job: Job) -> JobDetail:
    result = AnalyzeResult.from_state(job.result) if job.result else None
    return JobDetail(**_summary(job).model_dump(), result=result)


@router.post("", response_model=JobDetail, status_code=202)
async def create_job(
    query: str = Form(...),
    optical: UploadFile | None = File(default=None),
    sar: UploadFile | None = File(default=None),
    image_t1: UploadFile | None = File(default=None),
    image_t2: UploadFile | None = File(default=None),
    images: list[UploadFile] | None = File(default=None),
    max_retries: int | None = Form(default=None, ge=0, le=5),
) -> JobDetail:
    kwargs = await collect_upload_inputs(
        optical=optical, sar=sar, image_t1=image_t1, image_t2=image_t2, images=images
    )
    if not kwargs:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one image file (optical, sar, image_t1, image_t2 or images).",
        )

    def _task() -> dict:
        return run_analysis(query=query, max_retries=max_retries, **kwargs)

    job = get_job_manager().submit(_task)
    return _detail(job)


@router.get("", response_model=list[JobSummary])
async def list_jobs(limit: int = Query(default=50, ge=1, le=250)) -> list[JobSummary]:
    return [_summary(job) for job in get_job_manager().list(limit)]


@router.get("/{job_id}", response_model=JobDetail)
async def get_job(job_id: str) -> JobDetail:
    job = get_job_manager().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _detail(job)
