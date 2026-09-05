"""In-process async job queue for long-running analyses.

Single-process only: jobs live in memory and a bounded thread pool executes
them. Vision inference is serialised (``max_workers=1`` by default) because a
single GPU cannot run concurrent generations safely. For multi-worker
deployments put a real broker (Celery/RQ/Arq) behind the same interface.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

JobStatus = str  # "queued" | "running" | "succeeded" | "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None


class JobManager:
    def __init__(self, *, max_workers: int = 1, max_jobs: int = 250) -> None:
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._max_jobs = max_jobs
        self._lock = threading.Lock()

    def _ensure_executor_locked(self) -> ThreadPoolExecutor:
        if self._executor is None or getattr(self._executor, "_shutdown", False):
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="satquery-job"
            )
        return self._executor

    def submit(self, fn: Callable[[], dict]) -> Job:
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._evict_locked()
            executor = self._ensure_executor_locked()
        executor.submit(self._run, job.id, fn)
        return job

    def _run(self, job_id: str, fn: Callable[[], dict]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = time.time()
        try:
            result = fn()
            with self._lock:
                job.result = result
                job.status = "succeeded"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            with self._lock:
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "failed"
        finally:
            with self._lock:
                job.finished_at = time.time()

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            ids = self._order[-limit:][::-1]
            return [self._jobs[i] for i in ids]

    def _evict_locked(self) -> None:
        while len(self._order) > self._max_jobs:
            oldest = self._order.pop(0)
            self._jobs.pop(oldest, None)

    def shutdown(self) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager


def reset_job_manager() -> None:
    global _manager
    if _manager is not None:
        _manager.shutdown()
    _manager = None
