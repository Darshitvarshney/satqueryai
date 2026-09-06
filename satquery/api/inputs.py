"""Turn HTTP request inputs (uploads / URLs / base64) into ``run_analysis`` kwargs."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from satquery.config import get_settings
from satquery.utils.images import download_image, save_base64, save_upload

_SLOT_KEYS = ("optical_image", "sar_image", "image_t1", "image_t2")


def new_request_dir() -> Path:
    directory = get_settings().work_dir / uuid.uuid4().hex[:16]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


async def collect_upload_inputs(
    *,
    optical: Any = None,
    sar: Any = None,
    image_t1: Any = None,
    image_t2: Any = None,
    images: list[Any] | None = None,
) -> dict:
    dest = new_request_dir()
    kwargs: dict = {}
    for key, upload in (
        ("optical_image", optical),
        ("sar_image", sar),
        ("image_t1", image_t1),
        ("image_t2", image_t2),
    ):
        if upload is not None and getattr(upload, "filename", None):
            kwargs[key] = await save_upload(upload, dest)

    extra: list[str] = []
    for upload in images or []:
        if upload is not None and getattr(upload, "filename", None):
            extra.append(await save_upload(upload, dest))
    if extra:
        kwargs["images"] = extra
    return kwargs


def collect_json_inputs(request) -> dict:
    """Blocking (network + disk) — call via ``run_in_threadpool``."""
    dest = new_request_dir()
    kwargs: dict = {}
    for key in _SLOT_KEYS:
        url = getattr(request, f"{key}_url", None)
        b64 = getattr(request, f"{key}_b64", None)
        if url:
            kwargs[key] = str(download_image(url, dest))
        elif b64:
            kwargs[key] = str(save_base64(b64, dest, key))
    return kwargs
