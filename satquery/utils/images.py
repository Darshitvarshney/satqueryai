"""Helpers for turning arbitrary image inputs (uploads, URLs, base64, paths)
into files on disk and into PIL images for the vision model."""

from __future__ import annotations

import base64
import binascii
import io
import re
import uuid
from pathlib import Path
from typing import Any

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _unique_name(filename: str | None) -> str:
    stem = _SAFE_NAME.sub("_", filename or "image").strip("_") or "image"
    stem = stem[-80:]
    return f"{uuid.uuid4().hex[:12]}_{stem}"


def save_bytes(data: bytes, dest_dir: str | Path, filename: str = "image") -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / _unique_name(filename)
    path.write_bytes(data)
    return path


def save_base64(data: str, dest_dir: str | Path, filename: str = "image") -> Path:
    payload = data.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:  # pragma: no cover - trivial
        raise ValueError(f"Invalid base64 image data: {exc}") from exc
    if not raw:
        raise ValueError("Decoded base64 image is empty")
    return save_bytes(raw, dest_dir, filename)


def download_image(
    url: str,
    dest_dir: str | Path,
    *,
    timeout: float = 30.0,
    max_bytes: int = 30 * 1024 * 1024,
) -> Path:
    import httpx

    chunks: list[bytes] = []
    total = 0
    with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("Remote image exceeds the maximum allowed size")
            chunks.append(chunk)
    if total == 0:
        raise ValueError("Remote image download was empty")
    name = url.split("/")[-1].split("?")[0] or "download"
    return save_bytes(b"".join(chunks), dest_dir, name)


async def save_upload(upload: Any, dest_dir: str | Path, *, max_bytes: int | None = None) -> str:
    """Stream a Starlette/FastAPI ``UploadFile`` to disk with a size cap."""
    from satquery.config import get_settings

    limit = max_bytes or get_settings().max_upload_bytes
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / _unique_name(getattr(upload, "filename", None))

    size = 0
    try:
        with target.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError(
                        f"Upload '{getattr(upload, 'filename', 'file')}' exceeds the "
                        f"{limit // (1024 * 1024)} MB limit"
                    )
                handle.write(chunk)
    except ValueError:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    if size == 0:
        target.unlink(missing_ok=True)
        raise ValueError("Uploaded file is empty")
    return str(target)


def to_pil(image: Any) -> "Any":
    """Coerce a path / PIL image / bytes / file-like object into an RGB PIL image."""
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    if hasattr(image, "read"):
        return Image.open(image).convert("RGB")
    return Image.open(str(image)).convert("RGB")
