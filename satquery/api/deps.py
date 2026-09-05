"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from satquery.config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce ``X-API-Key`` when ``API_KEYS`` is configured; otherwise no-op."""
    keys = get_settings().api_key_list
    if not keys:
        return
    if x_api_key not in keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
