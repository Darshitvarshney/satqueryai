"""``python -m satquery`` — run the API with uvicorn."""

from __future__ import annotations

from satquery.config import get_settings


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "satquery.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
