"""Runtime configuration, sourced from environment variables / ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root (…/satquery). ``config.py`` lives in ``<root>/satquery/``.
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All tunables for the backend.

    Every field can be overridden with an environment variable of the same
    name (case-insensitive) or an entry in a ``.env`` file next to the project.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    reload: bool = False
    log_level: str = "INFO"
    cors_origins: str = "*"          # comma separated, or "*"
    api_keys: str = ""               # comma separated; empty => auth disabled
    max_upload_mb: int = 25

    # ----------------------------------------------------------------- storage
    work_dir: Path = ROOT_DIR / "var" / "uploads"

    # --------------------------------------------------------- LLM + fallbacks
    # The text model is a chain: primary Groq model -> extra Groq models ->
    # OpenRouter (if a key is set). Each link is tried in order on failure.
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_models: str = "llama-3.3-70b-versatile,llama-3.1-8b-instant"

    openrouter_api_key: str | None = None
    openrouter_model: str = "openai/gpt-oss-120b"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    llm_temperature: float = 0.0
    llm_max_retries: int = 2
    llm_timeout: float = 60.0

    # ------------------------------------------------------------------- VLM
    # local     -> load the Qwen2-VL checkpoint locally with transformers/torch
    # disabled  -> image specialists return a clear "vision model disabled" note
    vlm_backend: str = "local"
    vlm_model_id: str = "manny2706/satquery-qwen2vl-16bit"
    vlm_device_map: str = "auto"
    vlm_torch_dtype: str = "bfloat16"
    vlm_max_new_tokens: int = 256
    vlm_load_on_startup: bool = False

    # ------------------------------------------------------------------ graph
    default_max_retries: int = 2
    min_confidence: float = 0.75
    recursion_limit: int = 60

    # -------------------------------------------------------------- accessors
    @property
    def cors_origin_list(self) -> list[str]:
        value = self.cors_origins.strip()
        if value in ("", "*"):
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def api_key_list(self) -> list[str]:
        return [item.strip() for item in self.api_keys.split(",") if item.strip()]

    @property
    def groq_fallback_model_list(self) -> list[str]:
        return [item.strip() for item in self.groq_fallback_models.split(",") if item.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Rebuild the cached settings object (used by tests)."""
    get_settings.cache_clear()
    return get_settings()
