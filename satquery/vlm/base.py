"""Vision-language-model backend contract.

The graph never imports torch/transformers directly — it only talks to a
``VLMBackend``. That keeps the orchestration importable on machines without a
GPU and makes the model easy to stub in tests or swap for a remote service.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# A path (str/Path), a PIL image, raw bytes, or a file-like object.
ImageInput = Any


class VLMUnavailableError(RuntimeError):
    """Raised when an image request is made but no usable VLM is configured."""


@runtime_checkable
class VLMBackend(Protocol):
    name: str

    def caption(
        self,
        image: ImageInput,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        """Answer ``prompt`` about a single image."""

    def compare(
        self,
        image_a: ImageInput,
        image_b: ImageInput,
        prompt: str,
        *,
        label_a: str = "IMAGE 1",
        label_b: str = "IMAGE 2",
        max_new_tokens: int | None = None,
    ) -> str:
        """Answer ``prompt`` about a pair of images (change / cross-modal)."""

    def health(self) -> dict[str, Any]:
        """Lightweight status for the ``/api/v1/status`` endpoint."""
