"""Local Qwen2-VL backend.

Mirrors the ``run_vlm`` / ``run_change_vlm`` / ``run_cross_modal_vlm`` helpers
from the research notebook, behind the :class:`~satquery.vlm.base.VLMBackend`
protocol. torch / transformers are imported lazily so importing this module is
cheap and safe.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from satquery.config import get_settings
from satquery.utils.images import to_pil

logger = logging.getLogger(__name__)


class LocalQwen2VLBackend:
    name = "local-qwen2vl"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        default_max_new_tokens: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model_id = model_id or settings.vlm_model_id
        self.device_map = device_map or settings.vlm_device_map
        self.torch_dtype_name = torch_dtype or settings.vlm_torch_dtype
        self.default_max_new_tokens = default_max_new_tokens or settings.vlm_max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            import torch
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            dtype = getattr(torch, self.torch_dtype_name, torch.bfloat16)
            logger.info(
                "Loading VLM %s (device_map=%s, dtype=%s)",
                self.model_id,
                self.device_map,
                self.torch_dtype_name,
            )

            # Processor first: cheap and the usual failure point (a missing
            # torchvision / tokenizer backend). Assign to self only once BOTH
            # halves succeed so a partial failure doesn't leave a broken state
            # that the `is_loaded` guard would happily skip.
            try:
                processor = AutoProcessor.from_pretrained(self.model_id)
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    device_map=self.device_map,
                )
            except ImportError as exc:
                raise RuntimeError(
                    f"Could not load VLM '{self.model_id}': {exc} "
                    "(hint: `pip install torchvision`; quantized checkpoints also "
                    "need `pip install bitsandbytes` and an NVIDIA GPU)."
                ) from exc

            model.eval()
            self._processor = processor
            self._model = model
            logger.info("VLM ready: %s", self.model_id)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    # ------------------------------------------------------------- inference
    def _generate(self, messages: list[dict], images: list[Any], max_new_tokens: int | None) -> str:
        self.load()
        import torch

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        )
        with self._infer_lock:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or self.default_max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.15,
                )
            trimmed = [
                out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)
            ]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0].strip()

    def caption(self, image, prompt, *, max_new_tokens=None) -> str:
        img = to_pil(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(messages, [img], max_new_tokens)

    def compare(
        self,
        image_a,
        image_b,
        prompt,
        *,
        label_a="IMAGE 1",
        label_b="IMAGE 2",
        max_new_tokens=None,
    ) -> str:
        a = to_pil(image_a)
        b = to_pil(image_b)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": label_a},
                    {"type": "image", "image": a},
                    {"type": "text", "text": label_b},
                    {"type": "image", "image": b},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self._generate(messages, [a, b], max_new_tokens)

    def health(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model_id": self.model_id,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype_name,
            "loaded": self.is_loaded,
        }
