"""Apple MLX backend for IndicBlockOCR text transcription."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

from PIL.Image import Image

logger = logging.getLogger("idp_recognizer_mlx")


class CropRequest(NamedTuple):
    image: Image
    prompt: str


@runtime_checkable
class RecognizerBackend(Protocol):
    """Protocol matching idp_recognizer.RecognizerBackend."""

    def transcribe(self, requests: list[CropRequest]) -> list[str]: ...

    def close(self) -> None: ...


class MlxRecognizer:
    """High-performance text recognition backend on Apple Silicon Metal GPU."""

    def __init__(
        self,
        model_path: str | Path,
        max_tokens: int = 512,
        verbose: bool = False,
    ) -> None:
        from mlx_vlm import load

        self.model_path = Path(model_path)
        self.max_tokens = max_tokens
        self.verbose = verbose

        logger.info("Loading MLX recognizer model from %s...", self.model_path)
        self.model, self.processor = load(str(self.model_path))
        logger.info("MLX recognizer ready on Metal GPU.")

    def _format_prompt(self, text_prompt: str) -> str:
        """Format multimodal chat template for Qwen3.5."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def transcribe(self, requests: list[CropRequest]) -> list[str]:
        """Transcribe each crop sequentially on Apple Silicon unified memory."""
        from mlx_vlm import generate

        results: list[str] = []
        total = len(requests)
        for idx, req in enumerate(requests):
            prompt = self._format_prompt(req.prompt)
            res = generate(
                self.model,
                self.processor,
                prompt=prompt,
                image=[req.image],
                max_tokens=self.max_tokens,
                repetition_penalty=1.15,
                repetition_context_size=64,
                verbose=self.verbose,
            )
            text = res.text.strip() if hasattr(res, "text") else str(res).strip()
            results.append(text)
            logger.debug("Transcribed crop %d/%d (%d chars)", idx + 1, total, len(text))
        return results

    def close(self) -> None:
        self.model = None
        self.processor = None
