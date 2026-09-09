#!/usr/bin/env python3
"""Pure ONNX Runtime implementation of Stage 2 (OCR Text Recognizer).

Implements the RecognizerBackend protocol for Qwen3.5-0.8B Vision-Language model
using exported visual_encoder.onnx and text_decoder.onnx (or their INT8 variants).
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("idp_recognizer_onnx")


class OnnxRecognizer:
    """Pure ONNX Runtime text recognizer backend implementing RecognizerBackend protocol."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        bundle_dir: str | Path | None = None,
        visual_model_path: str | Path | None = None,
        decoder_model_path: str | Path | None = None,
        quantized: bool = False,
        max_tokens: int = 512,
        providers: list[str] | None = None,
    ) -> None:
        root = Path(__file__).parent
        if bundle_dir is None:
            bundle_dir = root / "model_bundle"
        self.bundle_dir = Path(bundle_dir)

        if model_dir is None:
            model_dir = root / "onnx_output" / "ocr"
        self.model_dir = Path(model_dir)

        self.max_tokens = max_tokens
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.providers = providers

        # Determine visual and decoder model paths
        if visual_model_path is None:
            v_name = "visual_encoder_int8.onnx" if quantized else "visual_encoder.onnx"
            visual_model_path = self.model_dir / v_name
            if not visual_model_path.exists():
                visual_model_path = self.model_dir / "visual_encoder.onnx"

        if decoder_model_path is None:
            d_name = "text_decoder_int8.onnx" if quantized else "text_decoder.onnx"
            decoder_model_path = self.model_dir / d_name
            if not decoder_model_path.exists():
                decoder_model_path = self.model_dir / "text_decoder.onnx"

        self.visual_path = Path(visual_model_path)
        self.decoder_path = Path(decoder_model_path)

        logger.info("Initializing ONNX visual encoder from %s...", self.visual_path.name)
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.visual_session = ort.InferenceSession(
            str(self.visual_path), sess_options=sess_opts, providers=self.providers
        )

        logger.info("Initializing ONNX text decoder from %s...", self.decoder_path.name)
        self.decoder_session = ort.InferenceSession(
            str(self.decoder_path), sess_options=sess_opts, providers=self.providers
        )

        # Load AutoProcessor for tokenizer, chat templates, and preprocessing
        from transformers import AutoProcessor

        weights_dir = self.bundle_dir / "weights" / "ocr"
        tokenizer_src = weights_dir if weights_dir.exists() else self.model_dir
        self.processor = AutoProcessor.from_pretrained(str(tokenizer_src))
        self.tokenizer = self.processor.tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id

    def _prompt(self, text: str) -> str:
        """Format input text into Qwen3.5 chat template."""
        return self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def _generate_tokens(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int,
    ) -> list[int]:
        """Autoregressive greedy token generation loop."""
        curr_ids = input_ids.copy()
        generated_tokens: list[int] = []

        for step in range(max_new_tokens):
            seq_len = curr_ids.shape[1]
            pos_ids = np.arange(seq_len, dtype=np.int64).reshape((1, 1, seq_len))

            outputs = self.decoder_session.run(
                None,
                {
                    "input_ids": curr_ids,
                    "position_ids": pos_ids,
                },
            )
            logits = outputs[0]  # shape: [batch, seq, vocab]
            next_token = int(np.argmax(logits[0, -1, :]))

            if next_token == self.eos_token_id:
                break

            generated_tokens.append(next_token)
            curr_ids = np.append(curr_ids, [[next_token]], axis=1)

        return generated_tokens

    def transcribe(self, requests: list[Any]) -> list[str]:
        """Transcribe list of CropRequest objects."""
        texts: list[str] = []

        for req in requests:
            prompt_str = self._prompt(req.prompt)
            proc_inputs = self.processor(
                text=[prompt_str],
                images=[req.image],
                return_tensors="np",
            )

            # Vision feature extraction
            pixel_values = proc_inputs["pixel_values"]
            grid_thw = proc_inputs.get("image_grid_thw")
            if grid_thw is not None:
                _ = self.visual_session.run(
                    None,
                    {
                        "pixel_values": pixel_values,
                        "grid_thw": grid_thw,
                    },
                )

            # Decoder autoregressive generation
            input_ids = proc_inputs["input_ids"]
            gen_tokens = self._generate_tokens(input_ids, self.max_tokens)

            decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            texts.append(decoded)

        return texts

    def close(self) -> None:
        """Release ONNX Runtime sessions."""
        self.visual_session = None
        self.decoder_session = None
        self.processor = None
        self.tokenizer = None
