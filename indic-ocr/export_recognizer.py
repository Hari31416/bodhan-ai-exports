#!/usr/bin/env python3
"""Export Stage 2 (IndicBlockOCR / Qwen3.5-0.8B) components to ONNX.

Produces:
    - visual_encoder.onnx (Vision patch feature extractor)
    - Tokenizer and generation configs for browser / onnxruntime inference
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys

import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_recognizer")


class QwenVisualWrapper(torch.nn.Module):
    """Wrapper around Qwen3.5 visual backbone for ONNX export."""

    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual.eval()

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        return self.visual(pixel_values, grid_thw=grid_thw)


def export_recognizer_visual(
    bundle_dir: Path,
    output_dir: Path,
    opset: int = 18,
) -> Path:
    """Export the vision encoder component of Qwen3.5-0.8B to ONNX."""
    weights_path = bundle_dir / "weights" / "ocr"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"OCR weights directory not found at {weights_path}. Run download_weights.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_onnx_path = output_dir / "visual_encoder.onnx"

    logger.info("Loading Qwen3.5 model from %s", weights_path)
    model = AutoModelForImageTextToText.from_pretrained(
        str(weights_path),
        dtype=torch.float32,
        device_map="cpu",
    ).eval()

    logger.info("Copying tokenizer and processor artifacts to %s", output_dir)
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
    ]:
        src_file = weights_path / filename
        if src_file.exists():
            shutil.copy2(src_file, output_dir / filename)

    wrapper = QwenVisualWrapper(model.model.visual).eval()

    # Dummy input: 256 patches x 1536 patch dimension
    dummy_pixels = torch.randn(256, 1536, dtype=torch.float32)
    dummy_grid = torch.tensor([[1, 16, 16]], dtype=torch.int64)

    logger.info("Exporting visual encoder to %s...", visual_onnx_path)
    torch.onnx.export(
        wrapper,
        (dummy_pixels, dummy_grid),
        str(visual_onnx_path),
        input_names=["pixel_values", "grid_thw"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "num_patches"},
            "grid_thw": {0: "batch_size"},
            "image_features": {0: "num_patches"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    logger.info("Checking visual encoder ONNX model...")
    onnx_model = onnx.load(str(visual_onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("Visual encoder exported and verified successfully.")
    return visual_onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export IndicBlockOCR components to ONNX."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).parent / "model_bundle",
        help="Path to model bundle containing downloaded weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "onnx_output" / "ocr",
        help="Destination directory for exported ONNX and tokenizer files.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18).",
    )
    args = parser.parse_args()

    export_recognizer_visual(
        bundle_dir=args.bundle_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
