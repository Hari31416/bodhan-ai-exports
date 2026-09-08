#!/usr/bin/env python3
"""Export Stage 2 (IndicBlockOCR / Qwen3.5-0.8B) to Apple MLX format.

Converts Hugging Face weights to MLX SafeTensors format with 4-bit/8-bit quantization
and packages tokenizer, processor, and chat template assets.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from mlx_vlm import convert, load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_recognizer_mlx")


def export_recognizer_mlx(
    bundle_dir: Path,
    output_dir: Path,
    quantize: bool = True,
    q_bits: int = 4,
) -> Path:
    """Convert Qwen3.5-0.8B recognizer weights to MLX."""
    ocr_weights = bundle_dir / "weights" / "ocr"
    if not ocr_weights.exists():
        raise FileNotFoundError(
            f"OCR weights directory not found at {ocr_weights}. Run download_weights.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Converting OCR weights to MLX (quantize=%s, q_bits=%d)...", quantize, q_bits
    )

    convert(
        hf_path=str(ocr_weights),
        mlx_path=str(output_dir),
        quantize=quantize,
        q_bits=q_bits,
    )
    logger.info("Conversion complete. Verifying exported MLX model...")

    model, processor = load(str(output_dir))
    logger.info(
        "Successfully loaded MLX recognizer model and processor from %s", output_dir
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export IndicBlockOCR (Qwen3.5-0.8B) to MLX format."
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
        default=None,
        help="Output directory for MLX model (default: auto-selected by precision).",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization and export full-precision BF16 weights.",
    )
    parser.add_argument(
        "--q-bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits (4 or 8). Default: 4.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all three precision variants (4-bit, 8-bit, and BF16).",
    )
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.resolve()
    base_mlx = Path(__file__).parent / "mlx_output"

    if args.all:
        logger.info("Exporting all three precision variants: 4-bit, 8-bit, and BF16...")
        export_recognizer_mlx(
            bundle_dir, base_mlx / "ocr_4bit", quantize=True, q_bits=4
        )
        export_recognizer_mlx(
            bundle_dir, base_mlx / "ocr_8bit", quantize=True, q_bits=8
        )
        export_recognizer_mlx(bundle_dir, base_mlx / "ocr_bf16", quantize=False)
        return

    quantize = not args.no_quantize
    if args.output_dir is None:
        if quantize:
            out_dir = base_mlx / f"ocr_{args.q_bits}bit"
        else:
            out_dir = base_mlx / "ocr_bf16"
    else:
        out_dir = args.output_dir

    export_recognizer_mlx(
        bundle_dir=bundle_dir,
        output_dir=out_dir.resolve(),
        quantize=quantize,
        q_bits=args.q_bits,
    )


if __name__ == "__main__":
    main()
