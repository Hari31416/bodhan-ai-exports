#!/usr/bin/env python3
"""Export Stage 1 (IndicDocLayout / RT-DETR) weights to Apple MLX format.

Produces:
    - mlx_output/layout/model.safetensors
    - mlx_output/layout/config.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

import mlx.core as mx
from safetensors import safe_open

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_layout_mlx")


def export_layout_mlx(
    bundle_dir: Path,
    output_dir: Path,
) -> Path:
    """Convert layout detection weights to MLX safetensors format."""
    layout_weights_dir = bundle_dir / "weights" / "layout"
    src_weights = layout_weights_dir / "model.safetensors"
    src_config = layout_weights_dir / "config.json"

    if not src_weights.exists():
        raise FileNotFoundError(
            f"Layout weights not found at {src_weights}. Run download_weights.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_weights = output_dir / "model.safetensors"
    out_config = output_dir / "config.json"

    logger.info("Loading PyTorch layout weights from %s...", src_weights)
    mlx_tensors: dict[str, mx.array] = {}

    with safe_open(str(src_weights), framework="pt") as f:
        keys = list(f.keys())
        logger.info("Found %d weight tensors in source checkpoint.", len(keys))

        for k in keys:
            tensor = f.get_tensor(k)
            # Check if convolution weight requiring transposition [O, I, H, W] -> [O, H, W, I]
            if tensor.dim() == 4 and "conv" in k.lower():
                arr = mx.array(tensor.permute(0, 2, 3, 1).contiguous().numpy())
            else:
                arr = mx.array(tensor.contiguous().numpy())
            mlx_tensors[k] = arr

    logger.info("Saving converted weights to %s...", out_weights)
    mx.save_safetensors(str(out_weights), mlx_tensors)

    if src_config.exists():
        logger.info("Copying layout config to %s...", out_config)
        shutil.copy2(src_config, out_config)

    logger.info(
        "Layout MLX export complete: %d tensors written (%.1f MB).",
        len(mlx_tensors),
        out_weights.stat().st_size / (1024 * 1024),
    )
    return out_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export IndicDocLayout to Apple MLX format."
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
        default=Path(__file__).parent / "mlx_output" / "layout",
        help="Destination directory for exported MLX layout files.",
    )
    args = parser.parse_args()

    export_layout_mlx(
        bundle_dir=args.bundle_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
