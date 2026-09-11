#!/usr/bin/env python3
"""Validate PyTorch MPS layout parity against a PyTorch CPU baseline for IndicOCR.

Compares Stage 1 (IndicDocLayout):
  - Block detection counts
  - Class label exact match
  - Reading order sequence match
  - Bounding box coordinate tolerances (< 1.5px)
  - Inference latency (PyTorch CPU vs PyTorch MPS)

Usage:
    indic-ocr/.venv/bin/python indic-ocr/validate_parity_mlx.py \
        --images-dir indic-ocr/fixtures/images \
        --output-report indic-ocr/fixtures/parity_report_mlx.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_parity_mlx")


def validate_image_parity(
    image_path: Path,
    pt_backend: Any,
    mlx_detector: Any,
    img_name: str,
) -> dict[str, Any]:
    """Run PyTorch and MLX layout inference and compare detected blocks."""
    img = Image.open(image_path)

    # 1. PyTorch CPU baseline
    t0 = time.perf_counter()
    pt_blocks = pt_backend.detect(img)
    t_pt = (time.perf_counter() - t0) * 1000.0

    # 2. MLX backend
    t0 = time.perf_counter()
    mlx_blocks = mlx_detector.detect(img)
    t_mlx = (time.perf_counter() - t0) * 1000.0

    mismatches: list[str] = []
    is_match = True

    # Check block counts
    if len(pt_blocks) != len(mlx_blocks):
        is_match = False
        mismatches.append(
            f"Block count mismatch: PyTorch={len(pt_blocks)} vs MLX={len(mlx_blocks)}"
        )

    # Check per-block labels and box coordinates
    max_box_delta = 0.0
    num_compared = min(len(pt_blocks), len(mlx_blocks))

    for i in range(num_compared):
        pt_b = pt_blocks[i]
        mlx_b = mlx_blocks[i]

        if pt_b.label != mlx_b.label:
            is_match = False
            mismatches.append(
                f"Block {i} label mismatch: PyTorch='{pt_b.label}' vs MLX='{mlx_b.label}'"
            )

        if pt_b.order != mlx_b.order:
            is_match = False
            mismatches.append(
                f"Block {i} order mismatch: PyTorch={pt_b.order} vs MLX={mlx_b.order}"
            )

        delta = max(abs(a - b) for a, b in zip(pt_b.bbox_xyxy, mlx_b.bbox_xyxy))
        if delta > max_box_delta:
            max_box_delta = float(delta)

        if delta > 1.5:
            is_match = False
            mismatches.append(
                f"Block {i} box delta {delta:.2f}px exceeds tolerance: "
                f"PyTorch={pt_b.bbox_xyxy} vs MLX={mlx_b.bbox_xyxy}"
            )

    return {
        "image": img_name,
        "is_match": is_match,
        "mismatches": mismatches,
        "pt_blocks": len(pt_blocks),
        "mlx_blocks": len(mlx_blocks),
        "max_box_delta_px": round(max_box_delta, 4),
        "latency_ms": {
            "pytorch": round(t_pt, 2),
            "mlx": round(t_mlx, 2),
            "speedup": round(t_pt / t_mlx, 2) if t_mlx > 0 else 1.0,
        },
    }


def run_parity_suite(
    bundle_dir: Path,
    images_dir: Path,
    output_report: Path,
    device: str = "mps",
    precision: str = "bf16",
) -> None:
    """Run full parity verification across document fixtures."""
    sys.path.insert(0, str(bundle_dir))
    sys.path.insert(0, str(bundle_dir.parent))

    from idp_layout import IndicDocLayoutBackend
    from idp_types import LayoutConfig
    from layout_mlx import MlxIndicDocLayout

    logger.info("Initializing PyTorch CPU baseline backend...")
    pt_config = LayoutConfig(device="cpu")
    pt_backend = IndicDocLayoutBackend(
        ckpt=str(bundle_dir / "weights" / "layout"),
        config=pt_config,
    )

    logger.info("Initializing PyTorch MPS layout backend on %s...", device)
    mlx_detector = MlxIndicDocLayout(
        bundle_dir=bundle_dir,
        device=device,
    )

    images = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    if not images:
        raise FileNotFoundError(f"No test images found in {images_dir}")

    logger.info("Running parity verification across %d document images...", len(images))

    results = []
    total_pt_time = 0.0
    total_mlx_time = 0.0
    passed_count = 0

    for idx, img_path in enumerate(images):
        logger.info("[%d/%d] Testing %s...", idx + 1, len(images), img_path.name)
        res = validate_image_parity(img_path, pt_backend, mlx_detector, img_path.name)
        results.append(res)

        total_pt_time += res["latency_ms"]["pytorch"]
        total_mlx_time += res["latency_ms"]["mlx"]

        if res["is_match"]:
            passed_count += 1
            logger.info(
                "  PASS | Blocks: %d | Max Delta: %.2fpx | PT: %.1fms | MLX: %.1fms (%.2fx)",
                res["pt_blocks"],
                res["max_box_delta_px"],
                res["latency_ms"]["pytorch"],
                res["latency_ms"]["mlx"],
                res["latency_ms"]["speedup"],
            )
        else:
            logger.warning("  FAIL | %s", res["mismatches"])

    summary = {
        "model": "bodhan-ai/indic-ocr",
        "precision": precision,
        "runtime": "PyTorch on Apple MPS",
        "total_images": len(images),
        "passed": passed_count,
        "failed": len(images) - passed_count,
        "pass_rate_percent": round(passed_count / len(images) * 100.0, 1),
        "avg_latency_ms": {
            "pytorch": round(total_pt_time / len(images), 2),
            "mlx": round(total_mlx_time / len(images), 2),
            "speedup": (
                round(total_pt_time / total_mlx_time, 2) if total_mlx_time > 0 else 1.0
            ),
        },
        "results": results,
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved parity report to %s", output_report)

    logger.info("================ Parity Summary ================")
    logger.info(
        "Passed: %d / %d (%.1f%%)",
        passed_count,
        len(images),
        summary["pass_rate_percent"],
    )
    logger.info(
        "Average Latency: PyTorch = %.2f ms | MLX = %.2f ms (%.2fx speedup)",
        summary["avg_latency_ms"]["pytorch"],
        summary["avg_latency_ms"]["mlx"],
        summary["avg_latency_ms"]["speedup"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PyTorch MPS layout parity against the PyTorch CPU baseline."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).parent / "model_bundle",
        help="Path to model bundle containing downloaded weights.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "images",
        help="Directory of benchmark document images.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "parity_report_mlx.json",
        help="Path to save JSON parity benchmark report.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "8bit", "4bit"],
        help="Precision variant tested (default: bf16).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Compute device for MLX backend (mps/cpu).",
    )
    args = parser.parse_args()

    run_parity_suite(
        bundle_dir=args.bundle_dir.resolve(),
        images_dir=args.images_dir.resolve(),
        output_report=args.output_report.resolve(),
        device=args.device,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
