#!/usr/bin/env python3
"""Validate ONNX model parity against PyTorch baseline for IndicOCR.

Compares:
  1. Stage 1 (IndicDocLayout):
     - Block detection counts
     - Label exact match
     - Reading order sequence match
     - Bounding box coordinate tolerances
     - Inference latency (PyTorch vs ONNX Runtime)
  2. Mismatch logging to JSON if any discrepancies occur.

Usage:
    indic-ocr/.venv/bin/python indic-ocr/validate_parity.py [--model-path onnx_output/layout/layout_model.onnx]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("validate_parity")


def validate_image_parity(
    image_path: Path,
    pt_infer_fn: Any,
    on_detector: Any,
    img_name: str,
) -> dict[str, Any]:
    """Run PyTorch and ONNX layout inference and compare results."""
    img = Image.open(image_path)

    # 1. PyTorch inference & timing
    t0 = time.perf_counter()
    pt_blocks = pt_infer_fn(img)
    t_pt = (time.perf_counter() - t0) * 1000.0

    # 2. ONNX inference & timing
    t0 = time.perf_counter()
    on_blocks = on_detector.detect(img)
    t_on = (time.perf_counter() - t0) * 1000.0

    mismatches: list[str] = []
    is_match = True

    # Count check
    if len(pt_blocks) != len(on_blocks):
        is_match = False
        mismatches.append(
            f"Count mismatch: PyTorch={len(pt_blocks)}, ONNX={len(on_blocks)}"
        )

    # Compare up to common count
    min_len = min(len(pt_blocks), len(on_blocks))
    max_box_delta = 0.0

    for i in range(min_len):
        pb = pt_blocks[i]
        ob = on_blocks[i]

        if pb.label != ob.label:
            is_match = False
            mismatches.append(f"Block {i} label mismatch: {pb.label} vs {ob.label}")

        if pb.order != ob.order:
            is_match = False
            mismatches.append(f"Block {i} order mismatch: {pb.order} vs {ob.order}")

        # Box delta
        box_delta = max(abs(p - o) for p, o in zip(pb.bbox_xyxy, ob.bbox_xyxy))
        if box_delta > max_box_delta:
            max_box_delta = box_delta

        if box_delta > 1.5:  # Tolerance in pixel space
            is_match = False
            mismatches.append(
                f"Block {i} box delta {box_delta:.2f}px exceeds tolerance: {pb.bbox_xyxy} vs {ob.bbox_xyxy}"
            )

    return {
        "image": img_name,
        "match": is_match,
        "pt_blocks": len(pt_blocks),
        "on_blocks": len(on_blocks),
        "max_box_delta_px": round(max_box_delta, 3),
        "pt_latency_ms": round(t_pt, 2),
        "on_latency_ms": round(t_on, 2),
        "speedup": round(t_pt / max(t_on, 0.001), 2),
        "mismatches": mismatches,
    }


def run_parity_suite(
    bundle_dir: Path,
    onnx_layout_path: Path,
    images_dir: Path,
    output_report: Path | None = None,
) -> bool:
    """Run parity verification across all benchmark images in images_dir."""
    sys.path.insert(0, str(bundle_dir))
    sys.path.insert(0, str(Path(__file__).parent))

    from idp_layout import IndicDocLayoutBackend
    from idp_types import LayoutConfig
    from layout_onnx import OnnxIndicDocLayout

    image_files: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_files.extend(images_dir.glob(ext))

    # Sort deterministically
    image_files = sorted(set(image_files))
    if not image_files:
        raise FileNotFoundError(f"No benchmark images found in: {images_dir}")

    logger.info("Initializing PyTorch baseline layout model on CPU...")
    layout_cfg = LayoutConfig(device="cpu")
    pt_backend = IndicDocLayoutBackend(str(bundle_dir / "weights" / "layout"), config=layout_cfg)

    logger.info("Initializing ONNX layout detector from %s...", onnx_layout_path.name)
    on_detector = OnnxIndicDocLayout(
        onnx_model_path=onnx_layout_path,
        bundle_dir=bundle_dir,
    )

    results: list[dict[str, Any]] = []
    all_passed = True

    logger.info("--- Starting Parity Validation on %d Images ---", len(image_files))

    for img_path in image_files:
        logger.info("Testing image: %s ...", img_path.name)
        res = validate_image_parity(
            image_path=img_path,
            pt_infer_fn=pt_backend.detect,
            on_detector=on_detector,
            img_name=img_path.name,
        )
        results.append(res)

        status_str = "PASS" if res["match"] else "FAIL"
        logger.info(
            "[%s] %s | PT: %d blocks (%.1fms) | ONNX: %d blocks (%.1fms, %.2fx) | Max Delta: %.2fpx",
            status_str,
            res["image"],
            res["pt_blocks"],
            res["pt_latency_ms"],
            res["on_blocks"],
            res["on_latency_ms"],
            res["speedup"],
            res["max_box_delta_px"],
        )

        if not res["match"]:
            all_passed = False
            for m in res["mismatches"]:
                logger.warning("  Mismatch: %s", m)

    # Generate summary report
    summary = {
        "model": onnx_layout_path.name,
        "total_images": len(image_files),
        "passed_images": sum(1 for r in results if r["match"]),
        "failed_images": sum(1 for r in results if not r["match"]),
        "avg_pt_latency_ms": round(
            float(np.mean([r["pt_latency_ms"] for r in results])), 2
        ),
        "avg_on_latency_ms": round(
            float(np.mean([r["on_latency_ms"] for r in results])), 2
        ),
        "details": results,
    }

    logger.info("=== Parity Validation Summary ===")
    logger.info("Passed: %d / %d", summary["passed_images"], summary["total_images"])
    logger.info(
        "Avg Latency: PyTorch = %.2fms | ONNX = %.2fms",
        summary["avg_pt_latency_ms"],
        summary["avg_on_latency_ms"],
    )

    if output_report:
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved parity report to %s", output_report)

    pt_backend.close()
    on_detector.close()
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ONNX parity for IndicOCR.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).parent / "model_bundle",
        help="Path to downloaded model bundle.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "images",
        help="Path to directory containing benchmark images.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).parent
        / "onnx_output"
        / "layout"
        / "layout_model.onnx",
        help="Path to exported ONNX model.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "parity_report.json",
        help="Path to save output JSON parity report.",
    )
    args = parser.parse_args()

    success = run_parity_suite(
        bundle_dir=args.bundle_dir.resolve(),
        onnx_layout_path=args.model_path.resolve(),
        images_dir=args.images_dir.resolve(),
        output_report=args.output_report.resolve(),
    )
    if not success:
        logger.warning("Parity suite reported one or more mismatches.")
        sys.exit(1)
    else:
        logger.info("All parity checks passed.")


if __name__ == "__main__":
    main()
