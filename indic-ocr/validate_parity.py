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
    pt_backend: Any,
    on_detector: Any,
    img_name: str,
    decode_order_fn: Any,
) -> dict[str, Any]:
    """Run PyTorch and ONNX layout inference and compare raw vectors and blocks."""
    img = Image.open(image_path)
    im_1024 = img.convert("RGB").resize((1024, 1024))
    x_np = np.expand_dims(
        np.array(im_1024, dtype=np.float32).transpose(2, 0, 1) / 255.0, axis=0
    )

    # 1. PyTorch raw vectors & block detection
    t0 = time.perf_counter()
    with torch.no_grad():
        pt_out = pt_backend.model(pixel_values=torch.from_numpy(x_np))
        pt_scores, _ = pt_out.logits.sigmoid().max(-1)
        pt_keep = (pt_scores[0] > 0.5).nonzero().squeeze(-1)
        pt_sub_order = (
            pt_out.order_logits[0, pt_keep][:, pt_keep].cpu().float()
            if pt_keep.numel() > 0
            else torch.empty((0, 0))
        )
        pt_seq = decode_order_fn(pt_sub_order).tolist() if pt_keep.numel() > 0 else []
    pt_blocks = pt_backend.detect(img)
    t_pt = (time.perf_counter() - t0) * 1000.0

    # 2. ONNX raw vectors & block detection
    t0 = time.perf_counter()
    on_logits, on_boxes, on_order = on_detector.session.run(
        None, {"pixel_values": x_np}
    )
    on_scores, _ = torch.from_numpy(on_logits).sigmoid().max(-1)
    on_keep = (on_scores[0] > 0.5).nonzero().squeeze(-1)
    on_sub_order = (
        torch.from_numpy(on_order[0, on_keep.numpy()][:, on_keep.numpy()]).float()
        if on_keep.numel() > 0
        else torch.empty((0, 0))
    )
    on_seq = decode_order_fn(on_sub_order).tolist() if on_keep.numel() > 0 else []
    on_blocks = on_detector.detect(img)
    t_on = (time.perf_counter() - t0) * 1000.0

    mismatches: list[str] = []
    is_match = True

    # 3. Vector Output Parity: One-to-one Query Indices Matching
    indices_match = bool(torch.equal(pt_keep, on_keep))
    if not indices_match:
        is_match = False
        mismatches.append(
            f"Query indices mismatch: PyTorch={pt_keep.tolist()} vs ONNX={on_keep.tolist()}"
        )

    # 4. Reading order sequence matching
    order_seq_match = pt_seq == on_seq
    if not order_seq_match:
        is_match = False
        mismatches.append(
            f"Reading order permutation mismatch: PyTorch={pt_seq} vs ONNX={on_seq}"
        )

    # 5. Raw tensor vector deltas
    if indices_match and pt_keep.numel() > 0:
        keep_idx = pt_keep.numpy()
        box_vec_delta = float(
            np.max(np.abs(pt_out.pred_boxes[0, pt_keep].numpy() - on_boxes[0, keep_idx]))
        )
        logits_vec_delta = float(
            np.max(np.abs(pt_out.logits[0, pt_keep].numpy() - on_logits[0, keep_idx]))
        )
        order_vec_delta = (
            float(np.max(np.abs(pt_sub_order.numpy() - on_sub_order.numpy())))
            if pt_sub_order.numel() > 0
            else 0.0
        )
    else:
        box_vec_delta = float(np.max(np.abs(pt_out.pred_boxes.numpy() - on_boxes)))
        logits_vec_delta = float(np.max(np.abs(pt_out.logits.numpy() - on_logits)))
        order_vec_delta = float(np.max(np.abs(pt_out.order_logits.numpy() - on_order)))

    if box_vec_delta > 1e-3:
        is_match = False
        mismatches.append(
            f"Normalized box vector delta {box_vec_delta:.6e} exceeds tolerance 1e-3"
        )
    if logits_vec_delta > 0.1:
        is_match = False
        mismatches.append(
            f"Logits vector delta {logits_vec_delta:.4f} exceeds tolerance 0.1"
        )

    # 6. Post-processed Block and coordinate check
    if len(pt_blocks) != len(on_blocks):
        is_match = False
        mismatches.append(
            f"Count mismatch: PyTorch={len(pt_blocks)}, ONNX={len(on_blocks)}"
        )

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
        "indices_match": indices_match,
        "order_seq_match": order_seq_match,
        "pt_blocks": len(pt_blocks),
        "on_blocks": len(on_blocks),
        "max_box_delta_px": round(max_box_delta, 3),
        "max_box_vec_delta": round(box_vec_delta, 6),
        "max_logits_delta": round(logits_vec_delta, 4),
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
    layout_weights = bundle_dir / "weights" / "layout"
    if not layout_weights.exists() or not (bundle_dir / "idp_layout.py").exists():
        raise FileNotFoundError(
            f"Model bundle not found at '{bundle_dir}'. "
            "Please run 'make download-weights' (or 'python download_weights.py') first."
        )

    sys.path.insert(0, str(bundle_dir))
    sys.path.insert(0, str(Path(__file__).parent))

    from idp_layout import IndicDocLayoutBackend
    from idp_model_order_loss import decode_order
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
            pt_backend=pt_backend,
            on_detector=on_detector,
            img_name=img_path.name,
            decode_order_fn=decode_order,
        )
        results.append(res)

        status_str = "PASS" if res["match"] else "FAIL"
        logger.info(
            "[%s] %s | PT: %d blocks (%.1fms) | ONNX: %d blocks (%.1fms, %.2fx) | Indices: %s | Max Box px: %.2fpx | Box Vec: %.6f | Logits: %.4f",
            status_str,
            res["image"],
            res["pt_blocks"],
            res["pt_latency_ms"],
            res["on_blocks"],
            res["on_latency_ms"],
            res["speedup"],
            "EXACT" if res["indices_match"] else "MISMATCH",
            res["max_box_delta_px"],
            res["max_box_vec_delta"],
            res["max_logits_delta"],
        )

        if not res["match"]:
            all_passed = False
            for m in res["mismatches"]:
                logger.warning("  Mismatch: %s", m)

    # Generate summary report
    indices_pass_count = sum(1 for r in results if r["indices_match"])
    order_seq_pass_count = sum(1 for r in results if r["order_seq_match"])
    summary = {
        "model": onnx_layout_path.name,
        "total_images": len(image_files),
        "passed_images": sum(1 for r in results if r["match"]),
        "failed_images": sum(1 for r in results if not r["match"]),
        "indices_pass_count": indices_pass_count,
        "indices_pass_rate": round(indices_pass_count / len(results) * 100, 2),
        "order_seq_pass_count": order_seq_pass_count,
        "order_seq_pass_rate": round(order_seq_pass_count / len(results) * 100, 2),
        "max_overall_box_delta_px": round(max(r["max_box_delta_px"] for r in results), 3),
        "max_overall_box_vec_delta": round(max(r["max_box_vec_delta"] for r in results), 6),
        "max_overall_logits_delta": round(max(r["max_logits_delta"] for r in results), 4),
        "avg_pt_latency_ms": round(
            float(np.mean([r["pt_latency_ms"] for r in results])), 2
        ),
        "avg_on_latency_ms": round(
            float(np.mean([r["on_latency_ms"] for r in results])), 2
        ),
        "details": results,
    }

    logger.info("=== Parity Validation Summary ===")
    logger.info("Overall Passed: %d / %d", summary["passed_images"], summary["total_images"])
    logger.info(
        "Query Indices Match: %d / %d (%.1f%%)",
        summary["indices_pass_count"],
        summary["total_images"],
        summary["indices_pass_rate"],
    )
    logger.info(
        "Reading Order Match: %d / %d (%.1f%%)",
        summary["order_seq_pass_count"],
        summary["total_images"],
        summary["order_seq_pass_rate"],
    )
    logger.info(
        "Max Deltas: Box px = %.2fpx | Box Vec = %.6f | Logits Vec = %.4f",
        summary["max_overall_box_delta_px"],
        summary["max_overall_box_vec_delta"],
        summary["max_overall_logits_delta"],
    )
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
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Do not exit with non-zero code on mismatches (useful for diagnostic reports).",
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
        if not args.warn_only:
            sys.exit(1)
    else:
        logger.info("All parity checks passed.")


if __name__ == "__main__":
    main()
