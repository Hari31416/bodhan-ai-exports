#!/usr/bin/env python3
"""Rigorous Bounding Box and Layout Parity / Accuracy Benchmark.

Evaluates Stage 1 Layout Detection across PyTorch, ONNX FP32, ONNX INT8, and MLX.
Computes:
  - Mean & Min IoU (Intersection over Union)
  - IoU distributions (>=0.50, >=0.75, >=0.90, >=0.95)
  - Max and Mean Coordinate Drift in pixels
  - Class label exact match rate
  - Reading order sequence match rate & edit distance
  - Precision, Recall, and F1 score relative to PyTorch baseline
  - Quantization degradation (INT8 vs FP32 baseline)
  - Latency (ms), Throughput (FPS), and Speedup factor

Usage:
    .venv/bin/python bench_bbox.py --backend onnx
    .venv/bin/python bench_bbox.py --backend onnx-int8
    .venv/bin/python bench_bbox.py --backend mlx
    .venv/bin/python bench_bbox.py --all
    .venv/bin/python bench_bbox.py --backend onnx --images-dir fixtures/benchmark_images
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bench_bbox")


def box_iou(box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]) -> float:
    """Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def box_coordinate_delta(box_a: list[float] | tuple[float, ...], box_b: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """Compute (max_delta, mean_delta) in pixels across [x1, y1, x2, y2]."""
    diffs = [abs(a - b) for a, b in zip(box_a, box_b)]
    return float(max(diffs)), float(sum(diffs) / len(diffs))


def levenshtein_distance(seq1: list[Any], seq2: list[Any]) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


class BBoxEvaluator:
    """Benchmark runner evaluating layout detection backends against PyTorch baseline."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir
        root = Path(__file__).parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(bundle_dir))

        from idp_offline import IndicDocLayout
        from idp_layout import IndicDocLayoutBackend
        from idp_types import LayoutConfig

        logger.info("Initializing PyTorch baseline layout model...")
        weights_path = bundle_dir / "weights" / "layout"
        layout_cfg = LayoutConfig(device="cpu")
        pt_detector = IndicDocLayoutBackend(str(weights_path), config=layout_cfg)
        self.baseline = IndicDocLayout(backend=pt_detector)

        self.cache_path = root / "fixtures" / "baseline_cache" / "layout_pt_cache.pkl"
        self._pt_cache: dict[str, tuple[Any, float]] = {}
        if self.cache_path.exists():
            try:
                import pickle
                with open(self.cache_path, "rb") as f:
                    self._pt_cache = pickle.load(f)
                logger.info("Loaded %d cached PyTorch baseline layout predictions from %s", len(self._pt_cache), self.cache_path.name)
            except Exception as err:
                logger.warning("Could not load baseline layout cache: %s", err)

    def save_cache(self) -> None:
        """Persist cached baseline layout inferences to disk."""
        if self._pt_cache:
            import pickle
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._pt_cache, f)
            logger.info("Saved %d baseline predictions to cache at %s", len(self._pt_cache), self.cache_path.name)

    def load_candidate(self, backend_type: str, model_path: Path | None = None) -> Any:
        """Instantiate candidate layout detector backend."""
        from idp_offline import IndicDocLayout
        root = Path(__file__).parent

        if backend_type == "onnx":
            from layout_onnx import OnnxIndicDocLayout
            target_path = model_path or (root / "onnx_output" / "layout" / "layout_model.onnx")
            logger.info("Loading ONNX FP32 detector from %s...", target_path)
            backend = OnnxIndicDocLayout(onnx_model_path=target_path, bundle_dir=self.bundle_dir)
            return IndicDocLayout(backend=backend)

        elif backend_type == "onnx-int8":
            from layout_onnx import OnnxIndicDocLayout
            target_path = model_path or (root / "onnx_output" / "layout" / "layout_model_int8.onnx")
            logger.info("Loading ONNX INT8 detector from %s...", target_path)
            backend = OnnxIndicDocLayout(onnx_model_path=target_path, bundle_dir=self.bundle_dir)
            return IndicDocLayout(backend=backend)

        elif backend_type == "mlx":
            try:
                from layout_mlx import MlxIndicDocLayout
                logger.info("Loading MLX layout detector...")
                backend = MlxIndicDocLayout(bundle_dir=self.bundle_dir)
                return IndicDocLayout(backend=backend)
            except ImportError as err:
                raise RuntimeError(f"MLX dependencies not available: {err}")

        else:
            raise ValueError(f"Unknown backend '{backend_type}'")

    def evaluate_image(
        self,
        image_path: Path,
        candidate_detector: Any,
    ) -> dict[str, Any]:
        """Compare bounding box predictions on a single document image."""
        path_str = str(image_path)

        # 1. Baseline inference & timing (cached)
        if path_str in self._pt_cache:
            pt_result, pt_ms = self._pt_cache[path_str]
        else:
            t0 = time.perf_counter()
            pt_result = self.baseline.detect(path_str)
            pt_ms = (time.perf_counter() - t0) * 1000.0
            self._pt_cache[path_str] = (pt_result, pt_ms)

        # 2. Candidate inference & timing
        t0 = time.perf_counter()
        cand_result = candidate_detector.detect(path_str)
        cand_ms = (time.perf_counter() - t0) * 1000.0

        pt_blocks = pt_result.blocks
        cand_blocks = cand_result.blocks

        # 3. Match candidate boxes to baseline boxes (Greedy bipartite matching by max IoU)
        matched_pairs: list[tuple[int, int, float]] = []
        unmatched_cand = set(range(len(cand_blocks)))
        unmatched_pt = set(range(len(pt_blocks)))

        ious_matrix = np.zeros((len(pt_blocks), len(cand_blocks)), dtype=np.float32)
        for i, pb in enumerate(pt_blocks):
            for j, cb in enumerate(cand_blocks):
                ious_matrix[i, j] = box_iou(pb.bbox_xyxy, cb.bbox_xyxy)

        while unmatched_pt and unmatched_cand:
            best_val = -1.0
            best_i, best_j = -1, -1
            for i in unmatched_pt:
                for j in unmatched_cand:
                    if ious_matrix[i, j] > best_val:
                        best_val = ious_matrix[i, j]
                        best_i, best_j = i, j
            if best_val < 0.1:  # No significant overlap remaining
                break
            matched_pairs.append((best_i, best_j, float(best_val)))
            unmatched_pt.remove(best_i)
            unmatched_cand.remove(best_j)

        # 4. Compute metrics
        matched_ious = [iou for _, _, iou in matched_pairs]
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
        min_iou = float(np.min(matched_ious)) if matched_ious else 0.0

        coord_deltas = [
            box_coordinate_delta(pt_blocks[i].bbox_xyxy, cand_blocks[j].bbox_xyxy)
            for i, j, _ in matched_pairs
        ]
        max_px_delta = float(max(d[0] for d in coord_deltas)) if coord_deltas else 0.0
        mean_px_delta = float(np.mean([d[1] for d in coord_deltas])) if coord_deltas else 0.0

        # Label matching
        label_matches = sum(
            1 for i, j, _ in matched_pairs if pt_blocks[i].label == cand_blocks[j].label
        )
        label_match_rate = float(label_matches / len(matched_pairs)) if matched_pairs else 1.0

        # Reading order sequence matching
        pt_order = [b.order for b in pt_blocks]
        cand_order = [b.order for b in cand_blocks]
        order_match = pt_order == cand_order
        order_edit_dist = levenshtein_distance(pt_order, cand_order)

        # Detection Classification Counts (IoU >= 0.5 threshold)
        tp = sum(1 for iou in matched_ious if iou >= 0.5)
        fp = len(cand_blocks) - tp
        fn = len(pt_blocks) - tp
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 1.0

        return {
            "image": image_path.name,
            "pt_blocks": len(pt_blocks),
            "cand_blocks": len(cand_blocks),
            "mean_iou": round(mean_iou, 4),
            "min_iou": round(min_iou, 4),
            "max_px_delta": round(max_px_delta, 2),
            "mean_px_delta": round(mean_px_delta, 2),
            "label_match_rate": round(label_match_rate * 100.0, 2),
            "order_match": order_match,
            "order_edit_dist": order_edit_dist,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "pt_latency_ms": round(pt_ms, 2),
            "cand_latency_ms": round(cand_ms, 2),
            "speedup": round(pt_ms / cand_ms, 2) if cand_ms > 0 else 1.0,
        }

    def benchmark_backend(
        self,
        backend_name: str,
        image_paths: list[Path],
    ) -> dict[str, Any]:
        """Run benchmark across all image paths for a given backend."""
        logger.info("Starting BBox benchmark for '%s' across %d images...", backend_name, len(image_paths))
        candidate = self.load_candidate(backend_name)

        per_image_results: list[dict[str, Any]] = []
        for idx, img_path in enumerate(image_paths, 1):
            res = self.evaluate_image(img_path, candidate)
            per_image_results.append(res)
            logger.info(
                "[%d/%d] %s | IoU: %.3f | Max px: %.1f | Order match: %s | Speedup: %.2fx",
                idx,
                len(image_paths),
                img_path.name,
                res["mean_iou"],
                res["max_px_delta"],
                res["order_match"],
                res["speedup"],
            )

        # Aggregate stats
        all_ious = [r["mean_iou"] for r in per_image_results]
        all_max_deltas = [r["max_px_delta"] for r in per_image_results]
        all_mean_deltas = [r["mean_px_delta"] for r in per_image_results]
        all_pt_latencies = [r["pt_latency_ms"] for r in per_image_results]
        all_cand_latencies = [r["cand_latency_ms"] for r in per_image_results]

        avg_iou = float(np.mean(all_ious))
        min_overall_iou = float(np.min(all_ious))
        max_overall_delta = float(np.max(all_max_deltas))
        mean_overall_delta = float(np.mean(all_mean_deltas))

        iou_50 = sum(1 for iou in all_ious if iou >= 0.50) / len(all_ious) * 100.0
        iou_75 = sum(1 for iou in all_ious if iou >= 0.75) / len(all_ious) * 100.0
        iou_90 = sum(1 for iou in all_ious if iou >= 0.90) / len(all_ious) * 100.0
        iou_95 = sum(1 for iou in all_ious if iou >= 0.95) / len(all_ious) * 100.0

        label_match_pct = float(np.mean([r["label_match_rate"] for r in per_image_results]))
        order_match_pct = sum(1 for r in per_image_results if r["order_match"]) / len(per_image_results) * 100.0
        mean_f1 = float(np.mean([r["f1"] for r in per_image_results]))

        avg_pt_ms = float(np.mean(all_pt_latencies))
        avg_cand_ms = float(np.mean(all_cand_latencies))
        avg_speedup = avg_pt_ms / avg_cand_ms if avg_cand_ms > 0 else 1.0

        summary = {
            "backend": backend_name,
            "total_images": len(image_paths),
            "mean_iou": round(avg_iou, 4),
            "min_iou": round(min_overall_iou, 4),
            "iou_distribution": {
                "ge_0.50_pct": round(iou_50, 1),
                "ge_0.75_pct": round(iou_75, 1),
                "ge_0.90_pct": round(iou_90, 1),
                "ge_0.95_pct": round(iou_95, 1),
            },
            "max_px_delta": round(max_overall_delta, 2),
            "mean_px_delta": round(mean_overall_delta, 2),
            "label_match_rate_pct": round(label_match_pct, 2),
            "order_sequence_match_pct": round(order_match_pct, 2),
            "mean_f1_score": round(mean_f1, 4),
            "latency": {
                "pytorch_mean_ms": round(avg_pt_ms, 2),
                "candidate_mean_ms": round(avg_cand_ms, 2),
                "fps": round(1000.0 / avg_cand_ms, 2) if avg_cand_ms > 0 else 0.0,
                "speedup": round(avg_speedup, 2),
            },
            "details": per_image_results,
        }

        self._print_summary_table(summary)
        return summary

    def _print_summary_table(self, s: dict[str, Any]) -> None:
        """Print formatted terminal table for benchmark summary."""
        lat = s["latency"]
        iou_dist = s["iou_distribution"]
        print("\n" + "=" * 76)
        print(f"  LAYOUT BBOX BENCHMARK RESULTS: {s['backend'].upper()} ({s['total_images']} Images)")
        print("=" * 76)
        print(f"  Mean IoU:                  {s['mean_iou']:.4f}  (Min: {s['min_iou']:.4f})")
        print(f"  IoU >= 0.75:               {iou_dist['ge_0.75_pct']:.1f}%")
        print(f"  IoU >= 0.95:               {iou_dist['ge_0.95_pct']:.1f}%")
        print(f"  Max Pixel Delta:           {s['max_px_delta']:.2f} px")
        print(f"  Mean Pixel Delta:          {s['mean_px_delta']:.2f} px")
        print(f"  Class Label Match Rate:    {s['label_match_rate_pct']:.1f}%")
        print(f"  Reading Order Match Rate:  {s['order_sequence_match_pct']:.1f}%")
        print(f"  Detection Mean F1 Score:   {s['mean_f1_score']:.4f}")
        print("-" * 76)
        print(f"  PyTorch Baseline Latency:  {lat['pytorch_mean_ms']:.2f} ms")
        print(f"  {s['backend'].upper()} Latency:          {lat['candidate_mean_ms']:.2f} ms ({lat['fps']:.1f} FPS)")
        print(f"  Speedup Factor:            {lat['speedup']:.2f}x")
        print("=" * 76 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rigorous Layout Bounding Box & Reading Order Benchmark."
    )
    parser.add_argument(
        "--backend",
        choices=["onnx", "onnx-int8", "onnx-both", "mlx", "all"],
        default="onnx",
        help="Backend to evaluate: onnx, onnx-int8, onnx-both, mlx, or all (default: onnx).",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).parent / "model_bundle",
        help="Path to model bundle directory.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing benchmark document images (default: fixtures/benchmark_images or fixtures/images).",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "bbox_benchmark_report.json",
        help="Destination JSON file for evaluation metrics.",
    )
    args = parser.parse_args()

    # Determine images directory
    img_dir = args.images_dir
    if img_dir is None:
        bench_dir = Path(__file__).parent / "fixtures" / "benchmark_images"
        if bench_dir.exists() and list(bench_dir.glob("*.png")):
            img_dir = bench_dir
        else:
            img_dir = Path(__file__).parent / "fixtures" / "images"

    image_paths = sorted([p for p in img_dir.glob("*.png") if not p.name.startswith(".")])
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    evaluator = BBoxEvaluator(bundle_dir=args.bundle_dir.resolve())

    if args.backend == "all":
        backends = ["onnx", "onnx-int8", "mlx"]
    elif args.backend == "onnx-both":
        backends = ["onnx", "onnx-int8"]
    else:
        backends = [args.backend]

    combined_reports: dict[str, Any] = {}

    for b in backends:
        try:
            report_data = evaluator.benchmark_backend(b, image_paths)
            combined_reports[b] = report_data
        except Exception as err:
            logger.error("Failed to run benchmark for backend '%s': %s", b, err)

    if not combined_reports:
        logger.error("No backend benchmarks succeeded.")
        sys.exit(1)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    # If benchmarking both ONNX variants, also save individual variant files for convenience
    if "onnx" in combined_reports:
        fp32_report = args.output_report.parent / "bbox_benchmark_report_fp32.json"
        with open(fp32_report, "w", encoding="utf-8") as f:
            json.dump(combined_reports["onnx"], f, indent=2)
        logger.info("Saved FP32 benchmark report to %s", fp32_report)

    if "onnx-int8" in combined_reports:
        int8_report = args.output_report.parent / "bbox_benchmark_report_int8.json"
        with open(int8_report, "w", encoding="utf-8") as f:
            json.dump(combined_reports["onnx-int8"], f, indent=2)
        logger.info("Saved INT8 benchmark report to %s", int8_report)

    with open(args.output_report, "w", encoding="utf-8") as f:
        final_out = combined_reports if len(backends) > 1 else combined_reports[backends[0]]
        json.dump(final_out, f, indent=2)
    logger.info("Saved benchmark report to %s", args.output_report)

    evaluator.save_cache()


if __name__ == "__main__":
    main()
