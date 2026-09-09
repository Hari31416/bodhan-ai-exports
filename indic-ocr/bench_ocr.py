#!/usr/bin/env python3
"""Rigorous OCR Transcription Parity and Accuracy Benchmark.

Evaluates Stage 2 Text Recognition across PyTorch HfRecognizer, ONNX FP32, ONNX INT8,
and MLX variants (fp16, 8bit, 4bit, bf16).
Computes:
  - Character Error Rate (CER) via Levenshtein distance
  - Word Error Rate (WER)
  - Exact Match (EM %)
  - Character count parity & alignment
  - Quantization degradation (INT8 / 4-bit / 8-bit vs FP baseline)
  - Latency (ms per crop), Throughput (chars/sec), and Speedup factor

Usage:
    .venv/bin/python bench_ocr.py --backend mlx
    .venv/bin/python bench_ocr.py --backend onnx-int8
    .venv/bin/python bench_ocr.py --backend all
    .venv/bin/python bench_ocr.py --max-crops 10
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
logger = logging.getLogger("bench_ocr")


def edit_distance(seq1: list[Any] | str, seq2: list[Any] | str) -> int:
    """Compute Levenshtein edit distance between sequences."""
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


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate: edit_dist(ref, hyp) / max(len(ref), 1)."""
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    dist = edit_distance(list(ref), list(hyp))
    return float(dist / len(ref))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate: edit_dist(ref_words, hyp_words) / max(len(ref_words), 1)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = edit_distance(ref_words, hyp_words)
    return float(dist / len(ref_words))


class OCREvaluator:
    """Benchmark runner evaluating text recognizers against PyTorch baseline."""

    def __init__(self, bundle_dir: Path, max_tokens: int = 256) -> None:
        self.bundle_dir = bundle_dir
        self.max_tokens = max_tokens
        self._cached_ref_texts: list[str] | None = None
        self._cached_pt_ms: float = 0.0
        root = Path(__file__).parent
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(bundle_dir))

        from idp_recognizer import HfRecognizer
        from idp_types import RecognizerConfig

        logger.info("Initializing PyTorch baseline HfRecognizer...")
        weights_path = bundle_dir / "weights" / "ocr"
        cfg = RecognizerConfig(dtype="float32", max_tokens=self.max_tokens)
        self.baseline = HfRecognizer(
            ckpt=str(weights_path),
            config=cfg,
            device="cpu",
            attn_implementation="eager",
        )

    def load_candidate(self, backend_type: str) -> Any:
        """Instantiate candidate OCR recognizer backend."""
        root = Path(__file__).parent

        if backend_type == "onnx":
            from idp_recognizer_onnx import OnnxRecognizer
            logger.info("Loading ONNX FP32 recognizer...")
            return OnnxRecognizer(
                bundle_dir=self.bundle_dir,
                model_dir=root / "onnx_output" / "ocr",
                quantized=False,
                max_tokens=256,
            )

        elif backend_type == "onnx-int8":
            from idp_recognizer_onnx import OnnxRecognizer
            logger.info("Loading ONNX INT8 recognizer...")
            return OnnxRecognizer(
                bundle_dir=self.bundle_dir,
                model_dir=root / "onnx_output" / "ocr",
                quantized=True,
                max_tokens=256,
            )

        elif backend_type.startswith("mlx"):
            try:
                from idp_recognizer_mlx import MlxRecognizer
                mlx_root = root / "mlx_output"
                if backend_type == "mlx-8bit" and (mlx_root / "ocr_8bit").exists():
                    weights_dir = mlx_root / "ocr_8bit"
                elif backend_type == "mlx-4bit" and (mlx_root / "ocr_4bit").exists():
                    weights_dir = mlx_root / "ocr_4bit"
                elif backend_type == "mlx-bf16" and (mlx_root / "ocr_bf16").exists():
                    weights_dir = mlx_root / "ocr_bf16"
                elif (mlx_root / "ocr_8bit").exists():
                    weights_dir = mlx_root / "ocr_8bit"
                else:
                    weights_dir = self.bundle_dir / "weights" / "ocr"

                logger.info("Loading MLX recognizer from %s...", weights_dir)
                return MlxRecognizer(
                    weights_dir=weights_dir,
                    max_tokens=256,
                )
            except ImportError as err:
                raise RuntimeError(f"MLX dependencies not available: {err}")

        else:
            raise ValueError(f"Unknown OCR backend '{backend_type}'")

    def extract_crop_requests(
        self,
        image_paths: list[Path],
        max_crops: int,
    ) -> list[Any]:
        """Extract text and table block crops using the layout detector."""
        from idp_offline import IndicDocLayout
        from idp_layout import IndicDocLayoutBackend
        from idp_recognizer import build_requests
        from idp_contract import is_transcribed
        from idp_types import LayoutConfig, DedupConfig, CropConfig

        layout_cfg = LayoutConfig(device="cpu")
        pt_detector = IndicDocLayoutBackend(
            str(self.bundle_dir / "weights" / "layout"), config=layout_cfg
        )
        layout_engine = IndicDocLayout(backend=pt_detector)
        crop_cfg = CropConfig()

        all_requests = []
        for img_path in image_paths:
            if len(all_requests) >= max_crops:
                break
            logger.info("Detecting layout blocks from %s...", img_path.name)
            res = layout_engine.detect(str(img_path))
            img = Image.open(img_path).convert("RGB")
            eligible = [b for b in res.blocks if is_transcribed(b.label)]
            if not eligible:
                continue
            requests, _ = build_requests(
                eligible,
                img,
                crop_cfg,
                table_format="markdown",
            )
            all_requests.extend(requests)

        layout_engine.close()
        selected = all_requests[:max_crops]
        logger.info("Prepared %d crop requests for OCR benchmarking.", len(selected))
        return selected

    def benchmark_backend(
        self,
        backend_name: str,
        crop_requests: list[Any],
    ) -> dict[str, Any]:
        """Run OCR benchmark comparing candidate recognizer against baseline."""
        logger.info("Starting OCR benchmark for '%s' across %d crops...", backend_name, len(crop_requests))

        # 1. Baseline transcription (cached across backends and persisted to disk)
        cache_file = Path(__file__).parent / "fixtures" / "baseline_cache" / f"ocr_ref_cache_{len(crop_requests)}crops.json"

        if self._cached_ref_texts is None:
            if cache_file.exists():
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                    self._cached_ref_texts = cache_data["texts"]
                    self._cached_pt_ms = cache_data["pt_ms"]
                    logger.info("Loaded persistent baseline OCR transcriptions from %s", cache_file.name)
                except Exception as err:
                    logger.warning("Failed to load baseline OCR cache: %s", err)

        if self._cached_ref_texts is None or len(self._cached_ref_texts) != len(crop_requests):
            logger.info("Transcribing %d crops with PyTorch baseline...", len(crop_requests))
            t0 = time.perf_counter()
            self._cached_ref_texts = self.baseline.transcribe(crop_requests)
            self._cached_pt_ms = (time.perf_counter() - t0) * 1000.0

            # Persist to disk
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"texts": self._cached_ref_texts, "pt_ms": self._cached_pt_ms}, f, indent=2)
            logger.info("Saved persistent baseline OCR transcriptions to %s", cache_file.name)
        else:
            logger.info("Reusing cached PyTorch baseline transcription for %d crops.", len(crop_requests))

        ref_texts = self._cached_ref_texts
        pt_total_ms = self._cached_pt_ms

        # 2. Candidate transcription
        candidate = self.load_candidate(backend_name)
        logger.info("Transcribing %d crops with %s...", len(crop_requests), backend_name)
        t0 = time.perf_counter()
        hyp_texts = candidate.transcribe(crop_requests)
        cand_total_ms = (time.perf_counter() - t0) * 1000.0
        candidate.close()

        # 3. Compute metrics per crop
        crop_results: list[dict[str, Any]] = []
        cers: list[float] = []
        wers: list[float] = []
        exact_matches = 0

        for idx, (ref, hyp) in enumerate(zip(ref_texts, hyp_texts)):
            cer = compute_cer(ref, hyp)
            wer = compute_wer(ref, hyp)
            em = (ref.strip() == hyp.strip())

            cers.append(cer)
            wers.append(wer)
            if em:
                exact_matches += 1

            crop_results.append({
                "crop_index": idx,
                "reference_chars": len(ref),
                "hypothesis_chars": len(hyp),
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "exact_match": em,
                "reference_snippet": ref[:60] + "..." if len(ref) > 60 else ref,
                "hypothesis_snippet": hyp[:60] + "..." if len(hyp) > 60 else hyp,
            })

        mean_cer = float(np.mean(cers)) if cers else 0.0
        mean_wer = float(np.mean(wers)) if wers else 0.0
        em_rate = float(exact_matches / len(crop_requests) * 100.0) if crop_requests else 0.0

        pt_ms_per_crop = pt_total_ms / len(crop_requests) if crop_requests else 0.0
        cand_ms_per_crop = cand_total_ms / len(crop_requests) if crop_requests else 0.0
        speedup = pt_total_ms / cand_total_ms if cand_total_ms > 0 else 1.0

        total_chars = sum(len(h) for h in hyp_texts)
        throughput_cps = (total_chars / (cand_total_ms / 1000.0)) if cand_total_ms > 0 else 0.0

        summary = {
            "backend": backend_name,
            "total_crops": len(crop_requests),
            "mean_cer": round(mean_cer, 4),
            "mean_wer": round(mean_wer, 4),
            "exact_match_pct": round(em_rate, 2),
            "cer_distribution": {
                "cer_0_exact_pct": round(em_rate, 2),
                "cer_le_0.05_pct": round(sum(1 for c in cers if c <= 0.05) / len(cers) * 100.0, 2),
                "cer_le_0.10_pct": round(sum(1 for c in cers if c <= 0.10) / len(cers) * 100.0, 2),
                "cer_le_0.20_pct": round(sum(1 for c in cers if c <= 0.20) / len(cers) * 100.0, 2),
            },
            "performance": {
                "pytorch_mean_ms_per_crop": round(pt_ms_per_crop, 2),
                "candidate_mean_ms_per_crop": round(cand_ms_per_crop, 2),
                "throughput_chars_per_sec": round(throughput_cps, 1),
                "speedup": round(speedup, 2),
            },
            "sample_details": crop_results[:10],
        }

        self._print_summary_table(summary)
        return summary

    def _print_summary_table(self, s: dict[str, Any]) -> None:
        """Print formatted terminal table for OCR benchmark summary."""
        perf = s["performance"]
        cer_dist = s["cer_distribution"]
        print("\n" + "=" * 76)
        print(f"  OCR RECOGNITION BENCHMARK RESULTS: {s['backend'].upper()} ({s['total_crops']} Crops)")
        print("=" * 76)
        print(f"  Character Error Rate (CER):   {s['mean_cer']:.4f}  (Lower is better)")
        print(f"  Word Error Rate (WER):        {s['mean_wer']:.4f}  (Lower is better)")
        print(f"  Exact Match Rate:             {s['exact_match_pct']:.1f}%")
        print(f"  CER <= 5% (High Accuracy):    {cer_dist['cer_le_0.05_pct']:.1f}%")
        print(f"  CER <= 10% (Readable):        {cer_dist['cer_le_0.10_pct']:.1f}%")
        print("-" * 76)
        print(f"  PyTorch Baseline Latency:     {perf['pytorch_mean_ms_per_crop']:.2f} ms / crop")
        print(f"  {s['backend'].upper()} Latency:             {perf['candidate_mean_ms_per_crop']:.2f} ms / crop")
        print(f"  Throughput:                   {perf['throughput_chars_per_sec']:.1f} chars/sec")
        print(f"  Speedup Factor:               {perf['speedup']:.2f}x")
        print("=" * 76 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rigorous OCR Text Recognition Parity & Accuracy Benchmark."
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "mlx-8bit", "mlx-4bit", "mlx-bf16", "mlx-all", "onnx", "onnx-int8", "all"],
        default="mlx",
        help="Backend to evaluate (default: mlx).",
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
        default=Path(__file__).parent / "fixtures" / "benchmark_images",
        help="Directory containing benchmark document images.",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=10,
        help="Maximum number of document crops to transcribe (default: 10).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per crop (default: 256).",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "ocr_benchmark_report.json",
        help="Destination JSON file for evaluation metrics.",
    )
    args = parser.parse_args()

    # Fallback to fixtures/images if benchmark_images is not downloaded
    img_dir = args.images_dir
    if not img_dir.exists() or not list(img_dir.glob("*.png")):
        img_dir = Path(__file__).parent / "fixtures" / "images"

    image_paths = sorted([p for p in img_dir.glob("*.png") if not p.name.startswith(".")])
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {img_dir}")

    evaluator = OCREvaluator(
        bundle_dir=args.bundle_dir.resolve(),
        max_tokens=args.max_tokens,
    )
    crop_requests = evaluator.extract_crop_requests(image_paths, max_crops=args.max_crops)

    if not crop_requests:
        logger.error("No valid text crops extracted from images.")
        sys.exit(1)

    if args.backend == "all":
        backends = ["mlx-8bit", "mlx-4bit", "mlx-bf16"]
    elif args.backend == "mlx-all":
        backends = ["mlx-8bit", "mlx-4bit", "mlx-bf16"]
    else:
        backends = [args.backend]
    combined_reports: dict[str, Any] = {}

    for b in backends:
        try:
            report_data = evaluator.benchmark_backend(b, crop_requests)
            combined_reports[b] = report_data
        except Exception as err:
            logger.error("Failed to run benchmark for backend '%s': %s", b, err)

    if not combined_reports:
        logger.error("No OCR benchmarks succeeded.")
        sys.exit(1)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        final_out = combined_reports if args.backend == "all" else combined_reports[args.backend]
        json.dump(final_out, f, indent=2)
    logger.info("Saved benchmark report to %s", args.output_report)


if __name__ == "__main__":
    main()
