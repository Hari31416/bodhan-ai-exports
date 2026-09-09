#!/usr/bin/env python3
"""End-to-end OCR pipeline using ONNX-accelerated layout detection and IndicBlockOCR.

Usage:
    indic-ocr/.venv/bin/python indic-ocr/pipeline_onnx.py --image indic-ocr/model_bundle/assets/diagram.png
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline_onnx")


class OnnxIndicOCR:
    """Integrated OCR pipeline powered by OnnxIndicDocLayout for stage 1 and flexible stage 2 backends."""

    def __init__(
        self,
        layout_onnx_path: str | Path | None = None,
        bundle_dir: str | Path | None = None,
        ocr_backend: str = "onnx",
        device: str = "cpu",
    ) -> None:
        root = Path(__file__).parent
        if bundle_dir is None:
            bundle_dir = root / "model_bundle"
        self.bundle_dir = Path(bundle_dir)

        if not (self.bundle_dir / "idp_offline.py").exists():
            raise FileNotFoundError(
                f"Model bundle not found at '{self.bundle_dir}'. "
                "Please run 'make download-weights' (or 'python download_weights.py') first."
            )

        sys.path.insert(0, str(root))
        sys.path.insert(0, str(self.bundle_dir))

        from layout_onnx import OnnxIndicDocLayout
        from idp_offline import IndicBlockOCR, IndicDocLayout
        from idp_types import LayoutConfig, RecognizerConfig

        logger.info("Initializing Stage 1 ONNX layout backend...")
        onnx_backend = OnnxIndicDocLayout(
            onnx_model_path=layout_onnx_path,
            bundle_dir=self.bundle_dir,
        )
        self.layout = IndicDocLayout(backend=onnx_backend)

        logger.info("Initializing Stage 2 OCR recognizer backend ('%s')...", ocr_backend)
        rec_config = RecognizerConfig()

        if ocr_backend in ("onnx", "onnx-int8"):
            from idp_recognizer_onnx import OnnxRecognizer

            quantized = (ocr_backend == "onnx-int8")
            backend = OnnxRecognizer(
                bundle_dir=self.bundle_dir,
                model_dir=root / "onnx_output" / "ocr",
                quantized=quantized,
            )
        elif ocr_backend.startswith("mlx"):
            from idp_recognizer_mlx import MlxRecognizer

            mlx_root = root / "mlx_output"
            if ocr_backend == "mlx-8bit" and (mlx_root / "ocr_8bit").exists():
                weights_dir = mlx_root / "ocr_8bit"
            elif ocr_backend == "mlx-4bit" and (mlx_root / "ocr_4bit").exists():
                weights_dir = mlx_root / "ocr_4bit"
            elif ocr_backend == "mlx-bf16" and (mlx_root / "ocr_bf16").exists():
                weights_dir = mlx_root / "ocr_bf16"
            elif (mlx_root / "ocr_8bit").exists():
                weights_dir = mlx_root / "ocr_8bit"
            else:
                weights_dir = self.bundle_dir / "weights" / "ocr"

            backend = MlxRecognizer(weights_dir=weights_dir)
        else:
            from idp_recognizer import HfRecognizer

            ocr_weights = self.bundle_dir / "weights" / "ocr"
            backend = HfRecognizer(str(ocr_weights), config=rec_config, device=device)

        self.ocr = IndicBlockOCR(backend=backend, config=rec_config)

    def parse(self, image_path: str | Path) -> Any:
        """Parse document image into reading-ordered blocks and markdown."""
        path_str = str(image_path)
        logger.info("Detecting document layout with ONNX for %s...", Path(path_str).name)
        layout_result = self.layout.detect(path_str)
        logger.info("Detected %d blocks. Transcribing content...", len(layout_result.blocks))
        page_result = self.ocr.run(path_str, layout_result)
        logger.info("Transcription complete.")
        return page_result

    def close(self) -> None:
        self.layout.close()
        self.ocr.close()

    def __enter__(self) -> OnnxIndicOCR:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end OCR with ONNX-accelerated layout detection."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input document image (PNG/JPG).",
    )
    parser.add_argument(
        "--layout-onnx",
        type=Path,
        default=None,
        help="Path to layout ONNX model (default: auto-detected layout_model.onnx).",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).parent / "model_bundle",
        help="Path to downloaded model bundle.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Path to save reading-ordered Markdown output.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to save per-block structured JSON output.",
    )
    parser.add_argument(
        "--ocr-backend",
        type=str,
        choices=["onnx", "onnx-int8", "mlx", "mlx-8bit", "mlx-4bit", "mlx-bf16", "pytorch"],
        default="onnx",
        help="Backend to use for Stage 2 OCR transcription (default: onnx).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for Stage 2 recognizer (cpu/cuda).",
    )
    args = parser.parse_args()

    with OnnxIndicOCR(
        layout_onnx_path=args.layout_onnx,
        bundle_dir=args.bundle_dir.resolve(),
        ocr_backend=args.ocr_backend,
        device=args.device,
    ) as pipeline:
        result = pipeline.parse(args.image.resolve())

        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(result.markdown, encoding="utf-8")
            logger.info("Saved Markdown output to %s", args.output_md)

        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Saved JSON output to %s", args.output_json)

        # Output preview
        logger.info("--- Generated Markdown Preview ---")
        for line in result.markdown.splitlines()[:20]:
            logger.info(line)


if __name__ == "__main__":
    main()
