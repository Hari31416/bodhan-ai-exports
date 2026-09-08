#!/usr/bin/env python3
"""Unified Apple Silicon MLX OCR pipeline for IndicOCR.

Executes both Stage 1 layout detection (IndicDocLayout) and Stage 2 text recognition
(IndicBlockOCR / Qwen3.5-0.8B) natively on Apple Silicon Metal GPU without ONNX Runtime.

Usage:
    indic-ocr/.venv/bin/python indic-ocr/pipeline_mlx.py \
        --image indic-ocr/model_bundle/assets/diagram.png \
        --output-md output.md \
        --output-json output.json
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
logger = logging.getLogger("pipeline_mlx")


class MlxIndicOCR:
    """End-to-end OCR pipeline powered 100% natively by Apple MLX."""

    def __init__(
        self,
        layout_weights_path: str | Path | None = None,
        ocr_model_path: str | Path | None = None,
        bundle_dir: str | Path | None = None,
    ) -> None:
        root = Path(__file__).parent
        if bundle_dir is None:
            bundle_dir = root / "model_bundle"
        self.bundle_dir = Path(bundle_dir)

        if not (self.bundle_dir / "idp_offline.py").exists():
            raise FileNotFoundError(
                f"Model bundle not found at '{self.bundle_dir}'. "
                "Please run 'make download-weights' first."
            )

        sys.path.insert(0, str(root))
        sys.path.insert(0, str(self.bundle_dir))

        from idp_offline import IndicBlockOCR, IndicDocLayout
        from idp_types import LayoutConfig, RecognizerConfig
        from idp_recognizer_mlx import MlxRecognizer
        from layout_mlx import MlxIndicDocLayout

        logger.info("Initializing Stage 1 MLX layout detector...")
        layout_backend = MlxIndicDocLayout(
            weights_path=layout_weights_path,
            bundle_dir=self.bundle_dir,
        )
        self.layout = IndicDocLayout(backend=layout_backend)

        logger.info("Initializing Stage 2 MLX recognizer...")
        if ocr_model_path is None:
            candidates = [
                root / "mlx_output" / "ocr_4bit",
                root / "mlx_output" / "ocr_8bit",
                root / "mlx_output" / "ocr_bf16",
                root / "mlx_output" / "ocr",
            ]
            for cand in candidates:
                if cand.exists():
                    ocr_model_path = cand
                    break
            if ocr_model_path is None:
                raise FileNotFoundError(
                    "No exported MLX recognizer model found in mlx_output/. "
                    "Run export_recognizer_mlx.py first."
                )

        rec_backend = MlxRecognizer(str(ocr_model_path))
        rec_config = RecognizerConfig()
        self.ocr = IndicBlockOCR(backend=rec_backend, config=rec_config)

    def parse(self, image_path: str | Path) -> Any:
        """Parse document image into reading-ordered blocks and markdown."""
        path_str = str(image_path)
        logger.info("Detecting document layout with MLX for %s...", Path(path_str).name)
        layout_result = self.layout.detect(path_str)
        logger.info(
            "Detected %d blocks. Transcribing content with MLX...",
            len(layout_result.blocks),
        )
        page_result = self.ocr.run(path_str, layout_result)
        logger.info("Transcription complete.")
        return page_result

    def close(self) -> None:
        self.layout.close()
        self.ocr.close()

    def __enter__(self) -> MlxIndicOCR:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run end-to-end OCR with native Apple Silicon MLX pipeline."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input document image (PNG/JPG).",
    )
    parser.add_argument(
        "--layout-weights",
        type=Path,
        default=None,
        help="Path to layout MLX model weights (default: mlx_output/layout/model.safetensors).",
    )
    parser.add_argument(
        "--ocr-model",
        type=Path,
        default=None,
        help="Path to converted OCR MLX model directory (default: mlx_output/ocr_4bit).",
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
    args = parser.parse_args()

    with MlxIndicOCR(
        layout_weights_path=args.layout_weights,
        ocr_model_path=args.ocr_model,
        bundle_dir=args.bundle_dir.resolve(),
    ) as pipeline:
        result = pipeline.parse(args.image.resolve())

        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(result.markdown, encoding="utf-8")
            logger.info("Saved Markdown output to %s", args.output_md)

        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(result.as_record(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Saved JSON output to %s", args.output_json)

        # Output preview
        logger.info("--- Generated Markdown Preview ---")
        for line in result.markdown.splitlines()[:20]:
            logger.info(line)


if __name__ == "__main__":
    main()
