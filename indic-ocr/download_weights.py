#!/usr/bin/env python3
"""Download bodhan-ai/indic-ocr model weights and assets from Hugging Face.

Usage:
    uv run python download_weights.py [--target-dir ./weights] [--include-assets]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_weights")

REPO_ID = "bodhan-ai/indic-ocr"


def download_ocr_bundle(target_dir: Path, include_assets: bool = True) -> Path:
    """Download the layout weights, OCR weights, schemas, and asset fixtures."""
    logger.info("Starting download of %s into %s", REPO_ID, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = [
        "config.json",
        "*.py",
        "weights/layout/*",
        "weights/ocr/*",
        "schemas/*",
    ]
    if include_assets:
        allow_patterns.append("assets/*")

    downloaded_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(target_dir),
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=False,
    )
    logger.info("Successfully downloaded bundle to: %s", downloaded_path)
    return Path(downloaded_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download bodhan-ai/indic-ocr weights.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(__file__).parent / "model_bundle",
        help="Directory to store downloaded weights (default: ./model_bundle)",
    )
    parser.add_argument(
        "--no-assets",
        action="store_true",
        help="Skip downloading sample test images",
    )
    args = parser.parse_args()

    download_ocr_bundle(args.target_dir, include_assets=not args.no_assets)


if __name__ == "__main__":
    main()
