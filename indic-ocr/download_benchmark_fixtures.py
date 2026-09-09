#!/usr/bin/env python3
"""Download real Indic document pages from ai4bharat/indicdlp for benchmarking.

Usage:
    .venv/bin/python download_benchmark_fixtures.py --count 50
    .venv/bin/python download_benchmark_fixtures.py --count 100 --output-dir fixtures/benchmark_images
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
import sys

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("download_benchmark_fixtures")

DATASET_REPO = "ai4bharat/indicdlp"


def download_benchmark_images(
    count: int,
    output_dir: Path,
) -> list[Path]:
    """Download N document images from ai4bharat/indicdlp parquet shards."""
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Discovering parquet shards from '%s'...", DATASET_REPO)
    api = HfApi()
    all_files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    parquet_shards = sorted(
        [f for f in all_files if f.startswith("data/test-") and f.endswith(".parquet")]
    )

    if not parquet_shards:
        raise RuntimeError(f"No test parquet shards found in {DATASET_REPO}")

    downloaded_paths: list[Path] = []
    saved_count = 0

    for shard in parquet_shards:
        if saved_count >= count:
            break

        logger.info("Fetching parquet shard '%s'...", shard)
        local_shard = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=shard,
            repo_type="dataset",
        )

        table = pq.read_table(local_shard, columns=["image"])
        images_col = table["image"]
        num_rows = len(images_col)
        logger.info("Found %d images in shard. Extracting...", num_rows)

        for row_idx in range(num_rows):
            if saved_count >= count:
                break

            img_bytes = images_col[row_idx]["bytes"].as_py()
            if not img_bytes:
                continue

            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                filename = f"indicdlp_bench_{saved_count:04d}.png"
                out_path = output_dir / filename
                img.save(out_path, format="PNG")
                downloaded_paths.append(out_path)
                saved_count += 1
                if saved_count % 10 == 0 or saved_count == count:
                    logger.info("Saved %d/%d benchmark images -> %s", saved_count, count, out_path.name)
            except Exception as err:
                logger.warning("Failed to decode image at index %d: %s", row_idx, err)

    logger.info("Successfully downloaded %d benchmark images to %s", len(downloaded_paths), output_dir)
    return downloaded_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download real Indic document images from ai4bharat/indicdlp for benchmarking."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of benchmark images to download (default: 50).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "fixtures" / "benchmark_images",
        help="Directory to save downloaded benchmark images (default: fixtures/benchmark_images).",
    )
    args = parser.parse_args()

    download_benchmark_images(
        count=args.count,
        output_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
