#!/usr/bin/env python3
"""Push IndicOCR MLX repositories to Hugging Face and configure gating via CLI.

Usage:
    python publish_to_hf.py [--user hari31416] [--variant 4bit|8bit|bf16|all] [--gated auto|manual|none]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("publish_hf")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
HF_REPOS_DIR = ROOT_DIR / "hf_repos"


def publish_repo(
    repo_dir: Path,
    repo_id: str,
    gated_mode: str = "auto",
    commit_message: str | None = None,
    dry_run: bool = False,
) -> None:
    api = HfApi()

    msg = commit_message or "docs: update base_model in frontmatter to link model tree"

    logger.info("=" * 65)
    logger.info("Processing %s -> %s", repo_dir.name, repo_id)
    logger.info("=" * 65)

    if dry_run:
        logger.info("[DRY RUN] Would create repo: %s", repo_id)
        logger.info("[DRY RUN] Would set gating to: %s", gated_mode)
        logger.info("[DRY RUN] Would upload folder: %s (commit: '%s')", repo_dir, msg)
        return

    # 1. Create repository on Hugging Face (idempotent)
    logger.info("Creating repository '%s' on Hugging Face Hub...", repo_id)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    logger.info("Repository confirmed.")

    # 2. Configure gating
    if gated_mode in ["auto", "manual"]:
        logger.info("Configuring gated access mode to '%s'...", gated_mode)
        api.update_repo_settings(
            repo_id=repo_id,
            gated=gated_mode,  # type: ignore[arg-type]
            repo_type="model",
        )
        logger.info("Gating enabled ('%s'). Users must accept license terms.", gated_mode)
    elif gated_mode == "none":
        logger.info("Disabling gating (open public access)...")
        api.update_repo_settings(repo_id=repo_id, gated=False, repo_type="model")

    # 3. Upload all files via Hugging Face Hub (native LFS handling & resume)
    logger.info("Uploading folder %s to %s...", repo_dir, repo_id)
    commit_info = api.upload_folder(
        folder_path=str(repo_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=msg,
        ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store", ".git/*"],
    )
    logger.info("Upload successful! Commit: %s", getattr(commit_info, "commit_id", "done"))
    logger.info("Model URL: https://huggingface.co/%s\n", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish IndicOCR MLX exports to Hugging Face Hub with gating."
    )
    parser.add_argument(
        "--user",
        type=str,
        default="hari31416",
        help="Hugging Face username or organization (default: hari31416)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["4bit", "8bit", "bf16", "all"],
        help="Which precision variant to publish (default: all)",
    )
    parser.add_argument(
        "--gated",
        type=str,
        default="auto",
        choices=["auto", "manual", "none"],
        help="Gated access setting: 'auto' (automatic approval upon agreeing to license), 'manual', or 'none' (default: auto)",
    )
    parser.add_argument(
        "-m",
        "--message",
        type=str,
        default=None,
        help="Custom commit message for the upload",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate actions without actually uploading",
    )
    args = parser.parse_args()

    variants = ["4bit", "8bit", "bf16"] if args.variant == "all" else [args.variant]

    for v in variants:
        repo_name = f"indic-ocr-mlx-{v}"
        repo_dir = HF_REPOS_DIR / repo_name
        if not repo_dir.exists():
            logger.warning("Directory %s not found, skipping.", repo_dir)
            continue

        repo_id = f"{args.user}/{repo_name}"
        publish_repo(
            repo_dir,
            repo_id,
            gated_mode=args.gated,
            commit_message=args.message,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
