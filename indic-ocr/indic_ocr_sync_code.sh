#!/usr/bin/env bash
# Synchronize shared pipeline, gradio app, and helper code to all MLX repos

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
HF_REPOS_DIR="$ROOT_DIR/hf_repos"

if [ -d "$SCRIPT_DIR/_indic_ocr_common" ]; then
    COMMON_DIR="$SCRIPT_DIR/_indic_ocr_common"
else
    COMMON_DIR="$HF_REPOS_DIR/_indic_ocr_common"
fi

TARGETS=(
    "$HF_REPOS_DIR/indic-ocr-mlx-4bit"
    "$HF_REPOS_DIR/indic-ocr-mlx-8bit"
    "$HF_REPOS_DIR/indic-ocr-mlx-bf16"
)

echo "=================================================================="
echo "Synchronizing shared code from _indic_ocr_common/ to precision repositories"
echo "=================================================================="

for target in "${TARGETS[@]}"; do
    if [ -d "$target" ]; then
        repo_name="$(basename "$target")"
        echo "Syncing code to $repo_name..."
        cp -f "$COMMON_DIR"/*.py "$target/"
        cp -f "$COMMON_DIR"/requirements.txt "$target/"
        cp -f "$COMMON_DIR"/.gitignore "$target/"
    fi
done

echo ""
echo "Regenerating Model Cards (README.md) from benchmark fixtures..."
python3 "$SCRIPT_DIR/indic_ocr_generate_readmes.py" --user "${1:-hari31416}"

echo ""
echo "Code and documentation synchronization complete across all repositories."
