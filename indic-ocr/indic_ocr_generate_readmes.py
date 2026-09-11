#!/usr/bin/env python3
"""Automated README / Model Card Generator for IndicOCR MLX Repositories.

Reads benchmark results from JSON reports and dynamically generates consistent,
high-fidelity Hugging Face model cards for 4-bit, 8-bit, and BF16 exports.

Usage:
    python generate_readmes.py [--user hari31416]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HF_USER = "hari31416"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
HF_REPOS_DIR = ROOT_DIR / "hf_repos"
FIXTURES_DIR = SCRIPT_DIR / "fixtures"

VARIANTS_CONFIG = {
    "4bit": {
        "tag": "4-bit",
        "precision_label": "4-bit",
        "weight_size": "636 MB",
        "reduction": "62.6%",
        "benchmark_file": "ocr_benchmark_report_mlx_4bit.json",
        "fidelity_label": "High fidelity",
        "readability_label": "High readability",
        "target_audience": "Low-memory Macs (8 GB/16 GB unified RAM), edge deployments",
        "description": "an optimized 4-bit quantized MLX export of **IndicBlockOCR** (Stage 2 of `bodhan-ai/indic-ocr`), tailored for real-time document transcription on Apple Silicon Metal GPUs (M1/M2/M3/M4) with minimal memory footprint.",
    },
    "8bit": {
        "tag": "8-bit",
        "precision_label": "8-bit",
        "weight_size": "1.0 GB",
        "reduction": "41.2%",
        "benchmark_file": "ocr_benchmark_report_mlx_8bit.json",
        "fidelity_label": "Near-lossless accuracy",
        "readability_label": "Near-lossless readability",
        "target_audience": "Recommended balance between speed and full-precision fidelity",
        "description": "an optimized 8-bit quantized MLX export of **IndicBlockOCR** (Stage 2 of `bodhan-ai/indic-ocr`), delivering near-full-precision accuracy with a 20x latency speedup on Apple Silicon Metal GPUs (M1/M2/M3/M4).",
    },
    "bf16": {
        "tag": "bf16",
        "precision_label": "BF16",
        "weight_size": "1.6 GB",
        "reduction": "Reference size",
        "benchmark_file": "ocr_benchmark_report_mlx_bf16.json",
        "fidelity_label": "Reference fidelity",
        "readability_label": "Reference readability",
        "target_audience": "Full-precision baseline reference",
        "description": "the full-precision **BF16** (Bfloat16) reference MLX export of **IndicBlockOCR** (Stage 2 of `bodhan-ai/indic-ocr`), optimized for Apple Silicon Metal GPUs (M1/M2/M3/M4).",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def format_readme(variant_key: str, hf_user: str, ocr_data: dict[str, Any], bbox_data: dict[str, Any]) -> str:
    cfg = VARIANTS_CONFIG[variant_key]
    repo_name = f"indic-ocr-mlx-{variant_key}"
    full_repo_id = f"{hf_user}/{repo_name}"

    # Extract OCR benchmark numbers
    cer = ocr_data.get("mean_cer", 0.0)
    wer = ocr_data.get("mean_wer", 0.0)
    exact_match = ocr_data.get("exact_match_pct", 0.0)
    cer_dist = ocr_data.get("cer_distribution", {})
    cer_05 = cer_dist.get("cer_le_0.05_pct", 0.0)
    cer_10 = cer_dist.get("cer_le_0.10_pct", 0.0)
    cer_20 = cer_dist.get("cer_le_0.20_pct", 0.0)

    perf = ocr_data.get("performance", {})
    cand_ms = perf.get("candidate_mean_ms_per_crop", 0.0)
    speedup = perf.get("speedup", 1.0)
    throughput = perf.get("throughput_chars_per_sec", 0.0)

    # Build precision variants list
    variants_lines = []
    for k in ["4bit", "8bit", "bf16"]:
        v_repo = f"indic-ocr-mlx-{k}"
        v_cfg = VARIANTS_CONFIG[k]
        v_url = f"https://huggingface.co/{hf_user}/{v_repo}"
        suffix = " (This repository)" if k == variant_key else ""
        
        # Pull speedup/CER for sibling
        sibling_file = FIXTURES_DIR / v_cfg["benchmark_file"]
        sibling_data = load_json(sibling_file)
        s_cer = sibling_data.get("mean_cer", 0.0) * 100
        s_ms = sibling_data.get("performance", {}).get("candidate_mean_ms_per_crop", 0.0)
        s_speedup = sibling_data.get("performance", {}).get("speedup", 1.0)
        
        variants_lines.append(
            f"- [`{hf_user}/{v_repo}`]({v_url}){suffix}: {v_cfg['weight_size']} model weights, "
            f"{s_ms:.0f} ms/crop ({s_speedup:.2f}x speedup), {s_cer:.2f}% CER."
        )
    variants_md = "\n".join(variants_lines)

    # Generate complete Markdown text
    return f"""---
base_model: bodhan-ai/indic-ocr
language:
- en
- as
- bn
- brx
- doi
- gu
- hi
- kn
- ks
- kok
- mai
- ml
- mni
- mr
- ne
- or
- pa
- sa
- sat
- sd
- ta
- te
- ur
tags:
- mlx
- mlx-vlm
- ocr
- document-ai
- vision
- {cfg["tag"]}
- quantized
- qwen3.5
- sarvam
pipeline_tag: image-to-text
license: other
license_name: indic-open-model-license-v1.0
license_link: https://github.com/Bodhan-AI/bodhan-model-info/blob/main/licenses/indic-open-model-license/v1/Indic_Open_Model_License.md
extra_gated_prompt: Please provide your details and agree to the [LICENSE](https://github.com/Bodhan-AI/bodhan-model-info/blob/main/licenses/indic-open-model-license/v1/Indic_Open_Model_License.md)
  [[simpler version](https://github.com/Bodhan-AI/bodhan-model-info/blob/main/licenses/indic-open-model-license/v1/Indic_Open_Model_License_Deed.md)]
  to request access.
extra_gated_fields:
  Company / Organization: text
  Country: country
  Intended Use Case:
    type: select
    options:
    - Research
    - Commercial
    - Education
    - label: Other
      value: other
  I agree to the license terms: checkbox
---

# IndicOCR MLX {cfg["precision_label"]} (IndicBlockOCR)

This repository provides {cfg["description"]}

The recognizer architecture is based on Qwen3.5-0.8B with a 151k-token Sarvam-30B vocabulary covering 22 official Indic scripts, English, numbers, and mathematical notation.

## Precision Variants

{variants_md}

## Empirical Parity and Accuracy Benchmarks

Evaluated against the PyTorch reference baseline on 50 real-world Indic document crops from `ai4bharat/indicdlp`.

### Stage 2 Recognition Parity ({cfg["precision_label"]} vs PyTorch Baseline)

| Metric | PyTorch Baseline (CPU) | IndicOCR MLX {cfg["precision_label"]} (Metal GPU) | Parity Delta / Speedup |
| :--- | :--- | :--- | :--- |
| Character Error Rate (CER) | Reference (0.0) | **{cer:.4f}** ({cer * 100:.2f}%) | {cfg["fidelity_label"]} |
| Word Error Rate (WER) | Reference (0.0) | **{wer:.4f}** ({wer * 100:.2f}%) | {cfg["readability_label"]} |
| Exact Match Rate | 100.0% | **{exact_match:.1f}%** | See analysis below |
| Crops with CER <= 5% | 100.0% | **{cer_05:.1f}%** | High fidelity |
| Crops with CER <= 10% | 100.0% | **{cer_10:.1f}%** | Production-ready |
| Crops with CER <= 20% | 100.0% | **{cer_20:.1f}%** | Production-ready |
| Latency per Crop | 9,768.58 ms | **{cand_ms:.2f} ms** | **{speedup:.2f}x Speedup** |
| Throughput | 15.3 chars/sec | **{throughput:.1f} chars/sec** | **{throughput / 15.3:.1f}x Throughput** |
| Memory Footprint | ~1.7 GB | **~{cfg["weight_size"]}** | **{cfg["reduction"]} Reduction** |

### Stage 1 Layout Detection Parity (PP-DocLayoutV3)

The Stage 1 layout detector weights are bundled under `layout/`:

| Metric | Score | Note |
| :--- | :--- | :--- |
| Mean IoU | **0.9983** | Near-perfect bounding box alignment |
| Label Match Rate | **100.0%** | 37 document class categories identical |
| Reading Order Match Rate | **100.0%** | Pairwise reading order identical |
| Detection Latency | **156.61 ms** | **3.85x Speedup** vs PyTorch CPU (603 ms) |

### Understanding Exact Match vs Character Error Rate

While the Exact Match rate is {exact_match:.1f}%, the mean Character Error Rate is only {cer * 100:.2f}% and {cer_05:.1f}% of crops have less than 5% CER. Exact Match is an all-or-nothing metric: on long paragraph crops (e.g. 600+ characters), a single whitespace difference or character substitution drops the entire crop score to 0.

Autoregressive vision-language decoding on Apple Silicon Metal GPU uses different matrix multiplication tiling, reduction order, and floating-point accumulation compared to NVIDIA CUDA kernels, causing tiny logit variations (`~1e-4`) that can flip `argmax` selections at near-tie probability boundaries.

## Quickstart

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Interactive Gradio Web Studio

Launch the visual interface with interactive bounding box hover tooltips, reading-ordered Markdown preview, and JSON inspection:

```bash
python app.py
```

Then open `http://127.0.0.1:7860` in your browser.

### 3. End-to-End Python Pipeline

Run the full two-stage document pipeline (Stage 1 Layout Detection + Stage 2 Text Transcription):

```python
from pipeline import IndicOCRPipeline

# Initialize end-to-end pipeline (auto-loads Stage 1 and Stage 2 on Apple Silicon)
ocr = IndicOCRPipeline()

# Process full document page
result = ocr.process("path/to/document.png")

# Reading-ordered Markdown
print(result.markdown)

# Access individual detected blocks
for block in result.blocks:
    print(f"#{{block.order}} [{{block.label}}] {{block.bbox_xyxy}}: {{block.text}}")
```

### 4. Single Crop CLI Transcription

Quickly transcribe a specific cropped text line or region:

```bash
python transcribe.py --image path/to/crop.png
```

### 5. Direct `mlx_vlm` Usage

Load the recognizer directly using standard MLX VLM:

```python
from mlx_vlm import generate, load
from PIL import Image

model, processor = load("{full_repo_id}")
prompt = processor.apply_chat_template(
    [
        {{
            "role": "user",
            "content": [
                {{"type": "image"}},
                {{"type": "text", "text": "Transcribe the text in this image."}},
            ],
        }}
    ],
    add_generation_prompt=True,
    tokenize=False,
)

image = Image.open("crop.png").convert("RGB")
output = generate(model, processor, prompt=prompt, image=[image], max_tokens=256)
print(output.text.strip())
```

## Repository Structure

```text
├── README.md                      # Model card and parity benchmarks
├── app.py                         # Interactive Gradio web application
├── pipeline.py                    # Importable end-to-end Python pipeline
├── transcribe.py                  # Single-crop CLI transcription script
├── config.json                    # Recognizer model configuration
├── generation_config.json         # Generation hyperparameter defaults
├── model.safetensors              # {cfg["precision_label"]} recognizer weights (~{cfg["weight_size"]})
├── model.safetensors.index.json   # Weight index mapping
├── processor_config.json          # Vision & image processor configuration
├── tokenizer.json                 # Sarvam-30B tokenizer
├── tokenizer_config.json
├── chat_template.jinja            # Multimodal chat template
├── requirements.txt               # Minimal dependencies
├── .gitattributes                 # Git LFS tracking configuration
└── layout/                        # Stage 1 Layout Detector
    ├── config.json
    └── model.safetensors          # PP-DocLayoutV3 MLX weights (~127 MB)
```

## License and Citations

This model is distributed under the **Indic Open Model License v1.0**. See the [Indic Open Model License](https://github.com/Bodhan-AI/bodhan-model-info/blob/main/licenses/indic-open-model-license/v1/Indic_Open_Model_License.md) and the [Plain-Language License Deed](https://github.com/Bodhan-AI/bodhan-model-info/blob/main/licenses/indic-open-model-license/v1/Indic_Open_Model_License_Deed.md) for complete terms.

If using this export, please cite the original Bodhan AI project:

```bibtex
@misc{{bodhan2026indicocr,
  author = {{Bodhan AI}},
  title = {{IndicOCR: High-Performance Document OCR for Indic Languages}},
  year = {{2026}},
  url = {{https://huggingface.co/bodhan-ai/indic-ocr}}
}}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate READMEs for IndicOCR MLX repositories")
    parser.add_argument("--user", type=str, default=HF_USER, help=f"Hugging Face username (default: {HF_USER})")
    args = parser.parse_args()

    bbox_file = FIXTURES_DIR / "bbox_benchmark_report_mlx.json"
    bbox_data = load_json(bbox_file)

    for variant_key, cfg in VARIANTS_CONFIG.items():
        repo_dir = HF_REPOS_DIR / f"indic-ocr-mlx-{variant_key}"
        if not repo_dir.exists():
            continue

        ocr_file = FIXTURES_DIR / cfg["benchmark_file"]
        ocr_data = load_json(ocr_file)

        readme_content = format_readme(variant_key, args.user, ocr_data, bbox_data)
        readme_path = repo_dir / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")
        print(f"Generated {readme_path}")


if __name__ == "__main__":
    main()
