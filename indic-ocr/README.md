# IndicOCR ONNX Export Suite

This directory contains the self-contained ONNX export, optimization, and parity validation suite for `bodhan-ai/indic-ocr`.

The architecture is completely decoupled from the root repository dependencies, running within a dedicated virtual environment with support for `transformers>=5.7.0` and operating under a 24 GB RAM ceiling.

## Architecture

`bodhan-ai/indic-ocr` is a modular, two-stage document intelligence pipeline:

- Stage 1: IndicDocLayout (PP-DocLayoutV3 / RT-DETR)
  - Purpose: Page-level layout analysis, 37 document class categorizations, and pairwise reading order ranking.
  - Weights: ~33M parameters (~127 MB).
  - ONNX Artifact: `onnx_output/layout/layout_model.onnx` (128 MB FP32, 35 MB INT8).
  - Acceleration: ~2x lower latency in ONNX Runtime CPU compared to PyTorch CPU.
- Stage 2: IndicBlockOCR (Qwen3.5-0.8B + Sarvam-30B Tokenizer)
  - Purpose: Crop-level multimodal text transcription generating reading-ordered Markdown and LaTeX.
  - Weights: ~867M parameters (~1.6 GB).
  - ONNX Artifact: `onnx_output/ocr/visual_encoder.onnx` (384 MB visual backbone) + tokenizer assets.

## Directory Structure

- `download_weights.py`: Downloads layout and OCR weights, schemas, and asset images from Hugging Face.
- `export_layout.py`: Exports Stage 1 `IndicDocLayout` to ONNX with graph verification and optional INT8 quantization.
- `layout_onnx.py`: High-performance ONNX Runtime backend implementing the `IndicDocLayoutBackend` protocol.
- `export_recognizer.py`: Exports Stage 2 visual backbone and packages tokenizers and chat templates.
- `validate_parity.py`: Comprehensive parity validation suite comparing PyTorch vs ONNX detection outputs and latency.
- `pipeline_onnx.py`: End-to-end OCR pipeline taking a document image and producing Markdown and structured JSON.
- `gradio_app.py`: Interactive Gradio testing studio for uploading images, visualizing bounding boxes, and inspecting OCR results.
- `requirements.txt`: Dedicated environment dependencies.
- `fixtures/images/`: Benchmark document images extracted from `ai4bharat/indicdlp`.
- `fixtures/parity_report.json`: Automated parity benchmark metrics and mismatch logs.
- `onnx_output/`: Storage for exported ONNX graphs and tokenizer artifacts.
- `model_bundle/`: Local cache of downloaded weights and assets.

## Setup and Installation

### Step 1: Environment Setup

Create and activate the dedicated virtual environment inside `indic-ocr/`:

```bash
uv venv indic-ocr/.venv --python 3.11
source indic-ocr/.venv/bin/activate
uv pip install -r indic-ocr/requirements.txt
```

Alternatively, from the `indic-ocr/` directory:

```bash
cd indic-ocr
make venv
```

### Step 2: Download Model Bundle (Prerequisite)

`model_bundle/` is git-ignored as it contains the original PyTorch model weights (~1.7 GB) and vendored architecture definitions. Downloading the bundle is a mandatory prerequisite before exporting ONNX graphs or running parity validation:

```bash
# Using Makefile
make download-weights

# Or directly with Python
indic-ocr/.venv/bin/python indic-ocr/download_weights.py
```

Requires terms acceptance for `bodhan-ai/indic-ocr` on Hugging Face.

### Quick Commands (Makefile)

From inside `indic-ocr/`:

- `make help`: Display available targets and descriptions.
- `make download-weights`: Fetch model weights and bundle code from Hugging Face.
- `make export`: Export both Stage 1 layout detector and Stage 2 visual recognizer.
- `make export-int8`: Export INT8 quantized variants for both layout and recognizer.
- `make export-layout`: Export Stage 1 layout model to ONNX with INT8 quantization.
- `make export-recognizer`: Export Stage 2 visual backbone (+ INT8) and tokenizer assets.
- `make parity-check`: Run full numerical parity validation across test fixtures.
- `make parity-check-int8`: Run parity check against the quantized INT8 layout model.
- `make demo`: Execute sample inference on a test document image.
- `make gradio`: Launch the interactive Gradio OCR testing web studio.
- `make clean`: Clean temporary outputs and Python caches.

## Exporting Models to ONNX

### Export Stage 1 Layout Detector

Export the `PPDocLayoutV3` RT-DETR architecture to ONNX with constant folding and validation:

```bash
indic-ocr/.venv/bin/python indic-ocr/export_layout.py --quantize-int8
```

Output:
- `onnx_output/layout/layout_model.onnx` (128 MB)
- `onnx_output/layout/layout_model_int8.onnx` (35 MB)

### Export Stage 2 Visual Backbone

Export the Qwen3.5-0.8B visual feature extractor and copy tokenizer configs:

```bash
indic-ocr/.venv/bin/python indic-ocr/export_recognizer.py
```

Output:
- `onnx_output/ocr/visual_encoder.onnx` (384 MB)
- `onnx_output/ocr/tokenizer.json` (32 MB)
- `onnx_output/ocr/tokenizer_config.json`
- `onnx_output/ocr/chat_template.jinja`

## Parity Validation

To verify numerical parity between the original PyTorch model and the ONNX Runtime graph using diverse real-world Indic document pages from `ai4bharat/indicdlp`:

```bash
indic-ocr/.venv/bin/python indic-ocr/validate_parity.py --images-dir indic-ocr/fixtures/images
```

Validation asserts:
- Exact block count match across all evaluation documents.
- Exact class label match across all detected bounding boxes.
- Exact reading order rank alignment.
- Bounding box pixel delta within tolerance (< 1.5px).
- Benchmark latency and speedup ratio.

### Benchmark Results on `ai4bharat/indicdlp` (25 Document Samples)

Tested across 10 document domains from `ai4bharat/indicdlp` (Newspapers, Textbooks, Research Papers, Question Papers, Acts & Rules, Novels, Magazines, Syllabi, Manuals, Brochures):

- Total Evaluated Documents: 25
- Passed: 25 / 25 (**100% Exact Parity**)
- Average Latency: PyTorch CPU = 609.08 ms | ONNX Runtime CPU = 439.62 ms (~1.4x faster on CPU)
- Max Box Coordinate Delta across all 25 documents: <= 0.20 px
- Complex dense document highlights:
  - `indicdlp_19_np_as_...png` (Newspaper, 78 blocks): **PASS**, 0.00 px delta (PyTorch 675.6 ms vs ONNX 475.4 ms)
  - `indicdlp_01_rp_as_...png` (Research Paper, 15 blocks): **PASS**, 0.00 px delta (PyTorch 596.5 ms vs ONNX 451.4 ms)
  - `indicdlp_14_nv_as_...png` (Novel, 15 blocks): **PASS**, 0.00 px delta (PyTorch 578.0 ms vs ONNX 433.3 ms)
  - `indicdlp_21_mg_as_...png` (Magazine, 12 blocks): **PASS**, 0.00 px delta (PyTorch 685.2 ms vs ONNX 445.4 ms)

Full per-image metrics and logs are saved to `indic-ocr/fixtures/parity_report.json`.

## End-to-End Inference

Run full document OCR on an image producing reading-ordered Markdown and per-block JSON:

```bash
indic-ocr/.venv/bin/python indic-ocr/pipeline_onnx.py \
  --image indic-ocr/model_bundle/assets/diagram.png \
  --output-md output.md \
  --output-json output.json
```

## Apple Silicon inference

On Apple Silicon, IndicOCR uses two native backends:

- Stage 1 layout detection runs with PyTorch on MPS.
- Stage 2 text recognition runs with MLX on Metal.

This split is deliberate. PP-DocLayoutV3 does not have an MLX runtime in this project, and converting its weights to MLX SafeTensors alone does not improve inference. The experimental `export_layout_mlx.py` artifact is not loaded by the pipeline or by the published model repositories. Do not use it as a layout backend.

### MLX Environment Setup

Install MLX dependencies into the virtual environment:

```bash
uv pip install --python indic-ocr/.venv/bin/python -r indic-ocr/requirements-mlx.txt
```

### Export Stage 2 Recognizer to MLX (4-bit, 8-bit, and BF16)

Three precision variants are supported for `IndicBlockOCR` (Qwen3.5-0.8B):

- **8-bit Quantized (`--q-bits 8`) - Recommended:** Highest transcription accuracy with ~1.5s page transcription time.
  ```bash
  indic-ocr/.venv/bin/python indic-ocr/export_recognizer_mlx.py --q-bits 8
  ```
  Artifacts: `mlx_output/ocr_8bit/model.safetensors` (1.0 GB)

- **4-bit Quantized (`--q-bits 4`):** Maximum compression and lowest memory footprint (~500 MB RAM).
  ```bash
  indic-ocr/.venv/bin/python indic-ocr/export_recognizer_mlx.py --q-bits 4
  ```
  Artifacts: `mlx_output/ocr_4bit/model.safetensors` (633 MB, 2.7x compression)

- **Full Precision BF16 (`--no-quantize`):** Exact floating-point parity with original PyTorch weights.
  ```bash
  indic-ocr/.venv/bin/python indic-ocr/export_recognizer_mlx.py --no-quantize
  ```
  Artifacts: `mlx_output/ocr_bf16/model.safetensors` (1.7 GB)

- **Export All Precisions at Once:**
  ```bash
  indic-ocr/.venv/bin/python indic-ocr/export_recognizer_mlx.py --all
  ```

### Layout validation

Verify layout detection parity between PyTorch CPU and PyTorch MPS across benchmark document images from `ai4bharat/indicdlp`:

```bash
indic-ocr/.venv/bin/python indic-ocr/validate_parity_mlx.py --images-dir indic-ocr/fixtures/images
```

Benchmark Results:
- Total Evaluated Documents: 25
- Passed: 25 / 25 (**100% Exact Parity**)
- Max Box Coordinate Delta across all 25 documents: <= 0.30 px
- Full report saved to `indic-ocr/fixtures/parity_report_mlx.json`.

### End-to-end Apple Silicon inference

Run the full pipeline with PyTorch MPS layout detection and MLX text recognition:

```bash
indic-ocr/.venv/bin/python indic-ocr/pipeline_mlx.py \
  --image indic-ocr/model_bundle/assets/diagram.png \
  --output-md output_mlx.md \
  --output-json output_mlx.json
```

## Benchmarking and Parity Suite

For evaluating layout detection bounding boxes and OCR text transcription accuracy across PyTorch, ONNX (FP32 and INT8), and MLX, see [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md).

Quick commands:

```bash
# Download real-world benchmark images from ai4bharat/indicdlp
make download-fixtures

# Run Stage 1 layout bounding box parity benchmark
make bench-bbox

# Run Stage 1 layout benchmark on quantized INT8
make bench-bbox-int8

# Run Stage 2 OCR accuracy and parity benchmark
make bench-ocr

# Run full Stage 1 and Stage 2 comparative benchmark suite
make bench-all
```

## Interactive Gradio Testing Server

The Gradio web interface allows users to upload custom images, execute layout detection and text transcription, visualize bounding boxes with reading order and confidence badges, and inspect/export results in rendered Markdown, raw text, and structured JSON.

### Launching the Server

```bash
# Launch with default settings (port 7860)
make gradio

# Or launch directly with custom host and port
indic-ocr/.venv/bin/python indic-ocr/gradio_app.py --host 127.0.0.1 --port 7860
```

### Features

- Image Upload: Drag-and-drop or select from built-in sample gallery assets.
- Dual Execution Modes: Choose between full "Layout + OCR (Full)" or quick "Layout Detection Only".
- Multi-Backend Selection: Apple Silicon MLX (4-bit, 8-bit, BF16), ONNX Runtime, or PyTorch.
- BBox Visualizer: High-resolution bounding boxes with color coding by category (Title, Text, Table, Header, Footer, Figure, Equation, List, etc.).
- Reading Order and Confidence: Toggle display of reading order numbers (`#1`, `#2`, ...), class labels, confidence scores, and fill overlays.
- Multi-Tab Inspection: Rendered Markdown viewer, raw copyable text editor, per-block inspection table, and structured JSON.
- Export Actions: Download buttons for `.md` Markdown and `.json` structured outputs.
