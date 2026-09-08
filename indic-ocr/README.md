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
- `requirements.txt`: Dedicated environment dependencies.
- `fixtures/images/`: Benchmark document images extracted from `ai4bharat/indicdlp`.
- `fixtures/parity_report.json`: Automated parity benchmark metrics and mismatch logs.
- `onnx_output/`: Storage for exported ONNX graphs and tokenizer artifacts.
- `model_bundle/`: Local cache of downloaded weights and assets.

## Setup and Installation

Create and activate the dedicated virtual environment inside `indic-ocr/`:

```bash
uv venv indic-ocr/.venv --python 3.11
source indic-ocr/.venv/bin/activate
uv pip install -r indic-ocr/requirements.txt
```

Alternatively, from the `indic-ocr/` directory, use the provided `Makefile`:

```bash
cd indic-ocr
make venv
```

### Quick Commands (Makefile)

From inside `indic-ocr/`:

- `make help`: Display available targets and descriptions.
- `make download-weights`: Fetch model weights from Hugging Face.
- `make export`: Export both Stage 1 layout detector and Stage 2 visual recognizer.
- `make export-layout`: Export Stage 1 layout model to ONNX with INT8 quantization.
- `make export-recognizer`: Export Stage 2 visual backbone and tokenizer assets.
- `make parity-check`: Run full parity validation across test fixtures.
- `make demo`: Execute sample inference on a test document image.
- `make clean`: Clean temporary outputs and Python caches.


## Downloading Model Weights

Download the pre-trained weights from Hugging Face (requires terms acceptance on `bodhan-ai/indic-ocr`):

```bash
indic-ocr/.venv/bin/python indic-ocr/download_weights.py
```

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
