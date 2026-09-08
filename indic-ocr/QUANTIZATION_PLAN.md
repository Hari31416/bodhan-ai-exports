# INT8 Quantization and Accuracy Evaluation Roadmap

This document outlines the step-by-step plan for implementing INT8 quantization across both stages of `indic-ocr` and replacing strict parity assertions with comprehensive accuracy benchmarking.

## Overview

The `indic-ocr` architecture consists of two stages:
- Stage 1: `IndicDocLayout` (RT-DETR, ~33M params) for layout detection and reading order.
- Stage 2: `IndicBlockOCR` (Qwen3.5-0.8B visual backbone + language decoder, ~867M params) for crop-level text transcription.

Strict numerical parity (< 1.5px delta, identical confidence scores) is fundamentally unsuitable for INT8 models due to quantization rounding noise. Instead, we evaluate INT8 models against full-precision (FP32) baseline predictions using standard accuracy metrics (mIoU, Precision, Recall, F1, CER, WER).

## Phase 1: Export OCR Visual Recognizer to INT8

The initial priority is exporting Stage 2 visual backbone to INT8.

### Tasks

- Add `--quantize-int8` support in `export_recognizer.py` to produce `onnx_output/ocr/visual_encoder_int8.onnx`.
- Configure dynamic quantization settings for the vision transformer backbone.
- Verify exported INT8 graph integrity and opset compatibility using ONNX checker.
- Add `export-recognizer-int8` target in `Makefile` and update the global `export` target.
- Benchmark file size reduction and CPU inference speedup against the 384 MB FP32 visual encoder.
- Investigate INT8 options for the autoregressive text generation component (e.g. bitsandbytes 8-bit or ONNX Runtime GenAI).

## Phase 2: Accuracy Evaluation Suite (INT8 vs FP32 Ground Truth)

Replace strict parity assertions with task-specific accuracy metrics treating the FP32 model predictions as pseudo ground truth.

### Tasks

- Build `benchmark_accuracy.py` to evaluate INT8 models against FP32 baselines.
- Implement Layout Detection Metrics (Stage 1):
  - Greedy IoU box matching at threshold >= 0.50.
  - Precision, Recall, and F1 score per document class and overall.
  - Mean Intersection-over-Union (mIoU) across matched bounding boxes.
  - Reading order sequence correlation (Spearman rank correlation or Kendall's tau).
- Implement Text Transcription Metrics (Stage 2):
  - Character Error Rate (CER) comparing INT8 OCR output against FP32 output.
  - Word Error Rate (WER).
  - Exact match percentage across text blocks.
- Generate structured report artifacts:
  - JSON output: `fixtures/accuracy_report_int8.json`.
  - Summary table in Markdown.


## Phase 3: Refine Stage 1 Layout INT8 Quantization

The current naive dynamic quantization of RT-DETR causes detection logits to collapse below the 0.5 confidence threshold on several documents.

### Tasks

- Analyze sensitive layers in `layout_model.onnx` that degrade under INT8 quantization (e.g. bounding box regression heads, query cross-attention, LayerNorm).
- Implement selective node exclusion in `export_layout.py` using `nodes_to_exclude` in `onnxruntime.quantization.quantize_dynamic`.
- Re-export `onnx_output/layout/layout_model_int8.onnx` with protected detection heads.
- Verify that block detections no longer collapse to zero on complex document samples.


## Phase 4: Integration and Automation

Integrate all quantization workflows into the developer toolchain.

### Tasks

- Add new Makefile targets:
  - `make export-int8`: Export INT8 graphs for both layout and OCR.
  - `make benchmark-int8`: Run the complete INT8 accuracy benchmark across 25 evaluation fixtures.
  - `make benchmark-layout-int8`: Run layout-specific accuracy checks.
  - `make benchmark-ocr-int8`: Run OCR-specific accuracy checks.
- Update `indic-ocr/README.md` with:
  - Accuracy metrics table (mIoU, Precision, Recall, CER, latency, compression ratio).
  - Recommended deployment guidelines for FP32 vs INT8.
