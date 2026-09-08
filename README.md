# Bodhan AI ONNX Exports

This repository hosts ONNX exports, optimizations, and parity verification suites for the Bodhan AI foundational model collection.

## Models Overview

- `indic-ocr/` (Active):
  - Architecture: Two-stage document parser (`IndicDocLayout` RT-DETR ~33M params + `IndicBlockOCR` Qwen3.5 ~867M params).
  - Status: Exported to ONNX and Apple MLX (4-bit, 8-bit, BF16), parity-validated on 25 real Indic documents from `ai4bharat/indicdlp` (100% pass across both ONNX and MLX).
- `indic-translate/` (Planned):
  - Architecture: Gemma-4-E4B (8B conditional generation transformer).
- `indic-transcribe-core/` (Planned):
  - Architecture: NVIDIA Canary FastConformer ASR (1B params).
- `indic-transcribe-flex/` (Planned):
  - Architecture: NVIDIA Canary FastConformer multi-script ASR (1B params).
- `indic-speak/` (Planned):
  - Architecture: Llama-3.2-3B + SNAC + Vocos neural vocoder TTS (3B params).

## Subdirectories

- `indic-ocr/`: See `indic-ocr/README.md` for layout and text recognition export instructions, benchmarks, and inference pipelines.
