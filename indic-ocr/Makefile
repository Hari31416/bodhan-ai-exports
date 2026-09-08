.DEFAULT_GOAL := help

PYTHON ?= .venv/bin/python
UV ?= uv
BUNDLE_DIR ?= model_bundle
IMAGES_DIR ?= fixtures/images
LAYOUT_MODEL ?= onnx_output/layout/layout_model.onnx
REPORT_FILE ?= fixtures/parity_report.json

.PHONY: help
help:
	@echo "IndicOCR ONNX Export and Parity Suite"
	@echo ""
	@echo "Usage:"
	@echo "  make <target> [VARIABLE=value]"
	@echo ""
	@echo "Targets:"
	@echo "  venv              Create .venv and install dependencies using uv"
	@echo "  download-weights  Download model weights and assets from Hugging Face"
	@echo "  export-layout     Export Stage 1 layout model (RT-DETR) to ONNX (+ INT8)"
	@echo "  export-recognizer Export Stage 2 visual backbone (+ INT8) and tokenizer"
	@echo "  export            Export both layout and recognizer models"
	@echo "  export-int8       Export INT8 versions for both layout and recognizer"
	@echo "  parity-check      Run numerical parity check comparing PyTorch vs ONNX"
	@echo "  parity-check-int8 Run parity check against the quantized INT8 layout model"
	@echo "  export-layout-mlx Export Stage 1 layout model to Apple MLX format"
	@echo "  export-recognizer-mlx-4bit Export Stage 2 recognizer to MLX 4-bit (633 MB)"
	@echo "  export-recognizer-mlx-8bit Export Stage 2 recognizer to MLX 8-bit (1.0 GB)"
	@echo "  export-recognizer-mlx-bf16 Export Stage 2 recognizer to MLX BF16 (1.7 GB)"
	@echo "  export-recognizer-mlx-all  Export all three MLX precision variants"
	@echo "  export-mlx        Export Stage 1 layout and all Stage 2 MLX models"
	@echo "  parity-check-mlx  Run parity check comparing PyTorch vs MLX layout"
	@echo "  demo-mlx          Run native MLX end-to-end OCR pipeline on Apple Silicon"
	@echo "  demo-mlx-8bit     Run demo using 8-bit MLX recognizer"
	@echo "  demo-mlx-4bit     Run demo using 4-bit MLX recognizer"
	@echo "  demo-mlx-bf16     Run demo using full-precision BF16 MLX recognizer"
	@echo "  demo              Run sample end-to-end inference pipeline on test image"
	@echo "  clean             Remove cache and temporary output files"
	@echo "  clean-onnx        Remove exported ONNX artifacts in onnx_output/"
	@echo "  clean-mlx         Remove exported MLX artifacts in mlx_output/"
	@echo "  clean-all         Remove onnx_output/, mlx_output/, caches, and temporary files"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON            Python executable path (default: .venv/bin/python)"
	@echo "  IMAGES_DIR        Parity benchmark images dir (default: fixtures/images)"
	@echo "  LAYOUT_MODEL      Layout model path to test (default: onnx_output/layout/layout_model.onnx)"

.PHONY: venv
venv:
	$(UV) venv .venv --python 3.11
	$(UV) pip install -r requirements.txt

.PHONY: download-weights
download-weights:
	$(PYTHON) download_weights.py

.PHONY: export-layout
export-layout:
	$(PYTHON) export_layout.py --bundle-dir $(BUNDLE_DIR) --quantize-int8

.PHONY: export-recognizer
export-recognizer:
	$(PYTHON) export_recognizer.py --bundle-dir $(BUNDLE_DIR) --quantize-int8

.PHONY: export-recognizer-int8
export-recognizer-int8: export-recognizer

.PHONY: export
export: export-layout export-recognizer

.PHONY: export-int8
export-int8: export-layout export-recognizer

.PHONY: parity-check
parity-check:
	$(PYTHON) validate_parity.py \
		--bundle-dir $(BUNDLE_DIR) \
		--model-path $(LAYOUT_MODEL) \
		--images-dir $(IMAGES_DIR) \
		--output-report $(REPORT_FILE)

.PHONY: parity-check-int8
parity-check-int8:
	$(PYTHON) validate_parity.py \
		--bundle-dir $(BUNDLE_DIR) \
		--model-path onnx_output/layout/layout_model_int8.onnx \
		--images-dir $(IMAGES_DIR) \
		--output-report fixtures/parity_report_int8.json \
		--warn-only

.PHONY: parity
parity: parity-check

.PHONY: check
check: parity-check

.PHONY: export-layout-mlx
export-layout-mlx:
	$(PYTHON) export_layout_mlx.py --bundle-dir $(BUNDLE_DIR)

.PHONY: export-recognizer-mlx-4bit
export-recognizer-mlx-4bit:
	$(PYTHON) export_recognizer_mlx.py --bundle-dir $(BUNDLE_DIR) --q-bits 4

.PHONY: export-recognizer-mlx-8bit
export-recognizer-mlx-8bit:
	$(PYTHON) export_recognizer_mlx.py --bundle-dir $(BUNDLE_DIR) --q-bits 8

.PHONY: export-recognizer-mlx-bf16
export-recognizer-mlx-bf16:
	$(PYTHON) export_recognizer_mlx.py --bundle-dir $(BUNDLE_DIR) --no-quantize

.PHONY: export-recognizer-mlx-all
export-recognizer-mlx-all:
	$(PYTHON) export_recognizer_mlx.py --bundle-dir $(BUNDLE_DIR) --all

.PHONY: export-recognizer-mlx
export-recognizer-mlx: export-recognizer-mlx-8bit

.PHONY: export-mlx
export-mlx: export-layout-mlx export-recognizer-mlx-all

.PHONY: parity-check-mlx
parity-check-mlx:
	$(PYTHON) validate_parity_mlx.py \
		--bundle-dir $(BUNDLE_DIR) \
		--images-dir $(IMAGES_DIR) \
		--output-report fixtures/parity_report_mlx.json

.PHONY: demo-mlx-4bit
demo-mlx-4bit:
	$(PYTHON) pipeline_mlx.py \
		--image model_bundle/assets/diagram.png \
		--ocr-model mlx_output/ocr_4bit \
		--output-md output_mlx_4bit.md \
		--output-json output_mlx_4bit.json

.PHONY: demo-mlx-8bit
demo-mlx-8bit:
	$(PYTHON) pipeline_mlx.py \
		--image model_bundle/assets/diagram.png \
		--ocr-model mlx_output/ocr_8bit \
		--output-md output_mlx_8bit.md \
		--output-json output_mlx_8bit.json

.PHONY: demo-mlx-bf16
demo-mlx-bf16:
	$(PYTHON) pipeline_mlx.py \
		--image model_bundle/assets/diagram.png \
		--ocr-model mlx_output/ocr_bf16 \
		--output-md output_mlx_bf16.md \
		--output-json output_mlx_bf16.json

.PHONY: demo-mlx
demo-mlx: demo-mlx-8bit

.PHONY: demo
demo:
	$(PYTHON) pipeline_onnx.py \
		--image model_bundle/assets/diagram.png \
		--output-md output_demo.md \
		--output-json output_demo.json

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f output_*.md output_*.json output.md output.json

.PHONY: clean-onnx
clean-onnx:
	rm -rf onnx_output/layout/* onnx_output/ocr/*

.PHONY: clean-mlx
clean-mlx:
	rm -rf mlx_output/layout/* mlx_output/ocr_4bit/* mlx_output/ocr_8bit/* mlx_output/ocr_bf16/*

.PHONY: clean-all
clean-all: clean clean-onnx clean-mlx
