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
	@echo "  demo              Run sample end-to-end inference pipeline on test image"
	@echo "  clean             Remove cache and temporary output files"
	@echo "  clean-onnx        Remove exported ONNX artifacts in onnx_output/"
	@echo "  clean-all         Remove onnx_output/, caches, and temporary files"
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

.PHONY: clean-all
clean-all: clean clean-onnx
