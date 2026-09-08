#!/usr/bin/env python3
"""Export Stage 1 (IndicDocLayout / RT-DETR) from bodhan-ai/indic-ocr to ONNX.

Produces:
    - layout_model.onnx (+ layout_model.onnx.data)
    - optional layout_model_int8.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_layout")


class IndicDocLayoutOnnxWrapper(torch.nn.Module):
    """Wrapper that extracts logits, bounding boxes, and order affinity matrix."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model.eval()

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits, outputs.pred_boxes, outputs.order_logits


def export_layout_model(
    bundle_dir: Path,
    output_dir: Path,
    opset: int = 18,
    quantize_int8: bool = False,
) -> Path:
    """Export the layout detection model to ONNX."""
    sys.path.insert(0, str(bundle_dir))
    from idp_model_ppdoc import PPDocLayoutV3Trainable

    weights_path = bundle_dir / "weights" / "layout"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Layout weights directory not found at: {weights_path}. Run download_weights.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "layout_model.onnx"

    logger.info("Loading PyTorch model from %s", weights_path)
    model = PPDocLayoutV3Trainable.from_pretrained(str(weights_path)).eval()
    wrapper = IndicDocLayoutOnnxWrapper(model).eval()

    dummy_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    logger.info("Exporting ONNX graph to %s (opset=%d)...", onnx_path, opset)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes", "order_logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
            "order_logits": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    logger.info("Verifying exported ONNX model with onnx.checker...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure is valid.")

    # Parity verification on sample image
    sample_img_path = bundle_dir / "assets" / "diagram.png"
    if sample_img_path.exists():
        logger.info("Running parity validation on sample image: %s", sample_img_path.name)
        _validate_parity(model, onnx_path, sample_img_path)

    if quantize_int8:
        _quantize_int8(onnx_path, output_dir / "layout_model_int8.onnx")

    return onnx_path


def _validate_parity(
    pt_model: torch.nn.Module,
    onnx_path: Path,
    image_path: Path,
) -> None:
    """Validate detection outputs between PyTorch and ONNX Runtime."""
    from idp_model_infer import cxcywh_to_xyxy
    from idp_model_order_loss import decode_order

    img = Image.open(image_path).convert("RGB").resize((1024, 1024))
    img_array = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # 1. PyTorch inference
    with torch.no_grad():
        pt_out = pt_model(pixel_values=torch.from_numpy(input_tensor))
        pt_scores, pt_labels = pt_out.logits.sigmoid().max(-1)
        pt_boxes = cxcywh_to_xyxy(pt_out.pred_boxes)[0]
        pt_order = pt_out.order_logits[0]

    pt_keep = (pt_scores[0] > 0.5).nonzero().squeeze(-1)
    logger.info("PyTorch detected %d blocks with conf > 0.5", pt_keep.numel())

    # 2. ONNX Runtime inference
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    on_logits, on_boxes, on_order = session.run(None, {"pixel_values": input_tensor})

    on_logits_t = torch.from_numpy(on_logits)
    on_scores, on_labels = on_logits_t.sigmoid().max(-1)
    on_boxes_t = cxcywh_to_xyxy(torch.from_numpy(on_boxes))[0]
    on_order_t = torch.from_numpy(on_order)[0]

    on_keep = (on_scores[0] > 0.5).nonzero().squeeze(-1)
    logger.info("ONNX Runtime detected %d blocks with conf > 0.5", on_keep.numel())

    if pt_keep.numel() != on_keep.numel():
        raise RuntimeError(
            f"Parity failure: PyTorch detected {pt_keep.numel()} blocks, ONNX detected {on_keep.numel()}"
        )

    # Check top box coordinate agreement
    box_diff = (pt_boxes[pt_keep] - on_boxes_t[on_keep]).abs().max().item()
    logger.info("Maximum coordinate delta on kept blocks: %.6f", box_diff)
    if box_diff > 1e-3:
        logger.warning("Coordinate delta is slightly high: %.6f", box_diff)
    else:
        logger.info("Parity verification passed successfully.")


def _quantize_int8(input_path: Path, output_path: Path) -> None:
    """Dynamic INT8 quantization of the layout detection model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    logger.info("Quantizing model to dynamic INT8 -> %s", output_path)
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )
    logger.info("INT8 quantization complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export IndicDocLayout to ONNX.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).parent / "model_bundle",
        help="Path to model bundle containing downloaded weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "onnx_output" / "layout",
        help="Destination directory for exported ONNX files.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18).",
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Generate an additional INT8 dynamically quantized model.",
    )
    args = parser.parse_args()

    export_layout_model(
        bundle_dir=args.bundle_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        opset=args.opset,
        quantize_int8=args.quantize_int8,
    )


if __name__ == "__main__":
    main()
