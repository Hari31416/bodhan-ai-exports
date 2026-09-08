#!/usr/bin/env python3
"""Export Stage 2 (IndicBlockOCR / Qwen3.5-0.8B) components to ONNX.

Produces:
    - visual_encoder.onnx (Vision patch feature extractor)
    - Tokenizer and generation configs for browser / onnxruntime inference
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys

import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("export_recognizer")


class QwenVisualWrapper(torch.nn.Module):
    """Wrapper around Qwen3.5 visual backbone for ONNX export."""

    def __init__(self, visual: torch.nn.Module) -> None:
        super().__init__()
        self.visual = visual.eval()

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        return self.visual(pixel_values, grid_thw=grid_thw)


class QwenTextDecoderWrapper(torch.nn.Module):
    """Wrapper around Qwen3.5 language model and lm_head for ONNX export."""

    def __init__(
        self, language_model: torch.nn.Module, lm_head: torch.nn.Module
    ) -> None:
        super().__init__()
        self.language_model = language_model.eval()
        self.lm_head = lm_head.eval()

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.language_model(
            input_ids=input_ids, position_ids=position_ids, use_cache=False
        ).last_hidden_state
        return self.lm_head(hidden)


def export_recognizer_visual(
    bundle_dir: Path,
    output_dir: Path,
    opset: int = 18,
    quantize_int8: bool = False,
) -> Path:
    """Export the vision encoder component of Qwen3.5-0.8B to ONNX."""
    weights_path = bundle_dir / "weights" / "ocr"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"OCR weights directory not found at {weights_path}. Run download_weights.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_onnx_path = output_dir / "visual_encoder.onnx"

    logger.info("Loading Qwen3.5 model from %s", weights_path)
    model = AutoModelForImageTextToText.from_pretrained(
        str(weights_path),
        dtype=torch.float32,
        device_map="cpu",
    ).eval()

    logger.info("Copying tokenizer and processor artifacts to %s", output_dir)
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
    ]:
        src_file = weights_path / filename
        if src_file.exists():
            shutil.copy2(src_file, output_dir / filename)

    wrapper = QwenVisualWrapper(model.model.visual).eval()

    # Dummy input: 256 patches x 1536 patch dimension
    dummy_pixels = torch.randn(256, 1536, dtype=torch.float32)
    dummy_grid = torch.tensor([[1, 16, 16]], dtype=torch.int64)

    logger.info("Exporting visual encoder to %s...", visual_onnx_path)
    torch.onnx.export(
        wrapper,
        (dummy_pixels, dummy_grid),
        str(visual_onnx_path),
        input_names=["pixel_values", "grid_thw"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "num_patches"},
            "grid_thw": {0: "batch_size"},
            "image_features": {0: "num_patches"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    logger.info("Checking visual encoder ONNX model...")
    onnx_model = onnx.load(str(visual_onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("Visual encoder exported and verified successfully.")

    if quantize_int8:
        int8_onnx_path = output_dir / "visual_encoder_int8.onnx"
        _quantize_int8(visual_onnx_path, int8_onnx_path)

    return visual_onnx_path


def _get_node_deps(node: onnx.NodeProto) -> set[str]:
    """Extract all input tensor dependencies for a node, including subgraph captures."""
    deps = set(inp for inp in node.input if inp != "")
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            subgraph = attr.g
            sub_produced = set(inp.name for inp in subgraph.input)
            for sn in subgraph.node:
                for inp in sn.input:
                    if inp and inp not in sub_produced:
                        deps.add(inp)
                for out in sn.output:
                    if out:
                        sub_produced.add(out)
    return deps


def _topological_sort(graph: onnx.GraphProto) -> bool:
    """Sort ONNX graph nodes in topological dependency order."""
    available = set(inp.name for inp in graph.input)
    available.update(init.name for init in graph.initializer)

    nodes = list(graph.node)
    sorted_nodes: list[onnx.NodeProto] = []

    while nodes:
        progress = False
        remaining: list[onnx.NodeProto] = []
        for node in nodes:
            deps = _get_node_deps(node)
            if all(dep in available for dep in deps):
                sorted_nodes.append(node)
                available.update(out for out in node.output if out != "")
                progress = True
            else:
                remaining.append(node)
        if not progress:
            return False
        nodes = remaining

    del graph.node[:]
    graph.node.extend(sorted_nodes)
    return True


def _quantize_int8(input_path: Path, output_path: Path) -> None:
    """Dynamic INT8 quantization for the visual encoder backbone."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    logger.info("Quantizing visual encoder to dynamic INT8 -> %s", output_path)
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    logger.info("Topologically sorting quantized graph nodes...")
    model = onnx.load(str(output_path))
    if not _topological_sort(model.graph):
        logger.warning("Topological sort could not resolve all dependencies.")
    onnx.save(model, str(output_path))

    logger.info("Verifying quantized model with ONNX checker...")
    onnx.checker.check_model(model)

    orig_mb = input_path.stat().st_size / (1024 * 1024)
    q_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "INT8 quantization complete: %.1f MB -> %.1f MB (%.1fx compression)",
        orig_mb,
        q_mb,
        orig_mb / q_mb,
    )


def export_recognizer_text_decoder(
    bundle_dir: Path,
    output_dir: Path,
    opset: int = 18,
    quantize_int8: bool = False,
) -> Path:
    """Export the 24-layer Qwen3.5 autoregressive text decoder to ONNX."""
    weights_path = bundle_dir / "weights" / "ocr"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"OCR weights directory not found at {weights_path}. Run download_weights.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    decoder_onnx_path = output_dir / "text_decoder.onnx"

    logger.info("Loading Qwen3.5 text model with eager attention from %s", weights_path)
    model = AutoModelForImageTextToText.from_pretrained(
        str(weights_path),
        dtype=torch.float32,
        attn_implementation="eager",
        device_map="cpu",
    ).eval()

    wrapper = QwenTextDecoderWrapper(model.model.language_model, model.lm_head).eval()
    dummy_ids = torch.tensor([[100, 200]], dtype=torch.long)
    dummy_pos = torch.tensor([[[0, 1]]], dtype=torch.long)

    logger.info("Exporting text decoder to %s...", decoder_onnx_path)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_pos),
        str(decoder_onnx_path),
        input_names=["input_ids", "position_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "position_ids": {1: "batch", 2: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("Text decoder ONNX graph exported successfully.")

    # Consolidate loose external data files produced by PyTorch into a single .data file
    logger.info("Consolidating external initializers into single data file...")
    model_proto = onnx.load(str(decoder_onnx_path), load_external_data=True)
    data_filename = f"{decoder_onnx_path.name}.data"
    onnx.save_model(
        model_proto,
        str(decoder_onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
    )
    # Remove any loose chunk files left behind by torch.onnx.export
    for loose_file in output_dir.glob("onnx__*"):
        loose_file.unlink(missing_ok=True)
    for loose_file in output_dir.glob("_m_*"):
        loose_file.unlink(missing_ok=True)
    for loose_file in output_dir.glob("*.weight"):
        loose_file.unlink(missing_ok=True)
    logger.info("Consolidation complete -> %s and %s", decoder_onnx_path.name, data_filename)

    if quantize_int8:
        int8_onnx_path = output_dir / "text_decoder_int8.onnx"
        logger.info("Quantizing text decoder to dynamic INT8 -> %s...", int8_onnx_path)
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=decoder_onnx_path,
            model_output=int8_onnx_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
        )
        logger.info("Text decoder INT8 quantization complete.")

    return decoder_onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export IndicBlockOCR components to ONNX."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).parent / "model_bundle",
        help="Path to model bundle containing downloaded weights.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "onnx_output" / "ocr",
        help="Destination directory for exported ONNX and tokenizer files.",
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
    parser.add_argument(
        "--export-decoder",
        action="store_true",
        help="Export the 24-layer Qwen3.5 text decoder language model.",
    )
    args = parser.parse_args()

    export_recognizer_visual(
        bundle_dir=args.bundle_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        opset=args.opset,
        quantize_int8=args.quantize_int8,
    )

    if args.export_decoder:
        export_recognizer_text_decoder(
            bundle_dir=args.bundle_dir.resolve(),
            output_dir=args.output_dir.resolve(),
            opset=args.opset,
            quantize_int8=args.quantize_int8,
        )


if __name__ == "__main__":
    main()
