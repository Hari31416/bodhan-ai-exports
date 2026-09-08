#!/usr/bin/env python3
"""ONNX Runtime implementation of Stage 1 (IndicDocLayout) for bodhan-ai/indic-ocr.

Provides OnnxIndicDocLayout as a high-performance, torch-free (or minimal torch)
layout detector that outputs reading-ordered document blocks.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import onnxruntime as ort
from PIL import Image

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layout_onnx")


class OnnxIndicDocLayout:
    """Drop-in ONNX Runtime replacement for IndicDocLayoutBackend."""

    def __init__(
        self,
        onnx_model_path: str | Path | None = None,
        bundle_dir: str | Path | None = None,
        conf_threshold: float = 0.5,
        img_size: int = 1024,
        providers: list[str] | None = None,
    ) -> None:
        self.img_size = img_size
        self.conf_threshold = conf_threshold

        root = Path(__file__).parent
        if bundle_dir is None:
            bundle_dir = root / "model_bundle"
        self.bundle_dir = Path(bundle_dir)

        if onnx_model_path is None:
            # Default to validated FP32 model
            fp32_path = root / "onnx_output" / "layout" / "layout_model.onnx"
            int8_path = root / "onnx_output" / "layout" / "layout_model_int8.onnx"
            if fp32_path.exists():
                onnx_model_path = fp32_path
            elif int8_path.exists():
                onnx_model_path = int8_path
            else:
                raise FileNotFoundError(
                    f"No exported ONNX model found at {fp32_path}. Run export_layout.py first."
                )

        self.onnx_model_path = Path(onnx_model_path)
        logger.info("Initializing ONNX layout session with: %s", self.onnx_model_path.name)

        if providers is None:
            # CPUExecutionProvider is the most numerically stable across architectures
            providers = ["CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.onnx_model_path),
            sess_options=sess_opts,
            providers=providers,
        )

        # Import helper utilities from model bundle
        sys.path.insert(0, str(self.bundle_dir))
        from idp_blocks import clamp_to_page, clean_layout
        from idp_contract import map_label
        from idp_model_labels import ID2LABEL
        from idp_model_order_loss import decode_order
        from idp_types import Block, DedupConfig

        self.id2label = ID2LABEL
        self.decode_order_fn = decode_order
        self.clamp_to_page_fn = clamp_to_page
        self.clean_layout_fn = clean_layout
        self.map_label_fn = map_label
        self.block_cls = Block
        self.dedup_cfg = DedupConfig()

    def _cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, w, h] to [x0, y0, x1, y1]."""
        out = np.empty_like(boxes)
        out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0
        out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0
        out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0
        out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0
        return out

    def detect_raw(self, image: PILImage) -> list[dict[str, Any]]:
        """Run ONNX detection and return raw viewer schema detections."""
        im = image.convert("RGB").resize((self.img_size, self.img_size))
        x_np = np.array(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        x_np = np.expand_dims(x_np, axis=0)

        logits, pred_boxes, order_logits = self.session.run(
            None, {"pixel_values": x_np}
        )

        # Sigmoid max
        probs = 1.0 / (1.0 + np.exp(-logits[0]))
        scores = np.max(probs, axis=-1)
        labels = np.argmax(probs, axis=-1)

        boxes_xyxy = self._cxcywh_to_xyxy(pred_boxes[0])
        order_scores = order_logits[0]

        keep = np.where(scores > self.conf_threshold)[0]
        if len(keep) == 0:
            return []

        kept_boxes = boxes_xyxy[keep]
        kept_scores = scores[keep]
        kept_labels = labels[keep]

        # Reading order decoding over kept queries
        sub_order = order_scores[keep][:, keep]
        import torch

        seq = self.decode_order_fn(torch.from_numpy(sub_order).float()).tolist()
        rank = [0] * len(seq)
        for r, pos in enumerate(seq):
            rank[pos] = r + 1

        detections = []
        for j in range(len(keep)):
            x0, y0, x1, y1 = kept_boxes[j]
            detections.append(
                {
                    "bbox": [
                        round(float(y0 * 1000.0), 1),
                        round(float(x0 * 1000.0), 1),
                        round(float(y1 * 1000.0), 1),
                        round(float(x1 * 1000.0), 1),
                    ],
                    "label": self.id2label[int(kept_labels[j])],
                    "reading_order": rank[j],
                    "score": round(float(kept_scores[j]), 3),
                }
            )
        return detections

    def detect(self, image: PILImage) -> list[Any]:
        """Detect and return cleaned, reading-ordered Block objects."""
        detections = self.detect_raw(image)
        width, height = image.size

        blocks = []
        for det in detections:
            y0, x0, y1, x1 = det["bbox"]
            bbox = [
                x0 / 1000.0 * width,
                y0 / 1000.0 * height,
                x1 / 1000.0 * width,
                y1 / 1000.0 * height,
            ]
            label = str(det["label"])
            clamped = self.clamp_to_page_fn(bbox, width, height)
            blocks.append(
                self.block_cls(
                    order=det["reading_order"],
                    label=label,
                    type=self.map_label_fn(label),
                    bbox_xyxy=[round(float(v), 1) for v in clamped],
                    conf=round(float(det.get("score", 1.0)), 3),
                )
            )

        # Sort by reading order and renumber gap-free
        cleaned = self.clean_layout_fn(blocks, self.dedup_cfg)
        ordered = sorted(cleaned, key=lambda b: b.order)
        for r, b in enumerate(ordered):
            b.order = r
        return ordered

    def close(self) -> None:
        self.session = None
