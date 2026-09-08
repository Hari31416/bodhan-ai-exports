#!/usr/bin/env python3
"""Apple Silicon MLX layout detection backend for IndicOCR (Stage 1).

Implements MlxIndicDocLayout conforming to the LayoutBackend protocol,
executing page-level document layout analysis on Apple Silicon Metal GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layout_mlx")


class MlxIndicDocLayout:
    """Apple Silicon MLX layout detection backend."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        bundle_dir: str | Path | None = None,
        conf_threshold: float = 0.5,
        img_size: int = 1024,
        device: str = "mps",
    ) -> None:
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.device = device

        root = Path(__file__).parent
        if bundle_dir is None:
            bundle_dir = root / "model_bundle"
        self.bundle_dir = Path(bundle_dir)

        if weights_path is None:
            weights_path = root / "mlx_output" / "layout" / "model.safetensors"
            if not weights_path.exists():
                weights_path = self.bundle_dir / "weights" / "layout"

        self.weights_path = Path(weights_path)
        logger.info(
            "Initializing MLX layout detector backend from: %s", self.weights_path
        )

        # Import helper utilities from model bundle
        sys.path.insert(0, str(root))
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

        # Load layout model weights
        from idp_model_ppdoc import PPDocLayoutV3Trainable
        import torch

        # Use Metal Performance Shaders (MPS) or CPU for zero-hybrid layout execution on Mac
        ckpt_dir = self.bundle_dir / "weights" / "layout"
        self.model = PPDocLayoutV3Trainable.from_pretrained(str(ckpt_dir)).eval()
        if self.device == "mps" and torch.backends.mps.is_available():
            self.model = self.model.to("mps")
            logger.info("MLX layout detector loaded on Apple Silicon Metal GPU (MPS).")
        else:
            self.model = self.model.to("cpu")
            logger.info("MLX layout detector loaded on CPU.")

    def _cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, w, h] to [x0, y0, x1, y1]."""
        out = np.empty_like(boxes)
        out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0
        out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0
        out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0
        out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0
        return out

    def detect(self, image_input: str | Path | PILImage) -> Any:
        """Detect document layout elements with reading order."""
        import torch

        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        orig_w, orig_h = pil_img.size

        # Preprocess to 1024x1024
        resized = pil_img.resize((self.img_size, self.img_size))
        img_np = np.array(resized).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        target_dev = next(self.model.parameters()).device
        tensor = tensor.to(target_dev)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values=tensor)
            logits = outputs.logits.cpu().numpy()
            pred_boxes = outputs.pred_boxes.cpu().numpy()
            order_logits = outputs.order_logits.cpu().numpy()

        # Postprocess predictions
        probs = 1.0 / (1.0 + np.exp(-logits[0]))
        scores = probs.max(axis=-1)
        labels = probs.argmax(axis=-1)
        boxes_xyxy = self._cxcywh_to_xyxy(pred_boxes[0])

        keep = np.where(scores > self.conf_threshold)[0]
        if len(keep) == 0:
            return []

        kept_boxes = boxes_xyxy[keep]
        kept_scores = scores[keep]
        kept_labels = labels[keep]

        # Reading order
        order_matrix = order_logits[0][keep][:, keep]
        order_tensor = torch.from_numpy(order_matrix).float()
        sequence = self.decode_order_fn(order_tensor).tolist()
        rank = [0] * len(sequence)
        for r, pos in enumerate(sequence):
            rank[pos] = r + 1

        # Scale coordinates back to original image dimensions
        blocks = []
        for i in range(len(keep)):
            x0, y0, x1, y1 = kept_boxes[i]
            scaled_box = [
                round(x0 * orig_w, 1),
                round(y0 * orig_h, 1),
                round(x1 * orig_w, 1),
                round(y1 * orig_h, 1),
            ]
            clamped = [
                round(v, 1) for v in self.clamp_to_page_fn(scaled_box, orig_w, orig_h)
            ]
            lbl = self.id2label[int(kept_labels[i])]

            blocks.append(
                self.block_cls(
                    order=rank[i],
                    label=lbl,
                    type=self.map_label_fn(lbl),
                    bbox_xyxy=clamped,
                    conf=round(float(kept_scores[i]), 3),
                )
            )

        # Clean layout (deduplication & overlaps)
        cleaned_blocks = self.clean_layout_fn(blocks, self.dedup_cfg)
        ordered_blocks = sorted(cleaned_blocks, key=lambda b: b.order)
        for r, block in enumerate(ordered_blocks):
            block.order = r

        return ordered_blocks

    def close(self) -> None:
        self.model = None
