#!/usr/bin/env python3
"""Interactive Gradio Web Application for Testing IndicOCR.

Provides an end-to-end interface to:
1. Upload document images (or choose from built-in gallery assets).
2. Execute Stage 1 Layout Detection and Stage 2 OCR Transcription.
3. Visualize bounding boxes with interactive hover tooltips showing transcribed text.
4. Inspect and export reading-ordered Markdown, raw text, and structured JSON.

Usage:
    .venv/bin/python gradio_app.py
    .venv/bin/python gradio_app.py --port 7860 --share
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFont

# Ensure model bundle and indic-ocr root are on sys.path
_ROOT = Path(__file__).resolve().parent
_BUNDLE_DIR = _ROOT / "model_bundle"
_EXAMPLES_DIR = _ROOT / "examples"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_DIR))

import gradio as gr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gradio_app")

# Color palette mapped by document entity label/category
LABEL_COLORS: dict[str, str] = {
    "text": "#2563EB",  # Royal Blue
    "paragraph": "#2563EB",
    "title": "#DC2626",  # Red
    "section-title": "#B91C1C",
    "table": "#D97706",  # Amber
    "figure": "#059669",  # Emerald
    "picture": "#059669",
    "header": "#7C3AED",  # Purple
    "page-header": "#7C3AED",
    "page-number": "#6366F1",  # Indigo
    "footer": "#4B5563",  # Slate
    "page-footer": "#4B5563",
    "list": "#0D9488",  # Teal
    "list-item": "#0D9488",
    "equation": "#DB2777",  # Pink
    "formula": "#DB2777",
    "caption": "#CA8A04",  # Gold
    "footnote": "#475569",
}
DEFAULT_COLOR = "#475569"


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a clean system font or fallback to PIL default."""
    font_candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFCompact.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for candidate in font_candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_layout_bboxes(
    image: Image.Image,
    blocks: list[Any],
    show_order: bool = True,
    show_labels: bool = True,
    show_conf: bool = True,
    fill_alpha: float = 0.20,
    line_width: int = 3,
    min_confidence: float = 0.0,
) -> Image.Image:
    """Draw bounding boxes, reading order numbers, and labels on a document image."""
    base_image = image.convert("RGBA")
    overlay = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(base_image)

    font_size = max(12, int(min(base_image.size) * 0.016))
    font = _get_font(size=font_size)

    def _sort_key(b: Any) -> tuple[int, float, int]:
        has_txt = 1 if _resolve_effective_text(b, blocks) else 0
        b_box = getattr(b, "bbox_xyxy", [0, 0, 0, 0])
        box_area = (
            (b_box[2] - b_box[0]) * (b_box[3] - b_box[1]) if len(b_box) == 4 else 0.0
        )
        return (has_txt, -box_area, getattr(b, "order", 0))

    sorted_blocks = sorted(blocks, key=_sort_key)

    for block in sorted_blocks:
        conf = getattr(block, "conf", 1.0)
        if conf < min_confidence:
            continue

        raw_label = getattr(block, "label", "text")
        color_key = raw_label.strip().lower()
        hex_color = LABEL_COLORS.get(color_key, DEFAULT_COLOR)
        rgb_color = ImageColor.getrgb(hex_color)

        bbox = getattr(block, "bbox_xyxy", None)
        if not bbox or len(bbox) != 4:
            continue

        x0, y0, x1, y1 = [float(v) for v in bbox]
        left, right = min(x0, x1), max(x0, x1)
        top, bottom = min(y0, y1), max(y0, y1)

        # Draw semi-transparent fill
        if fill_alpha > 0.0:
            fill_color = (*rgb_color, int(255 * fill_alpha))
            draw_overlay.rectangle([left, top, right, bottom], fill=fill_color)

        # Draw outline border
        draw_base.rectangle(
            [left, top, right, bottom],
            outline=rgb_color,
            width=line_width,
        )

        # Compose badge text
        order = getattr(block, "order", 0)
        parts: list[str] = []
        if show_order and order > 0:
            parts.append(f"#{order}")
        if show_labels:
            parts.append(str(raw_label))
        if show_conf and conf > 0.0:
            parts.append(f"{conf:.2f}")

        if not parts:
            continue

        badge_text = " ".join(parts)

        text_bbox = draw_base.textbbox((0, 0), badge_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        pad_x, pad_y = 6, 3

        badge_y0 = top - text_h - (2 * pad_y)
        if badge_y0 < 0:
            badge_y0 = top + 2

        badge_x0 = max(0.0, left)
        badge_x1 = min(float(base_image.width), badge_x0 + text_w + (2 * pad_x))
        badge_y1 = badge_y0 + text_h + (2 * pad_y)

        draw_base.rectangle([badge_x0, badge_y0, badge_x1, badge_y1], fill=rgb_color)
        draw_base.text(
            (badge_x0 + pad_x, badge_y0 + pad_y),
            badge_text,
            fill=(255, 255, 255, 255),
            font=font,
        )

    result = Image.alpha_composite(base_image, overlay)
    return result.convert("RGB")


TOOLTIP_HEAD = """
<script>
(function() {
  if (window.__ocrTooltipInstalled) return;
  window.__ocrTooltipInstalled = true;

  function getTooltip() {
    let el = document.getElementById('ocr-hover-tooltip');
    if (!el) {
      el = document.createElement('div');
      el.id = 'ocr-hover-tooltip';
      el.style.cssText = `
        position: fixed;
        display: none;
        z-index: 2147483647;
        pointer-events: none;
        max-width: 440px;
        min-width: 240px;
        background: rgba(15, 23, 42, 0.96);
        color: #f8fafc;
        padding: 12px 14px;
        border-radius: 8px;
        box-shadow: 0 10px 30px -5px rgba(0,0,0,0.6), 0 0 1px 1px rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        transition: opacity 0.1s ease-out;
        opacity: 0;
      `;
      document.body.appendChild(el);
    }
    return el;
  }

  function showTooltip(target, evt) {
    const tooltip = getTooltip();
    const order = target.getAttribute('data-order') || '';
    const label = target.getAttribute('data-label') || '';
    const conf = target.getAttribute('data-conf') || '';
    const color = target.getAttribute('data-color') || '#2563EB';
    const rawText = target.getAttribute('data-text') || '';

    const displayText = rawText.trim()
      ? rawText.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\\n/g, '<br/>')
      : '<span style="color:#94a3b8; font-style:italic;">(No transcribed text)</span>';

    const confPercent = conf ? Math.round(parseFloat(conf) * 100) : 0;

    tooltip.innerHTML = `
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.15); padding-bottom:6px;">
        <span style="background:${color}; color:#ffffff; font-size:12px; font-weight:700; padding:2px 8px; border-radius:4px;">#${order} ${label}</span>
        <span style="font-size:11px; color:#94a3b8; font-weight:600;">Conf: ${confPercent}%</span>
      </div>
      <div style="font-size:14px; line-height:1.55; max-height:260px; overflow-y:auto; color:#f8fafc; word-break:break-word;">
        ${displayText}
      </div>
      <div style="margin-top:8px; font-size:11px; color:#64748b; text-align:right;">
        Click box to copy text
      </div>
    `;
    tooltip.style.display = 'block';
    tooltip.style.opacity = '1';
    moveTooltip(evt);
  }

  function moveTooltip(evt) {
    const tooltip = getTooltip();
    if (tooltip.style.display === 'none') return;
    const pad = 16;
    let x = evt.clientX + pad;
    let y = evt.clientY + pad;
    const rect = tooltip.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - pad) {
      x = evt.clientX - rect.width - pad;
    }
    if (y + rect.height > window.innerHeight - pad) {
      y = evt.clientY - rect.height - pad;
    }
    tooltip.style.left = Math.max(pad, x) + 'px';
    tooltip.style.top = Math.max(pad, y) + 'px';
  }

  function hideTooltip() {
    const tooltip = getTooltip();
    tooltip.style.opacity = '0';
    tooltip.style.display = 'none';
  }

  function copyText(target) {
    const text = target.getAttribute('data-text') || '';
    if (text && navigator.clipboard) {
      navigator.clipboard.writeText(text);
      const tooltip = getTooltip();
      const alert = document.createElement('div');
      alert.style.cssText = "background:#10b981; color:#fff; font-size:11px; font-weight:bold; padding:4px 8px; border-radius:4px; margin-top:6px; text-align:center;";
      alert.innerText = "✓ Copied to clipboard!";
      tooltip.appendChild(alert);
      setTimeout(() => { if (alert.parentNode) alert.remove(); }, 1200);
    }
  }

  // Expose on window for inline handlers
  window.showOcrTooltip = showTooltip;
  window.moveOcrTooltip = moveTooltip;
  window.hideOcrTooltip = hideTooltip;
  window.copyOcrText = copyText;

  // Global document event delegation (works across dynamic renders)
  document.addEventListener('mouseover', function(e) {
    const group = e.target.closest && e.target.closest('.ocr-box-group');
    if (group) showTooltip(group, e);
  });
  document.addEventListener('mousemove', function(e) {
    const group = e.target.closest && e.target.closest('.ocr-box-group');
    if (group) moveTooltip(e);
  });
  document.addEventListener('mouseout', function(e) {
    const group = e.target.closest && e.target.closest('.ocr-box-group');
    if (group) {
      const rel = e.relatedTarget;
      if (!rel || !group.contains(rel)) {
        hideTooltip();
      }
    }
  });
  document.addEventListener('click', function(e) {
    const group = e.target.closest && e.target.closest('.ocr-box-group');
    if (group) copyText(group);
  });
})();
</script>
"""


def _resolve_effective_text(block: Any, all_blocks: list[Any]) -> str:
    """Get transcribed text for a block, falling back to overlapping block if skipped by OCR."""
    text = (getattr(block, "text", "") or "").strip()
    if text:
        return text

    b_box = getattr(block, "bbox_xyxy", None)
    if not b_box or len(b_box) != 4:
        return ""

    area_b = max(1.0, (b_box[2] - b_box[0]) * (b_box[3] - b_box[1]))
    best_iou = 0.0
    best_text = ""

    for other in all_blocks:
        if other is block:
            continue
        other_text = (getattr(other, "text", "") or "").strip()
        if not other_text:
            continue
        o_box = getattr(other, "bbox_xyxy", None)
        if not o_box or len(o_box) != 4:
            continue

        ix0 = max(b_box[0], o_box[0])
        iy0 = max(b_box[1], o_box[1])
        ix1 = min(b_box[2], o_box[2])
        iy1 = min(b_box[3], o_box[3])
        if ix1 <= ix0 or iy1 <= iy0:
            continue

        inter = (ix1 - ix0) * (iy1 - iy0)
        area_o = max(1.0, (o_box[2] - o_box[0]) * (o_box[3] - o_box[1]))
        iou = inter / (area_b + area_o - inter)
        frac = inter / min(area_b, area_o)

        if (iou > 0.35 or frac > 0.70) and iou > best_iou:
            best_iou = iou
            best_text = other_text

    return best_text


def build_interactive_svg(
    image: Image.Image,
    blocks: list[Any],
    show_order: bool = True,
    show_labels: bool = True,
    show_conf: bool = True,
    fill_alpha: float = 0.18,
    line_width: int = 2,
    min_confidence: float = 0.0,
) -> str:
    """Generate interactive SVG overlay with hover tooltips displaying transcribed text."""
    orig_w, orig_h = image.size
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    b64_data = base64.b64encode(buf.getvalue()).decode("ascii")
    data_uri = f"data:image/jpeg;base64,{b64_data}"

    font_size = max(11, int(min(orig_w, orig_h) * 0.016))
    badge_h = font_size + 8

    box_elements: list[str] = []

    # Sort blocks so that empty background boxes are drawn first,
    # and transcribed / smaller nested boxes are drawn in front.
    def _sort_key(b: Any) -> tuple[int, float, int]:
        has_txt = 1 if _resolve_effective_text(b, blocks) else 0
        b_box = getattr(b, "bbox_xyxy", [0, 0, 0, 0])
        box_area = (
            (b_box[2] - b_box[0]) * (b_box[3] - b_box[1]) if len(b_box) == 4 else 0.0
        )
        return (has_txt, -box_area, getattr(b, "order", 0))

    sorted_blocks = sorted(blocks, key=_sort_key)

    for block in sorted_blocks:
        conf = getattr(block, "conf", 1.0)
        if conf < min_confidence:
            continue

        raw_label = getattr(block, "label", "text")
        color_key = raw_label.strip().lower()
        hex_color = LABEL_COLORS.get(color_key, DEFAULT_COLOR)
        rgb_color = ImageColor.getrgb(hex_color)
        fill_rgba = (
            f"rgba({rgb_color[0]},{rgb_color[1]},{rgb_color[2]},{fill_alpha:.2f})"
        )

        bbox = getattr(block, "bbox_xyxy", None)
        if not bbox or len(bbox) != 4:
            continue

        x0, y0, x1, y1 = [float(v) for v in bbox]
        left, right = min(x0, x1), max(x0, x1)
        top, bottom = min(y0, y1), max(y0, y1)
        box_w = max(1.0, right - left)
        box_h = max(1.0, bottom - top)

        order = getattr(block, "order", 0)
        parts: list[str] = []
        if show_order and order > 0:
            parts.append(f"#{order}")
        if show_labels:
            parts.append(str(raw_label))
        if show_conf and conf > 0.0:
            parts.append(f"{conf:.2f}")

        badge_text = " ".join(parts) if parts else f"#{order}"
        badge_w = (len(badge_text) * (font_size * 0.62)) + 12

        badge_y0 = top - badge_h
        if badge_y0 < 0:
            badge_y0 = top + 2
        badge_x0 = max(0.0, left)

        transcribed_text = _resolve_effective_text(block, blocks)
        data_text_attr = html.escape(transcribed_text, quote=True)
        native_title = html.escape(
            f"#{order} {raw_label} ({conf:.2f})"
            + (
                f"\n\n{transcribed_text}"
                if transcribed_text
                else "\n\n(No transcribed text)"
            )
        )

        badge_rect = f"""
            <rect x="{badge_x0:.1f}" y="{badge_y0:.1f}" width="{badge_w:.1f}" height="{badge_h:.1f}"
                  fill="{hex_color}" rx="3" class="ocr-badge-bg" />
            <text x="{badge_x0 + 6:.1f}" y="{badge_y0 + badge_h - 5:.1f}"
                  fill="#ffffff" font-size="{font_size}px" font-family="system-ui, -apple-system, sans-serif"
                  font-weight="600" style="user-select:none;">{html.escape(badge_text)}</text>
        """

        box_elements.append(f"""
        <g class="ocr-box-group"
           data-order="{order}"
           data-label="{html.escape(str(raw_label))}"
           data-conf="{conf:.2f}"
           data-color="{hex_color}"
           data-text="{data_text_attr}"
           tabindex="0"
           onmouseenter="window.showOcrTooltip && window.showOcrTooltip(this, event)"
           onmousemove="window.moveOcrTooltip && window.moveOcrTooltip(event)"
           onmouseleave="window.hideOcrTooltip && window.hideOcrTooltip()"
           onclick="window.copyOcrText && window.copyOcrText(this)"
           style="cursor: pointer;">
          <rect x="{left:.1f}" y="{top:.1f}" width="{box_w:.1f}" height="{box_h:.1f}"
                fill="{fill_rgba}" stroke="{hex_color}" stroke-width="{line_width}"
                rx="3" class="ocr-box-rect" />
          {badge_rect}
          <title>{native_title}</title>
        </g>
        """)

    all_boxes = "\n".join(box_elements)

    html_content = f"""
    <div class="ocr-interactive-wrapper" style="position: relative; width: 100%; display: flex; justify-content: center; background: transparent; border-radius: 8px; padding: 12px; overflow: auto;">
      <style>
        .ocr-box-group {{
          transition: transform 0.1s ease, filter 0.1s ease;
        }}
        .ocr-box-group:hover .ocr-box-rect {{
          stroke: #ffffff !important;
          stroke-width: {line_width + 2}px !important;
          filter: drop-shadow(0 0 6px rgba(0,0,0,0.7));
        }}
        .ocr-box-group:hover .ocr-badge-bg {{
          filter: brightness(1.25);
        }}
      </style>

      <svg viewBox="0 0 {orig_w} {orig_h}"
           style="width: 100%; max-width: 950px; height: auto; display: block; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.12);">
        <image href="{data_uri}" width="{orig_w}" height="{orig_h}" />
        {all_boxes}
      </svg>
    </div>
    """
    return html_content


class OCRServerEngine:
    """Manages cached OCR pipelines across Apple MLX and ONNX backends."""

    def __init__(self, bundle_dir: Path | None = None) -> None:
        self.bundle_dir = bundle_dir or _BUNDLE_DIR
        self._mlx_pipelines: dict[str, Any] = {}
        self._onnx_pipelines: dict[str, Any] = {}

    def get_mlx_pipeline(self, recognizer_variant: str = "mlx-4bit") -> Any:
        """Get or initialize Apple MLX pipeline with chosen quantization variant."""
        from pipeline_mlx import MlxIndicOCR

        if recognizer_variant not in self._mlx_pipelines:
            logger.info(
                "Initializing MlxIndicOCR backend for '%s'...", recognizer_variant
            )
            weights_map = {
                "mlx-4bit": _ROOT / "mlx_output" / "ocr_4bit",
                "mlx-8bit": _ROOT / "mlx_output" / "ocr_8bit",
                "mlx-bf16": _ROOT / "mlx_output" / "ocr_bf16",
            }
            weights_path = weights_map.get(
                recognizer_variant, _ROOT / "mlx_output" / "ocr_4bit"
            )
            if not weights_path.exists():
                logger.warning(
                    "Weights path %s not found. Falling back to default available weights.",
                    weights_path,
                )
                weights_path = None

            pipeline = MlxIndicOCR(
                ocr_model_path=weights_path,
                bundle_dir=self.bundle_dir,
            )
            self._mlx_pipelines[recognizer_variant] = pipeline

        return self._mlx_pipelines[recognizer_variant]

    def get_onnx_pipeline(self, recognizer_backend: str = "onnx") -> Any:
        """Get or initialize ONNX pipeline."""
        from pipeline_onnx import OnnxIndicOCR

        if recognizer_backend not in self._onnx_pipelines:
            logger.info(
                "Initializing OnnxIndicOCR backend for '%s'...", recognizer_backend
            )
            pipeline = OnnxIndicOCR(
                bundle_dir=self.bundle_dir,
                ocr_backend=recognizer_backend,
                device="cpu",
            )
            self._onnx_pipelines[recognizer_backend] = pipeline

        return self._onnx_pipelines[recognizer_backend]

    def process(
        self,
        image_path: str | Path,
        mode: str = "Layout + OCR (Full)",
        backend: str = "mlx-4bit",
        min_confidence: float = 0.40,
    ) -> tuple[Any, float]:
        """Execute document layout and optional transcription."""
        path_str = str(image_path)
        start_time = time.perf_counter()

        if backend.startswith("mlx"):
            pipeline = self.get_mlx_pipeline(backend)
        else:
            pipeline = self.get_onnx_pipeline(backend)

        logger.info("Running Stage 1 Layout Detection on %s...", Path(path_str).name)
        layout_result = pipeline.layout.detect(path_str)

        # Filter blocks below confidence threshold
        kept_blocks = [
            b for b in layout_result.blocks if getattr(b, "conf", 1.0) >= min_confidence
        ]
        layout_result.blocks = kept_blocks

        if mode == "Layout Detection Only":
            logger.info("Skipping Stage 2 transcription as requested.")
            summary_lines = [
                f"# Layout Detection Summary for {Path(path_str).name}\n",
                f"- **Detected Blocks:** {len(kept_blocks)}",
                "- **Mode:** Layout Detection Only\n",
                "## Detected Regions\n",
            ]
            for b in kept_blocks:
                summary_lines.append(
                    f"- **#{b.order}** [{b.label}] `{b.type}` | Conf: {b.conf:.3f} | BBox: `{b.bbox_xyxy}`"
                )
            layout_result.markdown = "\n".join(summary_lines)
            elapsed = time.perf_counter() - start_time
            return layout_result, elapsed

        logger.info(
            "Running Stage 2 OCR Transcription on %d blocks...", len(kept_blocks)
        )
        page_result = pipeline.ocr.run(path_str, layout_result)
        elapsed = time.perf_counter() - start_time
        logger.info("OCR completed in %.2f seconds.", elapsed)
        return page_result, elapsed


# Global engine instance
_ENGINE: OCRServerEngine | None = None


def get_engine() -> OCRServerEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = OCRServerEngine()
    return _ENGINE


def run_ocr(
    input_image: Any,
    mode: str,
    backend: str,
    confidence_threshold: float,
    show_order: bool,
    show_labels: bool,
    show_conf: bool,
    fill_boxes: bool,
    line_width: int,
) -> tuple[
    str,
    Image.Image | None,
    str,
    str,
    list[list[Any]],
    dict[str, Any],
    str | None,
    str | None,
    str,
]:
    """Gradio event handler for processing an image."""
    if input_image is None:
        return (
            "<div style='text-align:center; padding:40px; color:#64748b;'>Please upload an image to begin.</div>",
            None,
            "Please upload an image to begin.",
            "",
            [],
            {},
            None,
            None,
            "Ready.",
        )

    temp_path: Path | None = None
    if isinstance(input_image, str):
        image_path = Path(input_image)
        pil_img = Image.open(image_path).convert("RGB")
    elif isinstance(input_image, Image.Image):
        pil_img = input_image.convert("RGB")
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil_img.save(temp_file.name)
        temp_path = Path(temp_file.name)
        image_path = temp_path
    elif hasattr(input_image, "name"):
        image_path = Path(input_image.name)
        pil_img = Image.open(image_path).convert("RGB")
    else:
        pil_img = Image.fromarray(input_image).convert("RGB")
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil_img.save(temp_file.name)
        temp_path = Path(temp_file.name)
        image_path = temp_path

    try:
        engine = get_engine()
        result, elapsed = engine.process(
            image_path=image_path,
            mode=mode,
            backend=backend,
            min_confidence=confidence_threshold,
        )

        fill_alpha = 0.22 if fill_boxes else 0.0

        # Generate interactive SVG with hover tooltips
        interactive_html = build_interactive_svg(
            image=pil_img,
            blocks=result.blocks,
            show_order=show_order,
            show_labels=show_labels,
            show_conf=show_conf,
            fill_alpha=fill_alpha,
            line_width=int(line_width),
            min_confidence=confidence_threshold,
        )

        # Generate static PIL image for download
        annotated_image = draw_layout_bboxes(
            image=pil_img,
            blocks=result.blocks,
            show_order=show_order,
            show_labels=show_labels,
            show_conf=show_conf,
            fill_alpha=fill_alpha,
            line_width=int(line_width),
            min_confidence=confidence_threshold,
        )

        # Prepare Markdown & Plain Text
        markdown_text = getattr(result, "markdown", "") or ""

        # Prepare Per-Block Structured Table
        table_rows: list[list[Any]] = []
        for b in result.blocks:
            bbox_str = f"[{b.bbox_xyxy[0]:.1f}, {b.bbox_xyxy[1]:.1f}, {b.bbox_xyxy[2]:.1f}, {b.bbox_xyxy[3]:.1f}]"
            table_rows.append(
                [
                    b.order,
                    b.label,
                    b.type,
                    f"{b.conf:.3f}",
                    bbox_str,
                    getattr(b, "text", "") or _resolve_effective_text(b, result.blocks),
                ]
            )

        # Prepare JSON data
        json_data = (
            result.as_record()
            if hasattr(result, "as_record")
            else {"blocks": [b.as_record() for b in result.blocks]}
        )

        # Write export files for download buttons
        md_export = tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        )
        md_export.write(markdown_text)
        md_export.close()

        json_export = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(json_data, json_export, indent=2, ensure_ascii=False)
        json_export.close()

        transcribed_count = sum(
            1 for b in result.blocks if getattr(b, "text", None) is not None
        )
        status_info = (
            f"Processed in **{elapsed:.2f}s** | "
            f"**{len(result.blocks)}** blocks detected | "
            f"**{transcribed_count}** transcribed | "
            f"Image size: **{pil_img.width}x{pil_img.height}**"
        )

        return (
            interactive_html,
            annotated_image,
            markdown_text,
            markdown_text,
            table_rows,
            json_data,
            md_export.name,
            json_export.name,
            status_info,
        )

    except Exception as exc:
        logger.exception("Error running OCR pipeline: %s", exc)
        error_msg = f"Pipeline execution failed: {type(exc).__name__}: {str(exc)}"
        return (
            f"<div style='padding:20px; color:#ef4444;'><strong>Error:</strong> {html.escape(error_msg)}</div>",
            pil_img,
            f"### Error\n\n```\n{error_msg}\n```",
            error_msg,
            [],
            {"error": error_msg},
            None,
            None,
            f"Failed: {error_msg}",
        )
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


def build_app() -> gr.Blocks:
    """Construct Gradio UI components and layout."""
    with gr.Blocks(title="IndicOCR Testing Server") as demo:
        gr.Markdown("""
            # IndicOCR Testing Studio
            Upload a document image to detect reading layout regions, transcribe text, and visualize bounding boxes with **interactive hover tooltips** and reading order.
            """)

        with gr.Row():
            # Left column: Input and Controls
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Document Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                )

                with gr.Accordion("Pipeline Configuration", open=True):
                    mode_radio = gr.Radio(
                        label="Execution Mode",
                        choices=["Layout + OCR (Full)", "Layout Detection Only"],
                        value="Layout + OCR (Full)",
                    )
                    backend_dropdown = gr.Dropdown(
                        label="Inference Backend",
                        choices=[
                            "mlx-4bit",
                            "mlx-8bit",
                            "mlx-bf16",
                        ],
                        value="mlx-4bit",
                        info="MLX backends optimized for Apple Silicon (Metal GPU).",
                    )
                    confidence_slider = gr.Slider(
                        label="Layout Confidence Threshold",
                        minimum=0.1,
                        maximum=0.95,
                        step=0.05,
                        value=0.40,
                    )

                with gr.Accordion("BBox Visualization Settings", open=False):
                    with gr.Row():
                        show_order_cb = gr.Checkbox(
                            label="Reading Order (#)", value=True
                        )
                        show_labels_cb = gr.Checkbox(label="Class Labels", value=True)
                    with gr.Row():
                        show_conf_cb = gr.Checkbox(
                            label="Confidence Scores", value=True
                        )
                        fill_boxes_cb = gr.Checkbox(label="Color Tint Fill", value=True)
                    line_width_slider = gr.Slider(
                        label="BBox Stroke Width",
                        minimum=1,
                        maximum=6,
                        step=1,
                        value=2,
                    )

                submit_btn = gr.Button("Run OCR Pipeline", variant="primary", size="lg")

                sample_dir = _EXAMPLES_DIR
                sample_files = (
                    sorted(
                        list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))
                    )
                    if sample_dir.exists()
                    else []
                )
                if sample_files:
                    gr.Examples(
                        examples=[[str(p)] for p in sample_files],
                        inputs=image_input,
                        label="Sample Test Images (Hindi & English)",
                    )

            # Right column: Visualizations and Transcribed Outputs
            with gr.Column(scale=2):
                gr.HTML("""
                    <style>
                    .metric-box {
                        background: var(--block-background-fill, var(--background-fill-secondary, rgba(255, 255, 255, 0.05))) !important;
                        color: var(--body-text-color, inherit) !important;
                        border-radius: 8px;
                        padding: 8px 14px;
                        border: 1px solid var(--block-border-color, var(--border-color-primary, rgba(128, 128, 128, 0.2))) !important;
                        font-size: 0.92rem;
                        margin-bottom: 8px;
                    }
                    .metric-box p {
                        margin: 0 !important;
                        color: inherit !important;
                    }
                    </style>
                    """)
                status_bar = gr.Markdown(
                    "Ready for input.", elem_classes=["metric-box"]
                )

                with gr.Tabs():
                    with gr.Tab("Layout & BBoxes"):
                        with gr.Tabs():
                            with gr.Tab("Interactive (Hover Tooltips)"):
                                interactive_html_output = gr.HTML(
                                    label="Interactive BBox Hover Viewer",
                                    value="<div style='text-align:center; padding:40px; color:#64748b;'>Upload an image and click <strong>Run OCR Pipeline</strong> to see interactive hover tooltips with transcribed text.</div>",
                                    head=TOOLTIP_HEAD,
                                )
                            with gr.Tab("Static Annotated Image"):
                                annotated_output = gr.Image(
                                    label="Static Annotated Image",
                                    type="pil",
                                    interactive=False,
                                )

                    with gr.Tab("Rendered Markdown"):
                        markdown_output = gr.Markdown(
                            label="Reading-Ordered Markdown Output"
                        )

                    with gr.Tab("Raw Text"):
                        raw_text_output = gr.Textbox(
                            label="Transcribed Text / Markdown",
                            lines=18,
                        )

                    with gr.Tab("Detected Blocks Table"):
                        blocks_table = gr.Dataframe(
                            headers=[
                                "#",
                                "Label",
                                "Type",
                                "Conf",
                                "Bounding Box [x0, y0, x1, y1]",
                                "Transcribed Text",
                            ],
                            datatype=["number", "str", "str", "str", "str", "str"],
                            interactive=False,
                            wrap=True,
                        )

                    with gr.Tab("Structured JSON"):
                        json_output = gr.JSON(label="Full Page Result JSON")

                with gr.Row():
                    download_md_btn = gr.DownloadButton(
                        label="Download Markdown (.md)",
                        variant="secondary",
                    )
                    download_json_btn = gr.DownloadButton(
                        label="Download JSON (.json)",
                        variant="secondary",
                    )

        # Wire click and submit actions
        submit_btn.click(
            fn=run_ocr,
            inputs=[
                image_input,
                mode_radio,
                backend_dropdown,
                confidence_slider,
                show_order_cb,
                show_labels_cb,
                show_conf_cb,
                fill_boxes_cb,
                line_width_slider,
            ],
            outputs=[
                interactive_html_output,
                annotated_output,
                markdown_output,
                raw_text_output,
                blocks_table,
                json_output,
                download_md_btn,
                download_json_btn,
                status_bar,
            ],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch Gradio Web Server for testing IndicOCR."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host IP to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to listen on (default: 7860).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable Gradio link.",
    )
    args = parser.parse_args()

    custom_css = """
    .gradio-container {
        max-width: 1440px !important;
        margin: auto;
    }
    .metric-box {
        background: var(--block-background-fill, var(--background-fill-secondary, rgba(255, 255, 255, 0.05))) !important;
        color: var(--body-text-color, inherit) !important;
        border-radius: 8px;
        padding: 8px 14px;
        border: 1px solid var(--block-border-color, var(--border-color-primary, rgba(128, 128, 128, 0.2))) !important;
        font-size: 0.92rem;
        margin-bottom: 8px;
    }
    .metric-box p {
        margin: 0 !important;
        color: inherit !important;
    }
    """

    app = build_app()
    logger.info("Starting Gradio OCR test server on %s:%d...", args.host, args.port)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=custom_css,
        head=TOOLTIP_HEAD,
    )


if __name__ == "__main__":
    main()
