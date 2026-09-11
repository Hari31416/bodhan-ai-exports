from __future__ import annotations

from pathlib import Path
import sys

from PIL import Image
import pytest

# Ensure indic-ocr root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gradio_app import build_app, build_interactive_svg, draw_layout_bboxes, run_ocr
from model_bundle.idp_types import Block


def test_draw_layout_bboxes() -> None:
    """Test bounding box drawing on an image."""
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    blocks = [
        Block(
            order=1,
            label="Title",
            type="Text",
            bbox_xyxy=[10.0, 10.0, 150.0, 50.0],
            conf=0.98,
            text="Document Title",
        ),
        Block(
            order=2,
            label="Text",
            type="Text",
            bbox_xyxy=[10.0, 60.0, 380.0, 200.0],
            conf=0.92,
            text="Paragraph body content here.",
        ),
    ]

    annotated = draw_layout_bboxes(
        image=img,
        blocks=blocks,
        show_order=True,
        show_labels=True,
        show_conf=True,
        fill_alpha=0.2,
        line_width=2,
    )

    assert annotated is not None
    assert annotated.size == (400, 400)
    assert annotated.mode == "RGB"


def test_build_interactive_svg() -> None:
    """Test interactive SVG generation with hover data attributes."""
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    blocks = [
        Block(
            order=1,
            label="Title",
            type="Text",
            bbox_xyxy=[10.0, 10.0, 150.0, 50.0],
            conf=0.98,
            text="सामान्य हिन्दी",
        ),
    ]

    svg_html = build_interactive_svg(
        image=img,
        blocks=blocks,
        show_order=True,
        show_labels=True,
        show_conf=True,
    )

    assert svg_html is not None
    assert "<svg" in svg_html
    assert "data-text=" in svg_html
    assert "सामान्य हिन्दी" in svg_html
    assert "ocr-box-group" in svg_html


def test_build_app() -> None:
    """Test that the Gradio app builds properly."""
    app = build_app()
    assert app is not None
    assert hasattr(app, "launch")


def test_run_ocr_layout_only() -> None:
    """Test layout detection only on sample image."""
    sample_img_path = _ROOT / "examples" / "hindi_sample_1.png"
    if not sample_img_path.exists():
        pytest.skip("Sample test image not found.")

    (
        interactive_html,
        annotated_image,
        markdown_text,
        raw_text,
        table_rows,
        json_data,
        md_path,
        json_path,
        status_info,
    ) = run_ocr(
        input_image=str(sample_img_path),
        mode="Layout Detection Only",
        backend="mlx-4bit",
        confidence_threshold=0.4,
        show_order=True,
        show_labels=True,
        show_conf=True,
        fill_boxes=True,
        line_width=2,
    )

    assert interactive_html is not None
    assert "<svg" in interactive_html
    assert annotated_image is not None
    assert len(table_rows) > 0
    assert "blocks detected" in status_info
    assert json_data is not None


def test_run_ocr_full() -> None:
    """Test full layout + OCR on sample image using MLX."""
    sample_img_path = _ROOT / "examples" / "hindi_sample_1.png"
    if not sample_img_path.exists():
        pytest.skip("Sample test image not found.")

    (
        interactive_html,
        annotated_image,
        markdown_text,
        raw_text,
        table_rows,
        json_data,
        md_path,
        json_path,
        status_info,
    ) = run_ocr(
        input_image=str(sample_img_path),
        mode="Layout + OCR (Full)",
        backend="mlx-4bit",
        confidence_threshold=0.4,
        show_order=True,
        show_labels=True,
        show_conf=True,
        fill_boxes=True,
        line_width=2,
    )

    assert interactive_html is not None
    assert "<svg" in interactive_html
    assert annotated_image is not None
    assert len(table_rows) > 0
    assert "transcribed" in status_info
    assert len(markdown_text) > 0
