#!/usr/bin/env python3
"""
Standalone PDF Annotator for Construction Takeoff
Draws bounding boxes around detected pay items on original PDFs.

Usage:
    python tools/annotate_pdf.py <pdf_file> [--output <path>] [--dpi 200] [--labels]
    python tools/annotate_pdf.py ./plans/drainage.pdf
    python tools/annotate_pdf.py ./plans/ --output ./annotated/
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# PDF handling
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF required. Install with: pip install pymupdf")
    sys.exit(1)

# Image handling
try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL/numpy required. Install with: pip install pillow numpy")
    sys.exit(1)

# OCR
try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Color definitions (RGB tuples, 0-1 range for PyMuPDF)
COLORS = {
    'pipe': (0.2, 0.4, 0.8),      # Blue
    'inlet': (0.2, 0.7, 0.3),     # Green
    'manhole': (0.6, 0.2, 0.6),   # Purple
    'endwall': (0.9, 0.5, 0.1),   # Orange
    'fdot': (0.1, 0.6, 0.6),      # Teal
    'generic': (0.7, 0.7, 0.2),   # Yellow
}


def extract_text_with_bboxes(pdf_path: str, dpi: int = 200) -> List[Dict]:
    """
    Run OCR and return words with bounding boxes.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for image conversion

    Returns:
        List of {text, bbox, confidence, page_num}
        bbox format: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
    """
    if not PADDLEOCR_AVAILABLE:
        logger.error("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")
        return []

    words = []
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')

    # Convert PDF to images
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        # Render page to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Run OCR with new API
        result = ocr.predict(np.array(img))

        if result and len(result) > 0:
            ocr_result = result[0]
            texts = ocr_result.get('rec_texts', [])
            scores = ocr_result.get('rec_scores', [])
            polys = ocr_result.get('dt_polys', [])

            for i, (text, poly) in enumerate(zip(texts, polys)):
                conf = scores[i] if i < len(scores) else 0.0
                # Convert numpy array to list format
                bbox = poly.tolist() if hasattr(poly, 'tolist') else list(poly)
                words.append({
                    'text': text,
                    'bbox': bbox,
                    'confidence': conf,
                    'page_num': page_num
                })

    doc.close()
    return words


def find_pay_item_bboxes(words: List[Dict]) -> List[Dict]:
    """
    Find pay items in OCR words and return their bounding boxes.

    Returns:
        List of {description, page_num, bbox, item_type, pay_item_no}
    """
    detections = []
    seen_items = set()  # Avoid duplicate detections

    # Group words by page and sort by position
    pages = {}
    for word in words:
        page = word['page_num']
        if page not in pages:
            pages[page] = []
        pages[page].append(word)

    # Sort words on each page by Y then X
    for page_num in pages:
        pages[page_num].sort(key=lambda w: (
            (w['bbox'][0][1] + w['bbox'][2][1]) / 2,
            (w['bbox'][0][0] + w['bbox'][2][0]) / 2
        ))

    # Patterns to detect - more flexible for individual words
    patterns = {
        # FDOT codes (highest priority)
        'fdot': [
            r'^(\d{3}-\d{1,3}-\d{1,3})$',  # Exact FDOT code
            r'^(\d{3}-\d{1,2})$',           # Short FDOT code
        ],
        # Pipe materials - single word matches
        'pipe': [
            r'^RCP$',
            r'^PVC$',
            r'^HDPE$',
            r'^CMP$',
            r'^DIP$',
            r'^PIPE$',
            r'^CULVERT$',
        ],
        # Inlets - keywords
        'inlet': [
            r'^INLET$',
            r'^INLETS$',
            r'^DBI$',        # Ditch Bottom Inlet abbrev
            r'^DBI,',        # DBI with comma (merged text)
            r'INLET',        # Inlet anywhere in text
        ],
        # Manholes - keywords
        'manhole': [
            r'^MANHOLE$',
            r'^MANHOLES$',
            r'^MH$',
        ],
        # Endwalls - keywords
        'endwall': [
            r'^ENDWALL$',
            r'^ENDWALLS$',
            r'^MES$',        # Mitered End Section
            r'^MITERED$',
        ],
    }

    # Search each word for patterns
    for page_num, page_words in pages.items():
        for word in page_words:
            text = word['text'].strip().upper()

            # Skip very short words or numbers only
            if len(text) < 2:
                continue

            for item_type, pattern_list in patterns.items():
                matched = False
                for pattern in pattern_list:
                    if re.match(pattern, text, re.IGNORECASE):
                        # Create unique key to avoid duplicates
                        bbox_key = f"{page_num}_{word['bbox'][0][0]}_{word['bbox'][0][1]}"
                        if bbox_key in seen_items:
                            continue
                        seen_items.add(bbox_key)

                        # Extract pay item number if FDOT
                        pay_item_no = None
                        if item_type == 'fdot':
                            match = re.match(pattern, text)
                            if match:
                                pay_item_no = match.group(1)

                        detections.append({
                            'description': word['text'],
                            'page_num': page_num,
                            'bbox': word['bbox'],
                            'item_type': item_type,
                            'pay_item_no': pay_item_no,
                            'confidence': word['confidence']
                        })
                        matched = True
                        break  # Only match first pattern per word
                if matched:
                    break  # Move to next word after finding match

    return detections


def scale_bbox_to_pdf(bbox_pixels: List, ocr_dpi: int = 200) -> fitz.Rect:
    """Convert OCR pixel coordinates to PDF points (72 DPI)."""
    scale = 72 / ocr_dpi
    x_coords = [p[0] for p in bbox_pixels]
    y_coords = [p[1] for p in bbox_pixels]
    return fitz.Rect(
        min(x_coords) * scale,
        min(y_coords) * scale,
        max(x_coords) * scale,
        max(y_coords) * scale
    )


def annotate_pdf(
    pdf_path: str,
    detections: List[Dict],
    output_path: str,
    dpi: int = 200,
    show_labels: bool = False
) -> str:
    """
    Draw bounding boxes on PDF pages.

    Args:
        pdf_path: Original PDF path
        detections: List of detected items with bboxes
        output_path: Where to save annotated PDF
        dpi: DPI used during OCR
        show_labels: Whether to add text labels

    Returns:
        Path to annotated PDF
    """
    doc = fitz.open(pdf_path)

    # Group detections by page
    by_page = {}
    for det in detections:
        page = det['page_num']
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(det)

    for page_num, page_detections in by_page.items():
        if page_num >= len(doc):
            continue

        page = doc[page_num]

        for det in page_detections:
            rect = scale_bbox_to_pdf(det['bbox'], dpi)
            color = COLORS.get(det['item_type'], COLORS['generic'])

            # Draw rectangle with semi-transparent fill
            page.draw_rect(rect, color=color, width=1.5)

            # Add fill with transparency
            shape = page.new_shape()
            shape.draw_rect(rect)
            shape.finish(color=color, fill=color, fill_opacity=0.15)
            shape.commit()

            # Add label if requested
            if show_labels and det.get('pay_item_no'):
                # Position label above the box
                label_point = fitz.Point(rect.x0, rect.y0 - 2)
                page.insert_text(
                    label_point,
                    det['pay_item_no'],
                    fontsize=8,
                    color=color
                )

    # Save annotated PDF
    doc.save(output_path)
    doc.close()

    return output_path


def process_pdf(pdf_path: str, output_dir: str, dpi: int = 200, show_labels: bool = False) -> Tuple[str, int]:
    """
    Process a single PDF file.

    Returns:
        Tuple of (output_path, detection_count)
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{pdf_path.stem}_annotated.pdf"

    logger.info(f"Processing: {pdf_path.name}")

    # Extract text with bounding boxes
    logger.info("  Running OCR...")
    words = extract_text_with_bboxes(str(pdf_path), dpi)
    logger.info(f"  Found {len(words)} text elements")

    # Find pay items
    logger.info("  Detecting pay items...")
    detections = find_pay_item_bboxes(words)
    logger.info(f"  Found {len(detections)} pay item matches")

    # Count by type
    type_counts = {}
    for det in detections:
        t = det['item_type']
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, count in sorted(type_counts.items()):
        logger.info(f"    {t}: {count}")

    # Annotate PDF
    if detections:
        logger.info("  Annotating PDF...")
        annotate_pdf(str(pdf_path), detections, str(output_path), dpi, show_labels)
        logger.info(f"  Saved: {output_path}")
    else:
        logger.info("  No detections - skipping annotation")
        return None, 0

    return str(output_path), len(detections)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate construction PDFs with bounding boxes around detected pay items"
    )
    parser.add_argument(
        "input",
        help="PDF file or folder of PDFs to annotate"
    )
    parser.add_argument(
        "--output", "-o",
        default="./annotated",
        help="Output directory for annotated PDFs (default: ./annotated)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for OCR processing (default: 200)"
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="Add text labels showing pay item numbers"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)

    # Collect PDFs to process
    if input_path.is_file():
        pdf_files = [input_path]
    else:
        pdf_files = list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF"))

    if not pdf_files:
        logger.error(f"No PDF files found in: {input_path}")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF(s) to process")
    logger.info(f"Output directory: {args.output}")
    logger.info("")

    total_detections = 0
    processed = 0

    for pdf_file in pdf_files:
        try:
            output_path, count = process_pdf(
                str(pdf_file),
                args.output,
                args.dpi,
                args.labels
            )
            if output_path:
                processed += 1
                total_detections += count
        except Exception as e:
            logger.error(f"Failed to process {pdf_file.name}: {e}")
        logger.info("")

    logger.info("=" * 50)
    logger.info(f"Complete: {processed}/{len(pdf_files)} PDFs annotated")
    logger.info(f"Total detections: {total_detections}")


if __name__ == "__main__":
    main()
