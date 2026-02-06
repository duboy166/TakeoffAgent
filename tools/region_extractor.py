"""
Region Extractor Utility
Extracts image regions from PDFs based on bounding box coordinates.
"""

import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def extract_region(
    pdf_path: str,
    page_num: int,
    bbox: List[List[float]],
    dpi: int = 150,
    padding: int = 20
) -> Optional[bytes]:
    """
    Extract an image region from a PDF page.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        bbox: Bounding box as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        dpi: Resolution for rendering
        padding: Pixels to add around the bbox

    Returns:
        PNG image bytes of the extracted region, or None on failure
    """
    if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE:
        logger.error("PyMuPDF and Pillow required for region extraction")
        return None

    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return None

        doc = fitz.open(str(pdf_path))

        # Convert to 0-indexed page number
        page_idx = page_num - 1
        if page_idx < 0 or page_idx >= len(doc):
            logger.error(f"Invalid page number: {page_num}")
            doc.close()
            return None

        page = doc[page_idx]

        # Calculate bounding rectangle from 4-point bbox
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = x2 + padding
        y2 = y2 + padding

        # Create clip rectangle
        clip_rect = fitz.Rect(x1, y1, x2, y2)

        # Render the clipped region
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, clip=clip_rect)

        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Convert to PNG bytes
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        doc.close()

        return img_bytes

    except Exception as e:
        logger.error(f"Region extraction failed: {e}")
        return None


def extract_region_to_base64(
    pdf_path: str,
    page_num: int,
    bbox: List[List[float]],
    dpi: int = 150,
    padding: int = 20
) -> Optional[str]:
    """
    Extract an image region and return as base64-encoded string.

    Useful for sending to Vision API.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        bbox: Bounding box coordinates
        dpi: Resolution for rendering
        padding: Pixels to add around the bbox

    Returns:
        Base64-encoded PNG string, or None on failure
    """
    import base64

    img_bytes = extract_region(pdf_path, page_num, bbox, dpi, padding)

    if img_bytes:
        return base64.standard_b64encode(img_bytes).decode('utf-8')

    return None


def expand_bbox(
    bbox: List[List[float]],
    context_factor: float = 1.5
) -> List[List[float]]:
    """
    Expand a bounding box to include surrounding context.

    Args:
        bbox: Original bounding box
        context_factor: How much to expand (1.5 = 50% larger)

    Returns:
        Expanded bounding box
    """
    # Calculate center and dimensions
    x_coords = [pt[0] for pt in bbox]
    y_coords = [pt[1] for pt in bbox]

    x_center = sum(x_coords) / 4
    y_center = sum(y_coords) / 4
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    # Expand dimensions
    new_width = width * context_factor
    new_height = height * context_factor

    # Calculate new corners
    x1 = x_center - new_width / 2
    y1 = y_center - new_height / 2
    x2 = x_center + new_width / 2
    y2 = y_center + new_height / 2

    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def merge_nearby_bboxes(
    bboxes: List[List[List[float]]],
    threshold: float = 50
) -> List[List[List[float]]]:
    """
    Merge bounding boxes that are close together.

    Useful for grouping related text blocks into a single region.

    Args:
        bboxes: List of bounding boxes
        threshold: Maximum distance to consider boxes as nearby

    Returns:
        List of merged bounding boxes
    """
    if not bboxes:
        return []

    # Convert to rectangles for easier processing
    rects = []
    for bbox in bboxes:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        rects.append({
            'x1': min(x_coords),
            'y1': min(y_coords),
            'x2': max(x_coords),
            'y2': max(y_coords),
            'merged': False
        })

    merged = []

    for i, rect in enumerate(rects):
        if rect['merged']:
            continue

        # Find all nearby rectangles
        cluster = [rect]
        rect['merged'] = True

        for j, other in enumerate(rects[i+1:], i+1):
            if other['merged']:
                continue

            # Check if rectangles are close
            if _rects_are_close(rect, other, threshold):
                cluster.append(other)
                other['merged'] = True

        # Merge cluster into single bbox
        if cluster:
            x1 = min(r['x1'] for r in cluster)
            y1 = min(r['y1'] for r in cluster)
            x2 = max(r['x2'] for r in cluster)
            y2 = max(r['y2'] for r in cluster)
            merged.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    return merged


def _rects_are_close(r1: dict, r2: dict, threshold: float) -> bool:
    """Check if two rectangles are within threshold distance."""
    # Check horizontal distance
    h_dist = max(0, max(r1['x1'], r2['x1']) - min(r1['x2'], r2['x2']))
    # Check vertical distance
    v_dist = max(0, max(r1['y1'], r2['y1']) - min(r1['y2'], r2['y2']))

    return h_dist <= threshold and v_dist <= threshold


def get_page_image(pdf_path: str, page_num: int, dpi: int = 150) -> Optional[bytes]:
    """
    Get the full page as an image.

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        dpi: Resolution for rendering

    Returns:
        PNG image bytes of the full page, or None on failure
    """
    if not PYMUPDF_AVAILABLE or not PIL_AVAILABLE:
        logger.error("PyMuPDF and Pillow required for page image extraction")
        return None

    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return None

        doc = fitz.open(str(pdf_path))

        page_idx = page_num - 1
        if page_idx < 0 or page_idx >= len(doc):
            doc.close()
            return None

        page = doc[page_idx]

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        doc.close()

        return img_bytes

    except Exception as e:
        logger.error(f"Page image extraction failed: {e}")
        return None
