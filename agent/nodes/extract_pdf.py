"""
Node 1: PDF Text Extraction
Extracts text from PDF files using local PaddleOCR or Claude Vision API.

Supports three extraction modes:
- ocr_only: Use PaddleOCR (default, free, local)
- hybrid: Use OCR first, then Vision API for low-confidence pages
- vision_only: Use Claude Vision API for all pages
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from ..state import TakeoffState

logger = logging.getLogger(__name__)

# Confidence threshold for flagging pages for Vision review
OCR_CONFIDENCE_THRESHOLD = 0.65


def extract_pdf_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Extract text from the current PDF file.

    Supports three extraction modes:
    1. ocr_only (default): PaddleOCR local OCR, free, works offline
    2. hybrid: OCR first, then Vision API for low-confidence pages
    3. vision_only: Claude Vision API for all pages

    Args:
        state: Current workflow state

    Returns:
        State updates with extracted_text, extraction_method, or last_error
        In hybrid mode, also sets pages_ocr_results and pages_flagged_for_vision
    """
    current_file = state.get("current_file")
    dpi = state.get("dpi", 200)
    use_vision = state.get("use_vision", False)
    extraction_mode = state.get("extraction_mode", "ocr_only")
    recommended_extraction = state.get("recommended_extraction", None)

    if not current_file:
        return {
            "last_error": "No file specified for extraction",
            "extraction_method": "error"
        }

    pdf_path = Path(current_file)

    # Validate file exists
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return {
            "last_error": f"File not found: {pdf_path}",
            "extraction_method": "error"
        }

    if not pdf_path.suffix.lower() == '.pdf':
        logger.error(f"Not a PDF file: {pdf_path}")
        return {
            "last_error": f"Not a PDF file: {pdf_path}",
            "extraction_method": "error"
        }

    logger.info(f"Extracting text from: {pdf_path.name} (mode={extraction_mode})")

    # Determine extraction method:
    # 1. extraction_mode="vision_only" OR use_vision flag → full Vision extraction
    # 2. extraction_mode="hybrid" → OCR first, flag low-confidence pages for Vision
    # 3. extraction_mode="ocr_only" → PaddleOCR only
    # 4. recommended_extraction from analyze_document can override to vision

    should_use_vision_only = (
        extraction_mode == "vision_only" or
        use_vision or
        (recommended_extraction == "vision" and extraction_mode != "hybrid")
    )

    if should_use_vision_only:
        logger.info(f"Using full Vision extraction (mode={extraction_mode}, explicit={use_vision})")
        return _extract_with_vision(pdf_path, dpi)
    else:
        # OCR extraction (for both 'ocr_only' and 'hybrid' modes)
        # In hybrid mode, we also track per-page confidence for selective Vision
        is_hybrid = extraction_mode == "hybrid"
        logger.info(f"Using PaddleOCR extraction (hybrid={is_hybrid})")
        return _extract_with_paddleocr(pdf_path, dpi, track_per_page=is_hybrid)


def _extract_with_paddleocr(pdf_path: Path, dpi: int, track_per_page: bool = False) -> Dict[str, Any]:
    """
    Extract text using local PaddleOCR.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for OCR
        track_per_page: If True (hybrid mode), track per-page confidence
                        and flag low-confidence pages for Vision API

    Returns:
        State updates including extracted_text, and optionally
        pages_ocr_results and pages_flagged_for_vision for hybrid mode
    """
    try:
        from tools.ocr_extractor import OCRExtractor

        logger.info("Using local PaddleOCR for extraction")
        extractor = OCRExtractor()
        doc = extractor.extract_from_pdf(str(pdf_path), dpi=dpi)

        if doc.errors:
            logger.warning(f"Extraction warnings: {doc.errors}")

        if not doc.full_text.strip():
            return {
                "last_error": "No text extracted from PDF",
                "extraction_method": doc.extraction_method,
                "extracted_text": "",
                "page_count": doc.page_count,
                "text_blocks": [],
                "pages_ocr_results": [],
                "pages_flagged_for_vision": []
            }

        # Convert TextBlock objects to dicts for state serialization
        text_blocks = []
        for block in doc.all_text_blocks:
            text_blocks.append({
                "text": block.text,
                "confidence": block.confidence,
                "bbox": block.bbox,
                "page": block.page,
                "y_center": block.y_center,
                "x_center": block.x_center
            })

        logger.info(f"Extracted {len(doc.full_text)} chars from {doc.page_count} pages using {doc.extraction_method} ({len(text_blocks)} blocks)")

        result = {
            "extracted_text": doc.full_text,
            "extraction_method": doc.extraction_method,
            "page_count": doc.page_count,
            "text_blocks": text_blocks,
            "last_error": None
        }

        # For hybrid mode: track per-page results and flag low-confidence pages
        if track_per_page:
            pages_ocr_results, pages_flagged = _analyze_ocr_pages(doc)
            result["pages_ocr_results"] = pages_ocr_results
            result["pages_flagged_for_vision"] = pages_flagged

            if pages_flagged:
                logger.info(f"Hybrid mode: Flagged {len(pages_flagged)} pages for Vision: {pages_flagged}")
            else:
                logger.info("Hybrid mode: All pages have good OCR confidence, no Vision needed")

        return result

    except ImportError as e:
        logger.error(f"Failed to import OCR extractor: {e}")
        return {
            "last_error": f"PaddleOCR module not available: {e}",
            "extraction_method": "error",
            "extracted_text": "",
            "text_blocks": []
        }
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {
            "last_error": f"Extraction failed: {str(e)}",
            "extraction_method": "error",
            "extracted_text": "",
            "text_blocks": []
        }


def _analyze_ocr_pages(doc) -> tuple:
    """
    Analyze OCR results per page and flag low-confidence pages.

    A page is flagged for Vision if:
    1. Average OCR confidence < 0.65
    2. Very few text blocks detected (< 5)
    3. Page has minimal text content (< 50 chars)

    Args:
        doc: ExtractedDocument from OCR

    Returns:
        Tuple of (pages_ocr_results list, pages_flagged_for_vision list)
    """
    pages_ocr_results = []
    pages_flagged = []

    for page in doc.pages:
        page_num = page.page_num
        confidence = page.confidence
        text_length = len(page.text.strip()) if page.text else 0
        block_count = len(page.text_blocks) if page.text_blocks else 0

        page_result = {
            "page_num": page_num,
            "confidence": confidence,
            "text_length": text_length,
            "block_count": block_count,
            "text_preview": page.text[:200] if page.text else ""
        }
        pages_ocr_results.append(page_result)

        # Determine if this page needs Vision review
        needs_vision = False
        reasons = []

        # Check confidence threshold
        if confidence < OCR_CONFIDENCE_THRESHOLD:
            needs_vision = True
            reasons.append(f"low_confidence ({confidence:.2f})")

        # Check for very few text blocks (suggests OCR struggled)
        if block_count < 5 and text_length > 0:
            needs_vision = True
            reasons.append(f"few_blocks ({block_count})")

        # Check for minimal text (might be image-heavy page)
        if text_length < 50:
            needs_vision = True
            reasons.append(f"minimal_text ({text_length} chars)")

        if needs_vision:
            pages_flagged.append(page_num)
            logger.debug(f"Page {page_num} flagged for Vision: {', '.join(reasons)}")

    return pages_ocr_results, pages_flagged


def _extract_with_vision(pdf_path: Path, dpi: int) -> Dict[str, Any]:
    """
    Extract pay items using Claude Vision API.

    This method sends page images to Claude Vision for intelligent extraction.
    It can understand spatial relationships, read complex tables, and follow
    annotation callouts - capabilities beyond traditional OCR.

    Returns both raw text (for compatibility) and pre-extracted pay items.
    """
    try:
        from tools.vision_extractor import VisionExtractor, vision_to_text

        logger.info("Using Claude Vision API for extraction")

        # Use higher DPI for vision to read small callout text
        vision_dpi = max(dpi, 200)

        extractor = VisionExtractor()
        doc = extractor.extract_from_pdf(str(pdf_path), dpi=vision_dpi)

        if doc.errors and not doc.all_pay_items:
            # Total failure
            return {
                "last_error": f"Vision extraction failed: {'; '.join(doc.errors)}",
                "extraction_method": "error",
                "extracted_text": "",
                "page_count": doc.page_count
            }

        # Convert vision results to text format for compatibility
        # This allows the parse_items node to work with vision output
        extracted_text = vision_to_text(doc)

        # Also pass the pre-extracted items directly
        # The parse_items node can use these if available
        logger.info(
            f"Vision extracted {len(doc.all_pay_items)} pay items, "
            f"{len(doc.all_drainage_structures)} structures from {doc.page_count} pages "
            f"(tokens: {doc.total_tokens})"
        )

        return {
            "extracted_text": extracted_text,
            "extraction_method": "claude_vision",
            "page_count": doc.page_count,
            "last_error": None,
            # Pass pre-extracted items for parse_items to use
            "pay_items": doc.all_pay_items,
            "drainage_structures": doc.all_drainage_structures,
            # Vision API doesn't provide bounding boxes like OCR
            "text_blocks": [],
        }

    except ImportError as e:
        logger.error(f"Failed to import vision extractor: {e}")
        return {
            "last_error": f"Vision extractor module not available: {e}",
            "extraction_method": "error",
            "extracted_text": ""
        }
    except ValueError as e:
        # Usually missing API key
        logger.error(f"Vision extraction config error: {e}")
        return {
            "last_error": f"Vision extraction config error: {str(e)}",
            "extraction_method": "error",
            "extracted_text": ""
        }
    except Exception as e:
        logger.error(f"Vision extraction failed: {e}")
        return {
            "last_error": f"Vision extraction failed: {str(e)}",
            "extraction_method": "error",
            "extracted_text": ""
        }
