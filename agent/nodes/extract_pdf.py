"""
Node 1: PDF Text Extraction
Extracts text from PDF files using local PaddleOCR.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from ..state import TakeoffState

logger = logging.getLogger(__name__)


def extract_pdf_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Extract text from the current PDF file using PaddleOCR.

    Args:
        state: Current workflow state

    Returns:
        State updates with extracted_text, extraction_method, or last_error
    """
    current_file = state.get("current_file")
    dpi = state.get("dpi", 200)

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

    logger.info(f"Extracting text from: {pdf_path.name}")

    return _extract_with_paddleocr(pdf_path, dpi)


def _extract_with_paddleocr(pdf_path: Path, dpi: int) -> Dict[str, Any]:
    """Extract text using local PaddleOCR."""
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
                "page_count": doc.page_count
            }

        logger.info(f"Extracted {len(doc.full_text)} chars from {doc.page_count} pages using {doc.extraction_method}")

        return {
            "extracted_text": doc.full_text,
            "extraction_method": doc.extraction_method,
            "page_count": doc.page_count,
            "last_error": None
        }

    except ImportError as e:
        logger.error(f"Failed to import OCR extractor: {e}")
        return {
            "last_error": f"PaddleOCR module not available: {e}",
            "extraction_method": "error",
            "extracted_text": ""
        }
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {
            "last_error": f"Extraction failed: {str(e)}",
            "extraction_method": "error",
            "extracted_text": ""
        }
