"""
Node: Selective Vision Extraction (Hybrid Mode)
Runs Claude Vision API only on pages flagged as low-confidence by OCR.

This node is part of the hybrid extraction workflow that reduces API costs
by 85%+ while maintaining quality. It only processes pages where OCR struggled.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List

from ..state import TakeoffState

logger = logging.getLogger(__name__)


def selective_vision_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Run Vision extraction on flagged low-confidence pages only.

    This node:
    1. Reads pages_flagged_for_vision from state
    2. Respects vision_page_budget to control costs
    3. Extracts only those page images from PDF
    4. Runs VisionExtractor on selected pages
    5. Merges Vision results with existing OCR text

    Args:
        state: Current workflow state with pages_flagged_for_vision

    Returns:
        State updates with merged extracted_text and updated extraction_method
    """
    current_file = state.get("current_file")
    pages_flagged = state.get("pages_flagged_for_vision", [])
    vision_page_budget = state.get("vision_page_budget", 5)
    dpi = state.get("dpi", 200)
    pages_ocr_results = state.get("pages_ocr_results", [])
    existing_text = state.get("extracted_text", "")

    if not current_file:
        return {
            "last_error": "No file specified for selective vision",
            "extraction_method": "error"
        }

    # Check if we should skip vision extraction
    if not pages_flagged:
        logger.info("No pages flagged for Vision - using OCR results only")
        return {
            "extraction_method": "paddleocr",
            "last_error": None
        }

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("No ANTHROPIC_API_KEY found - skipping Vision extraction")
        return {
            "extraction_method": "paddleocr",
            "last_error": "Vision API key not configured (using OCR only)"
        }

    pdf_path = Path(current_file)
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return {
            "last_error": f"File not found: {pdf_path}",
            "extraction_method": "error"
        }

    # Apply budget limit - process most problematic pages first
    # Sort by confidence (lowest first) using OCR results
    if len(pages_flagged) > vision_page_budget:
        # Create a confidence map from OCR results
        confidence_map = {p["page_num"]: p["confidence"] for p in pages_ocr_results}

        # Sort flagged pages by confidence (lowest first)
        pages_flagged_sorted = sorted(
            pages_flagged,
            key=lambda p: confidence_map.get(p, 0.0)
        )

        # Take only the budget amount
        pages_to_process = pages_flagged_sorted[:vision_page_budget]
        pages_skipped = pages_flagged_sorted[vision_page_budget:]

        logger.warning(
            f"Vision page budget ({vision_page_budget}) exceeded. "
            f"Processing {len(pages_to_process)} pages, skipping {len(pages_skipped)}"
        )
    else:
        pages_to_process = pages_flagged

    logger.info(
        f"Running selective Vision on {len(pages_to_process)}/{state.get('page_count', 0)} pages: {pages_to_process}"
    )

    try:
        from tools.vision_extractor import VisionExtractor

        extractor = VisionExtractor()

        # Extract only specific pages using the new method
        vision_results = extractor.extract_specific_pages(
            str(pdf_path),
            page_numbers=pages_to_process,
            dpi=max(dpi, 150)  # Ensure minimum DPI for vision
        )

        if vision_results.errors:
            logger.warning(f"Vision extraction warnings: {vision_results.errors}")

        # Merge Vision results with OCR text
        merged_text, vision_items, vision_structures = _merge_results(
            existing_text=existing_text,
            pages_ocr_results=pages_ocr_results,
            vision_results=vision_results,
            pages_processed=pages_to_process
        )

        logger.info(
            f"Vision extracted {len(vision_items)} pay items, "
            f"{len(vision_structures)} structures from {len(pages_to_process)} pages "
            f"(tokens: {vision_results.total_tokens})"
        )

        return {
            "extracted_text": merged_text,
            "extraction_method": "hybrid_ocr_vision",
            "last_error": None,
            # Pass pre-extracted items if Vision found any
            "pay_items": vision_items if vision_items else None,
            "drainage_structures": vision_structures if vision_structures else None,
        }

    except ImportError as e:
        logger.error(f"Failed to import vision extractor: {e}")
        return {
            "extraction_method": "paddleocr",
            "last_error": f"Vision extractor not available: {e}"
        }
    except ValueError as e:
        # Usually missing API key
        logger.error(f"Vision extraction config error: {e}")
        return {
            "extraction_method": "paddleocr",
            "last_error": f"Vision config error: {e}"
        }
    except Exception as e:
        logger.error(f"Selective vision extraction failed: {e}")
        return {
            "extraction_method": "paddleocr",
            "last_error": f"Vision extraction failed: {e}"
        }


def _merge_results(
    existing_text: str,
    pages_ocr_results: List[Dict],
    vision_results,
    pages_processed: List[int]
) -> tuple:
    """
    Merge Vision results with OCR text.

    For pages processed by Vision, replace OCR text with Vision text.
    For other pages, keep OCR text as-is.

    Args:
        existing_text: Full OCR extracted text
        pages_ocr_results: Per-page OCR results with confidence
        vision_results: VisionExtractedDocument with page results
        pages_processed: List of page numbers that were processed by Vision

    Returns:
        Tuple of (merged_text, all_pay_items, all_drainage_structures)
    """
    # Build a map of page number -> vision page result
    vision_page_map = {page.page_num: page for page in vision_results.pages}

    # Split existing text by page markers
    # Format: [Page X]\ntext\n\n--- Page Break ---\n\n
    page_sections = existing_text.split("--- Page Break ---")

    merged_sections = []
    all_pay_items = []
    all_drainage_structures = []

    for i, section in enumerate(page_sections):
        page_num = i + 1  # 1-indexed

        if page_num in pages_processed and page_num in vision_page_map:
            # Use Vision result for this page
            vision_page = vision_page_map[page_num]

            # Format vision page text
            vision_text = f"[Page {page_num}] (Vision)\n"
            vision_text += vision_page.page_summary + "\n\n"

            # Add pay items as text
            for item in vision_page.pay_items:
                pay_no = item.get("pay_item_no", "")
                desc = item.get("description", "")
                unit = item.get("unit", "")
                qty = item.get("quantity", "")
                vision_text += f"{pay_no} {desc} {unit} {qty}\n"

            merged_sections.append(vision_text)

            # Collect pay items and structures from vision
            all_pay_items.extend(vision_page.pay_items)
            all_drainage_structures.extend(vision_page.drainage_structures)
        else:
            # Keep OCR result for this page
            merged_sections.append(section.strip())

    merged_text = "\n\n--- Page Break ---\n\n".join(merged_sections)

    return merged_text, all_pay_items, all_drainage_structures
