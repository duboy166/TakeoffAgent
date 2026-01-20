"""
Hybrid Mode Router
Conditional routing logic for hybrid OCR + Vision extraction.
"""

import os
import logging
from typing import Literal

from ..state import TakeoffState

logger = logging.getLogger(__name__)


def route_after_ocr(state: TakeoffState) -> Literal["selective_vision", "parse_items"]:
    """
    Route after OCR extraction in hybrid mode.

    Decision logic:
    - If pages are flagged for Vision AND API key is available: selective_vision
    - Otherwise: continue directly to parse_items

    Args:
        state: Current workflow state

    Returns:
        Next node: "selective_vision" or "parse_items"
    """
    extraction_mode = state.get("extraction_mode", "ocr_only")
    pages_flagged = state.get("pages_flagged_for_vision", [])
    last_error = state.get("last_error")

    # If there was an extraction error, go directly to parse (which will handle it)
    if last_error:
        logger.debug("Extraction had errors, routing directly to parse")
        return "parse_items"

    # Only route to selective vision in hybrid mode
    if extraction_mode != "hybrid":
        logger.debug(f"Not in hybrid mode ({extraction_mode}), routing to parse")
        return "parse_items"

    # Check if we have flagged pages
    if not pages_flagged:
        logger.info("Hybrid mode: No pages flagged for Vision, routing to parse")
        return "parse_items"

    # Check for API key availability
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "Hybrid mode: Pages flagged for Vision but ANTHROPIC_API_KEY not set. "
            "Using OCR-only results."
        )
        return "parse_items"

    # Route to selective vision
    logger.info(
        f"Hybrid mode: Routing to selective_vision for {len(pages_flagged)} flagged pages"
    )
    return "selective_vision"
