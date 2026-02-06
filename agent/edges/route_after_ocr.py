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

    API key sources (in order of priority):
    1. vision_api_key from state (passed from GUI settings)
    2. Environment variable based on vision_provider

    Args:
        state: Current workflow state

    Returns:
        Next node: "selective_vision" or "parse_items"
    """
    extraction_mode = state.get("extraction_mode", "ocr_only")
    pages_flagged = state.get("pages_flagged_for_vision") or []
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
    # Priority: 1) state vision_api_key, 2) env var based on provider
    api_key = state.get("vision_api_key")

    if not api_key:
        # Fall back to environment variable based on provider
        provider = state.get("vision_provider", "anthropic")
        env_var_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY"
        }
        env_var = env_var_map.get(provider, "ANTHROPIC_API_KEY")
        api_key = os.getenv(env_var)

    if not api_key:
        provider = state.get("vision_provider", "anthropic")
        logger.warning(
            f"Hybrid mode: Pages flagged for Vision but no API key available for {provider}. "
            "Using OCR-only results."
        )
        return "parse_items"

    # Route to selective vision
    logger.info(
        f"Hybrid mode: Routing to selective_vision for {len(pages_flagged)} flagged pages"
    )
    return "selective_vision"
