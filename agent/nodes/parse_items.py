"""
Node 2: Pay Item Parsing
Extracts pay items from the extracted text using regex patterns.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from ..state import TakeoffState

logger = logging.getLogger(__name__)


def parse_items_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Parse pay items from extracted text.

    Steps:
    1. Check if Vision extraction already provided pay_items
    2. If not, identify FDOT pay item patterns using regex
    3. Extract quantities and units
    4. Attach source locations from text_blocks when available
    5. Store parsed pay items in state

    Args:
        state: Current workflow state

    Returns:
        State updates with pay_items, project_info, or last_error
    """
    extracted_text = state.get("extracted_text")
    current_file = state.get("current_file", "")
    text_blocks = state.get("text_blocks") or []
    extraction_method = state.get("extraction_method", "")

    # Check if Vision extraction already provided items
    vision_pay_items = state.get("pay_items") or []
    vision_drainage = state.get("drainage_structures") or []

    # Step 1: Validate input
    if not extracted_text:
        logger.error("No extracted text available")
        return {
            "last_error": "No extracted text available for parsing",
            "pay_items": []
        }

    try:
        # Import here to avoid circular imports
        from tools.analyze_takeoff import TakeoffAnalyzer

        analyzer = TakeoffAnalyzer()
        filename = Path(current_file).name if current_file else ""
        project_info = analyzer.extract_project_info(extracted_text, filename)

        # If Vision extraction already found items, use them directly
        if extraction_method == "claude_vision" and (vision_pay_items or vision_drainage):
            logger.info(f"Using {len(vision_pay_items)} pay items from Vision extraction")
            logger.info(f"Using {len(vision_drainage)} drainage structures from Vision extraction")

            return {
                "pay_items": vision_pay_items,
                "drainage_structures": vision_drainage,
                "project_info": project_info,
                "last_error": None
            }

        # Standard path: Parse items from text using regex patterns
        logger.info(f"Parsing pay items from {len(extracted_text)} chars of text")

        # Step 2 & 3: Extract pay items with patterns and quantities
        # Pass text_blocks for location tracking
        pay_items = analyzer.extract_pay_items(extracted_text, text_blocks=text_blocks)

        if not pay_items:
            logger.warning("No pay items found in text")
            return {
                "pay_items": [],
                "project_info": project_info,
                "last_error": "No pay items found in document"
            }

        # Count items with locations
        items_with_location = sum(1 for item in pay_items if item.get("source_location"))
        logger.info(f"Found {len(pay_items)} pay items ({items_with_location} with source locations)")

        # Step 4: Return parsed items
        return {
            "pay_items": pay_items,
            "project_info": project_info,
            "last_error": None
        }

    except ImportError as e:
        logger.error(f"Failed to import analyzer: {e}")
        return {
            "last_error": f"Analyzer module not available: {e}",
            "pay_items": []
        }
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return {
            "last_error": f"Parsing failed: {str(e)}",
            "pay_items": []
        }
