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
    1. Load extracted text from state
    2. Identify FDOT pay item patterns using regex
    3. Extract quantities and units
    4. Store parsed pay items in state

    Args:
        state: Current workflow state

    Returns:
        State updates with pay_items, project_info, or last_error
    """
    extracted_text = state.get("extracted_text")
    current_file = state.get("current_file", "")

    # Step 1: Validate input
    if not extracted_text:
        logger.error("No extracted text available")
        return {
            "last_error": "No extracted text available for parsing",
            "pay_items": []
        }

    logger.info(f"Parsing pay items from {len(extracted_text)} chars of text")

    try:
        # Import here to avoid circular imports
        from tools.analyze_takeoff import TakeoffAnalyzer

        analyzer = TakeoffAnalyzer()

        # Step 2 & 3: Extract pay items with patterns and quantities
        pay_items = analyzer.extract_pay_items(extracted_text)

        if not pay_items:
            logger.warning("No pay items found in text")
            return {
                "pay_items": [],
                "project_info": analyzer.extract_project_info(
                    extracted_text,
                    Path(current_file).name if current_file else ""
                ),
                "last_error": "No pay items found in document"
            }

        # Extract project info
        filename = Path(current_file).name if current_file else ""
        project_info = analyzer.extract_project_info(extracted_text, filename)

        logger.info(f"Found {len(pay_items)} pay items")

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
