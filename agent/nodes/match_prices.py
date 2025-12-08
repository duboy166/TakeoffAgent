"""
Node 3: Price Matching & Calculation
Matches pay items to the FL 2025 price list and calculates costs.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from ..state import TakeoffState

logger = logging.getLogger(__name__)


def match_prices_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Match pay items to price list and calculate costs.

    Steps:
    1. Load parsed pay items from state
    2. Match items to FL 2025 price list by FDOT code and size
    3. Calculate line costs (qty x unit price)
    4. Categorize drainage structures

    Args:
        state: Current workflow state

    Returns:
        State updates with priced_items, drainage_structures, or last_error
    """
    pay_items = state.get("pay_items", [])
    price_list_path = state.get("price_list_path", "")

    # Step 1: Validate input
    if not pay_items:
        logger.warning("No pay items to match")
        return {
            "priced_items": [],
            "drainage_structures": [],
            "last_error": "No pay items available for price matching"
        }

    logger.info(f"Matching {len(pay_items)} items to price list")

    try:
        # Import here to avoid circular imports
        from tools.analyze_takeoff import TakeoffAnalyzer

        # Initialize analyzer with price list
        analyzer = TakeoffAnalyzer(price_list_path if price_list_path else None)

        if price_list_path and not analyzer.price_list_loaded:
            logger.warning(f"Price list not loaded from: {price_list_path}")

        # Step 2 & 3: Match to price list and calculate costs
        priced_items = analyzer.match_all_items(pay_items)

        # Count matched items
        matched_count = sum(1 for item in priced_items if item.get("matched"))
        total_cost = sum(item.get("line_cost", 0) or 0 for item in priced_items)

        logger.info(f"Matched {matched_count}/{len(priced_items)} items, total: ${total_cost:,.2f}")

        # Step 4: Categorize drainage structures
        drainage_structures = analyzer.categorize_drainage(priced_items)

        logger.info(f"Found {len(drainage_structures)} drainage structures")

        return {
            "priced_items": priced_items,
            "drainage_structures": drainage_structures,
            "last_error": None
        }

    except ImportError as e:
        logger.error(f"Failed to import analyzer: {e}")
        return {
            "last_error": f"Analyzer module not available: {e}",
            "priced_items": pay_items,  # Return unpriced items
            "drainage_structures": []
        }
    except Exception as e:
        logger.error(f"Price matching failed: {e}")
        return {
            "last_error": f"Price matching failed: {str(e)}",
            "priced_items": pay_items,
            "drainage_structures": []
        }
