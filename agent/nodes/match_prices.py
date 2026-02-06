"""
Node 3: Price Matching & Calculation
Matches pay items to the FL 2025 price list and calculates costs.

Includes price sanity checks via validation_gates:
- Line costs > $1,000,000 flagged
- Unit prices outside typical ranges flagged
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

from ..state import TakeoffState

# Import validation gates for price checks
try:
    from tools.validation_gates import validate_price, ValidationWarning
except ImportError:
    validate_price = None
    ValidationWarning = None

logger = logging.getLogger(__name__)


def match_prices_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Match pay items to price list and calculate costs.

    Steps:
    1. Load parsed pay items from state
    2. Match items to FL 2025 price list by FDOT code and size
    3. Calculate line costs (qty x unit price)
    4. Categorize drainage structures (preserve Vision-extracted if available)

    Args:
        state: Current workflow state

    Returns:
        State updates with priced_items, drainage_structures, or last_error
    """
    pay_items = state.get("pay_items") or []
    price_list_path = state.get("price_list_path", "")

    # Preserve Vision-extracted drainage structures if available
    existing_drainage = state.get("drainage_structures") or []

    # Step 1: Validate input
    if not pay_items:
        logger.warning("No pay items to match")
        # Preserve existing drainage structures from Vision extraction
        return {
            "priced_items": [],
            "drainage_structures": existing_drainage,
            "last_error": "No pay items available for price matching" if not existing_drainage else None
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
        
        # Step 2.5: Price sanity checks
        price_warnings = []
        if validate_price:
            for idx, item in enumerate(priced_items):
                warnings = validate_price(
                    unit_price=item.get("unit_price"),
                    line_cost=item.get("line_cost"),
                    unit=item.get("unit", ""),
                    quantity=item.get("quantity", 0),
                    item_index=idx,
                    pay_item_no=item.get("pay_item_no", "")
                )
                if warnings:
                    # Attach warnings to item
                    existing_warnings = item.get("validation_warnings", [])
                    item["validation_warnings"] = existing_warnings + [w.to_dict() for w in warnings]
                    price_warnings.extend(warnings)
            
            if price_warnings:
                logger.warning(f"Price sanity check: {len(price_warnings)} warnings")

        # Step 4: Categorize drainage structures
        # If Vision extraction already provided structures, use those
        # Otherwise, categorize from priced items
        if existing_drainage:
            drainage_structures = existing_drainage
            logger.info(f"Using {len(drainage_structures)} drainage structures from Vision extraction")
        else:
            drainage_structures = analyzer.categorize_drainage(priced_items)
            logger.info(f"Categorized {len(drainage_structures)} drainage structures from pay items")

        return {
            "priced_items": priced_items,
            "drainage_structures": drainage_structures,
            "last_error": None
        }

    except ImportError as e:
        logger.error(f"Failed to import analyzer: {e}")
        # Mark items as unpriced so downstream knows costs aren't available
        unpriced_items = [{**item, "matched": False, "unit_price": 0, "line_cost": 0, "pricing_failed": True} for item in pay_items]
        return {
            "last_error": f"Analyzer module not available: {e}",
            "priced_items": unpriced_items,
            "drainage_structures": existing_drainage  # Preserve Vision structures
        }
    except Exception as e:
        logger.error(f"Price matching failed: {e}")
        # Mark items as unpriced so downstream knows costs aren't available
        unpriced_items = [{**item, "matched": False, "unit_price": 0, "line_cost": 0, "pricing_failed": True} for item in pay_items]
        return {
            "last_error": f"Price matching failed: {str(e)}",
            "priced_items": unpriced_items,
            "drainage_structures": existing_drainage  # Preserve Vision structures
        }
