"""
AI-Powered Price Matching Node
Uses Claude to match pay items that couldn't be matched by regex patterns.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional

from ..state import TakeoffState

logger = logging.getLogger(__name__)


def _get_api_key(state: TakeoffState) -> Optional[str]:
    """Get API key from state first, then fall back to environment variable."""
    return state.get("vision_api_key") or os.getenv("ANTHROPIC_API_KEY")


def _get_ai_model() -> str:
    """Get AI model from environment variable with sensible default."""
    return os.getenv("ANTHROPIC_AI_MODEL", "claude-3-5-haiku-20241022")

# Price catalog summary for AI matching
# This is a condensed version of the FL 2025 catalog for prompt context
CATALOG_SUMMARY = """
Florida 2025 Drainage Price Catalog Categories:

PIPE CULVERTS (430-175-XXX):
- RCP (Reinforced Concrete Pipe): Class III, IV, V
- Sizes: 12", 15", 18", 24", 30", 36", 42", 48", 54", 60", 66", 72", 84", 96"
- Unit: LF (Linear Feet)
- Elliptical sizes: 14"x23", 19"x30", 24"x38", 29"x45", 34"x53", 38"x60", 43"x68", 48"x76"

ENDWALLS (430-030-XXX, 430-040-XXX):
- Straight Concrete Endwalls: Single, Double, Triple, Quad barrel
- Winged Endwalls: 45 degree
- U-Type Endwalls: With slope ratios 2:1, 3:1, 4:1, 6:1
- Sizes match pipe sizes (12" through 96")
- Unit: EA (Each)

MITERED END SECTIONS - MES (430-982-XXX):
- Concrete MES with 4:1 slope
- Galvanized Steel MES: Single/Double/Triple Run, With/No Frame
- Sizes match pipe sizes
- Unit: EA

FLARED END SECTIONS:
- For round and elliptical pipes
- Unit: EA

PIPE CRADLES:
- Support structures for pipes
- Sizes match pipe sizes
- Unit: EA

INLETS (425-1-XXX):
- Type D, Type E, Type P, Ditch Bottom
- Unit: EA

MANHOLES (425-2-XXX):
- Type 7, Type P-7
- Unit: EA
"""


def ai_match_unmatched_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Use AI to match pay items that weren't matched by rule-based matching.

    This node runs AFTER match_prices and attempts to match any remaining
    unmatched items using Claude Haiku for cost efficiency.

    Args:
        state: Current workflow state

    Returns:
        State updates with ai_matched_items, match_explanations
    """
    priced_items = state.get("priced_items") or []

    if not priced_items:
        logger.info("No priced items to check for AI matching")
        return {
            "ai_matched_items": [],
            "match_explanations": {}
        }

    # Find unmatched items
    unmatched = [item for item in priced_items if not item.get("matched")]

    if not unmatched:
        logger.info("All items already matched, skipping AI matching")
        return {
            "ai_matched_items": [],
            "match_explanations": {}
        }

    # Check if we have API access (check state first, then env var)
    api_key = _get_api_key(state)
    if not api_key:
        logger.info("No API key available, skipping AI matching")
        return {
            "ai_matched_items": [],
            "match_explanations": {}
        }

    logger.info(f"Attempting AI matching for {len(unmatched)} unmatched items")

    try:
        ai_matches, explanations = _call_ai_matching(unmatched, api_key)

        # Update priced_items with AI matches
        updated_items = []
        for item in priced_items:
            item_desc = item.get("description", "")
            if item_desc in ai_matches:
                match_info = ai_matches[item_desc]
                item = item.copy()
                item["matched"] = True
                item["match_source"] = "ai"
                item["matched_code"] = match_info.get("matched_code")
                item["ai_confidence"] = match_info.get("confidence", 0.0)
                item["ai_match_reason"] = match_info.get("reason", "")
                updated_items.append(item)
            else:
                updated_items.append(item)

        logger.info(f"AI matched {len(ai_matches)} items")

        return {
            "priced_items": updated_items,
            "ai_matched_items": list(ai_matches.values()),
            "match_explanations": explanations
        }

    except Exception as e:
        logger.error(f"AI matching failed: {e}")
        return {
            "ai_matched_items": [],
            "match_explanations": {},
            "last_error": f"AI matching failed: {str(e)}"
        }


def _call_ai_matching(unmatched: List[Dict], api_key: str) -> tuple:
    """
    Call Claude API to match unmatched items.

    Args:
        unmatched: List of unmatched pay item dicts
        api_key: Anthropic API key

    Returns:
        Tuple of (matches_dict, explanations_dict)
    """
    try:
        import anthropic
    except ImportError:
        logger.error("anthropic package not installed")
        return {}, {}

    client = anthropic.Anthropic(api_key=api_key)

    # Build items list for prompt
    items_text = json.dumps([{
        "description": item.get("description", ""),
        "quantity": item.get("quantity", 0),
        "unit": item.get("unit", ""),
        "pay_item_no": item.get("pay_item_no", ""),
        "source": item.get("source", "")
    } for item in unmatched], indent=2)

    prompt = f"""You are matching construction pay items to the Florida 2025 drainage price catalog.

{CATALOG_SUMMARY}

Here are unmatched items that need matching:
{items_text}

For each item, find the best catalog match. Consider:
1. SIZE: Exact or nearest available size
2. MATERIAL: RCP, PVC, HDPE, CMP, etc.
3. PRODUCT TYPE: Pipe, endwall, MES, inlet, manhole, etc.
4. CONFIGURATION: Single/Double/Triple barrel, slope ratios, etc.

Return ONLY valid JSON (no markdown, no explanation before/after):
{{
  "matches": [
    {{
      "original_description": "the item description",
      "matched_code": "430-XXX-XXX",
      "matched_description": "catalog description",
      "confidence": 0.85,
      "reason": "brief explanation of why this match"
    }}
  ]
}}

If an item cannot be matched, omit it from the matches array.
Only include matches with confidence >= 0.5.
"""

    try:
        response = client.messages.create(
            model=_get_ai_model(),
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            timeout=60.0
        )

        # Safety check for empty response
        if not response.content or len(response.content) == 0:
            logger.warning("AI matching returned empty response")
            return []

        response_text = response.content[0].text.strip()

        # Parse JSON response
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Show more context for debugging (BUG-036 fix)
                truncated = response_text[:500] + "..." if len(response_text) > 500 else response_text
                logger.error(f"Could not parse AI response (len={len(response_text)}): {truncated}")
                return {}, {}

        matches = {}
        explanations = {}

        for match in (result.get("matches") or []):
            desc = match.get("original_description", "")
            if desc and match.get("confidence", 0) >= 0.5:
                matches[desc] = {
                    "original_description": desc,
                    "matched_code": match.get("matched_code", ""),
                    "matched_description": match.get("matched_description", ""),
                    "confidence": match.get("confidence", 0),
                    "reason": match.get("reason", "")
                }
                explanations[desc] = match.get("reason", "")

        return matches, explanations

    except Exception as e:
        logger.error(f"AI API call failed: {e}")
        return {}, {}


def _load_catalog_summary() -> str:
    """
    Load a summarized version of the price catalog.

    Returns a condensed catalog suitable for prompt context.
    """
    return CATALOG_SUMMARY
