"""
Low-Confidence Item Verification Node
Uses AI to verify items with low confidence or ambiguous extraction.
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

# Confidence threshold below which items should be verified
LOW_CONFIDENCE_THRESHOLD = 0.6

# Maximum items to send for verification (cost control)
MAX_ITEMS_TO_VERIFY = 10


def verify_low_confidence_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Verify low-confidence items using AI.

    This node identifies items with:
    - Low OCR confidence scores
    - Confidence level marked as 'low'
    - needs_verification flag set

    For each low-confidence item with source location, we could send
    the image region to Vision API. For now, we use text-based verification.

    Args:
        state: Current workflow state

    Returns:
        State updates with items_for_review, verification_results
    """
    pay_items = state.get("pay_items") or []

    if not pay_items:
        logger.info("No pay items to verify")
        return {
            "items_for_review": [],
            "verification_results": []
        }

    # Find items needing verification
    items_to_verify = _identify_low_confidence_items(pay_items)

    if not items_to_verify:
        logger.info("No low-confidence items to verify")
        return {
            "items_for_review": [],
            "verification_results": []
        }

    logger.info(f"Found {len(items_to_verify)} items needing verification")

    # Check if we have API access (check state first, then env var)
    api_key = _get_api_key(state)
    if not api_key:
        # No API, just flag for human review
        items_for_review = _prepare_for_human_review(items_to_verify)
        return {
            "items_for_review": items_for_review,
            "verification_results": []
        }

    # Limit items for cost control
    items_to_verify = items_to_verify[:MAX_ITEMS_TO_VERIFY]

    # Perform AI verification
    verification_results = _verify_with_ai(items_to_verify, api_key)

    # Apply verification results
    updated_items, items_for_review = _apply_verification_results(
        pay_items, verification_results
    )

    return {
        "pay_items": updated_items,
        "items_for_review": items_for_review,
        "verification_results": verification_results
    }


def _identify_low_confidence_items(pay_items: List[Dict]) -> List[Dict]:
    """
    Identify items that need verification.

    Args:
        pay_items: List of pay item dicts

    Returns:
        List of items needing verification with their indices
    """
    items_to_verify = []

    for idx, item in enumerate(pay_items):
        needs_verification = False
        reason = []

        # Check confidence field
        if item.get("confidence") == "low":
            needs_verification = True
            reason.append("low confidence extraction")

        # Check OCR confidence from source_location
        source_loc = item.get("source_location", {})
        ocr_conf = source_loc.get("ocr_confidence", 1.0)
        if ocr_conf < LOW_CONFIDENCE_THRESHOLD:
            needs_verification = True
            reason.append(f"low OCR confidence ({ocr_conf:.2f})")

        # Check needs_verification flag
        if item.get("needs_verification"):
            needs_verification = True
            reason.append("flagged for verification")

        # Check for zero quantity (might be parsing error)
        if item.get("quantity", 0) == 0:
            needs_verification = True
            reason.append("zero quantity")

        if needs_verification:
            items_to_verify.append({
                "index": idx,
                "item": item,
                "reason": ", ".join(reason)
            })

    return items_to_verify


def _prepare_for_human_review(items_to_verify: List[Dict]) -> List[Dict]:
    """
    Prepare items for human review queue.

    Args:
        items_to_verify: List of items needing verification

    Returns:
        List of items formatted for human review
    """
    review_items = []

    for entry in items_to_verify:
        item = entry["item"]
        source_loc = item.get("source_location", {})

        review_item = {
            "description": item.get("description", ""),
            "quantity": item.get("quantity", 0),
            "unit": item.get("unit", ""),
            "confidence": item.get("confidence", "unknown"),
            "reason": entry["reason"],
            "page": source_loc.get("page", 0),
            "region_hint": _get_region_hint(source_loc),
            "text_context": source_loc.get("text_context", ""),
            "needs_human_review": True
        }
        review_items.append(review_item)

    return review_items


def _get_region_hint(source_loc: Dict) -> str:
    """
    Get a human-readable region hint from bounding box.

    Args:
        source_loc: Source location dict with bbox

    Returns:
        Human-readable region description
    """
    bbox = source_loc.get("bbox", [])
    if not bbox:
        return "Unknown location"

    # bbox is typically [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    try:
        if len(bbox) >= 2:
            x_center = (bbox[0][0] + bbox[2][0]) / 2
            y_center = (bbox[0][1] + bbox[2][1]) / 2

            # Assuming typical page dimensions (~800x1100 pixels at 100 DPI)
            h_pos = "left" if x_center < 400 else "center" if x_center < 600 else "right"
            v_pos = "top" if y_center < 350 else "middle" if y_center < 750 else "bottom"

            return f"{v_pos.capitalize()}-{h_pos} region"
    except (IndexError, TypeError):
        pass

    return "Unknown location"


def _verify_with_ai(items_to_verify: List[Dict], api_key: str) -> List[Dict]:
    """
    Use AI to verify low-confidence items.

    Args:
        items_to_verify: List of items needing verification
        api_key: Anthropic API key

    Returns:
        List of verification results
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed, skipping AI verification")
        return []

    client = anthropic.Anthropic(api_key=api_key)

    # Format items for prompt
    items_for_prompt = []
    for entry in items_to_verify:
        item = entry["item"]
        source_loc = item.get("source_location", {})
        items_for_prompt.append({
            "index": entry["index"],
            "description": item.get("description", ""),
            "quantity": item.get("quantity", 0),
            "unit": item.get("unit", ""),
            "text_context": source_loc.get("text_context", ""),
            "reason_for_review": entry["reason"]
        })

    prompt = f"""You are verifying extracted pay items from a Florida construction plan.

These items have low confidence or need verification:
{json.dumps(items_for_prompt, indent=2)}

For each item:
1. Check if the description makes sense for a drainage construction item
2. Verify the quantity seems reasonable for the item type
3. Check if the text_context supports the extracted data
4. Look for parsing errors (e.g., size/quantity swapped)

Return ONLY valid JSON:
{{
  "verifications": [
    {{
      "index": 0,
      "correct": true,
      "corrected_description": null,
      "corrected_quantity": null,
      "confidence": 0.8,
      "notes": "explanation"
    }}
  ]
}}

If an item looks incorrect, provide corrections.
If you can't determine correctness, set confidence low and explain in notes.
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
            logger.warning("AI verification returned empty response")
            return []

        response_text = response.content[0].text.strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                return []

        return result.get("verifications", [])

    except Exception as e:
        logger.error(f"AI verification failed: {e}")
        return []


def _apply_verification_results(
    pay_items: List[Dict],
    verification_results: List[Dict]
) -> tuple:
    """
    Apply AI verification results to pay items.

    Args:
        pay_items: Original pay items
        verification_results: Results from AI verification

    Returns:
        Tuple of (updated_items, items_for_review)
    """
    updated_items = [item.copy() for item in pay_items]
    items_for_review = []

    # Index results by item index
    results_by_idx = {r["index"]: r for r in verification_results}

    for idx, item in enumerate(updated_items):
        if idx in results_by_idx:
            result = results_by_idx[idx]

            # Apply corrections if provided
            if result.get("corrected_description"):
                item["description"] = result["corrected_description"]
                item["ai_corrected"] = True

            if result.get("corrected_quantity") is not None:
                item["quantity"] = result["corrected_quantity"]
                item["ai_corrected"] = True

            # Update confidence
            item["verification_confidence"] = result.get("confidence", 0.5)
            item["verification_notes"] = result.get("notes", "")

            # If still low confidence, add to review queue
            if result.get("confidence", 0) < 0.7 or not result.get("correct", True):
                source_loc = item.get("source_location", {})
                items_for_review.append({
                    "description": item.get("description", ""),
                    "quantity": item.get("quantity", 0),
                    "unit": item.get("unit", ""),
                    "confidence": result.get("confidence", 0),
                    "reason": result.get("notes", "Low confidence after verification"),
                    "page": source_loc.get("page", 0),
                    "region_hint": _get_region_hint(source_loc),
                    "needs_human_review": True
                })

    return updated_items, items_for_review
