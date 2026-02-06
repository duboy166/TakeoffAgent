"""
AI-Powered Item Validation Node
Validates extracted pay items for errors before matching.
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

# Valid pipe sizes from FL 2025 catalog
VALID_PIPE_SIZES = {12, 15, 18, 24, 30, 36, 42, 48, 54, 60, 64, 66, 72, 84, 96}

# Valid elliptical pipe sizes (rise x span)
VALID_ELLIPTICAL_SIZES = {
    (12, 18), (14, 23), (19, 30), (24, 38), (29, 45),
    (34, 53), (38, 60), (43, 68), (48, 76), (53, 83), (58, 91)
}

# Reasonable quantity ranges by unit type
QUANTITY_RANGES = {
    'LF': (1, 5000),      # Linear feet - pipes typically 10-2000
    'EA': (1, 100),       # Each - structures typically 1-50
    'SF': (1, 50000),     # Square feet
    'SY': (1, 10000),     # Square yards
    'CY': (1, 5000),      # Cubic yards
    'TON': (1, 10000),    # Tons
    'LS': (1, 1),         # Lump sum - always 1
}


def validate_items_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Validate extracted pay items for common errors.

    Checks for:
    1. Invalid pipe sizes
    2. Swapped size/quantity values
    3. Unreasonable quantities
    4. Missing required fields
    5. Duplicate detection

    If items need AI review, calls Claude Haiku for intelligent validation.

    Args:
        state: Current workflow state

    Returns:
        State updates with validation_issues, items_corrected, validation_confidence
    """
    pay_items = state.get("pay_items") or []

    if not pay_items:
        logger.info("No pay items to validate")
        return {
            "validation_issues": [],
            "items_corrected": 0,
            "validation_confidence": 1.0
        }

    # Skip validation for trivial cases
    if len(pay_items) < 3:
        logger.info(f"Only {len(pay_items)} items, skipping AI validation")
        return {
            "validation_issues": [],
            "items_corrected": 0,
            "validation_confidence": 1.0
        }

    # Perform rule-based validation first
    issues, corrected_items, corrections_made = _validate_with_rules(pay_items)

    # If there are potential issues, try AI validation
    api_key = _get_api_key(state)
    if issues and api_key:
        ai_issues, ai_corrections = _validate_with_ai(corrected_items, issues, api_key)
        issues.extend(ai_issues)
        corrections_made += ai_corrections

    # Calculate overall confidence
    if pay_items:
        items_with_issues = len([i for i in issues if i.get("severity") in ("high", "medium")])
        confidence = max(0.0, 1.0 - (items_with_issues / len(pay_items)))
    else:
        confidence = 1.0

    logger.info(f"Validation complete: {len(issues)} issues, {corrections_made} corrections, confidence: {confidence:.2f}")

    return {
        "pay_items": corrected_items,
        "validation_issues": issues,
        "items_corrected": corrections_made,
        "validation_confidence": confidence
    }


def _validate_with_rules(pay_items: List[Dict]) -> tuple:
    """
    Apply rule-based validation to pay items.

    Args:
        pay_items: List of pay item dicts

    Returns:
        Tuple of (issues_list, corrected_items, corrections_count)
    """
    issues = []
    corrected_items = []
    corrections_made = 0

    for idx, item in enumerate(pay_items):
        item = item.copy()  # Don't modify original
        desc = item.get("description", "").upper()
        qty = item.get("quantity", 0)
        unit = item.get("unit", "")

        # Check 1: Extract size from description and validate
        size = _extract_size_from_description(desc)
        if size:
            # Check for swapped size/quantity
            if size > 96 and qty in VALID_PIPE_SIZES:
                # Likely swapped
                issues.append({
                    "index": idx,
                    "severity": "high",
                    "issue": f"Size {size} likely swapped with quantity {qty}",
                    "correction": f"size={qty}, qty={size}",
                    "auto_corrected": True
                })
                # Apply correction
                item["quantity"] = size
                # Update description with corrected size
                item["description"] = desc.replace(str(size), str(qty))
                corrections_made += 1
            elif size not in VALID_PIPE_SIZES and not _is_elliptical(desc):
                issues.append({
                    "index": idx,
                    "severity": "medium",
                    "issue": f"Non-standard pipe size: {size}\"",
                    "auto_corrected": False
                })

        # Check 2: Quantity range validation
        if unit and qty:
            min_qty, max_qty = QUANTITY_RANGES.get(unit, (0, float('inf')))
            if qty < min_qty or qty > max_qty:
                issues.append({
                    "index": idx,
                    "severity": "low",
                    "issue": f"Unusual quantity {qty} {unit} (expected {min_qty}-{max_qty})",
                    "auto_corrected": False
                })

        # Check 3: Missing fields
        if not item.get("unit"):
            issues.append({
                "index": idx,
                "severity": "medium",
                "issue": "Missing unit of measure",
                "auto_corrected": False
            })

        if item.get("quantity", 0) == 0 and not item.get("needs_verification"):
            issues.append({
                "index": idx,
                "severity": "medium",
                "issue": "Zero quantity",
                "auto_corrected": False
            })

        corrected_items.append(item)

    # Check 4: Duplicate detection
    seen_items = {}
    for idx, item in enumerate(corrected_items):
        key = f"{item.get('description', '')}_{item.get('quantity', '')}_{item.get('unit', '')}"
        if key in seen_items:
            issues.append({
                "index": idx,
                "severity": "low",
                "issue": f"Possible duplicate of item at index {seen_items[key]}",
                "auto_corrected": False
            })
        else:
            seen_items[key] = idx

    return issues, corrected_items, corrections_made


def _validate_with_ai(items: List[Dict], existing_issues: List[Dict], api_key: str) -> tuple:
    """
    Use AI to validate items that have potential issues.

    Args:
        items: List of pay items
        existing_issues: Issues already found by rules
        api_key: Anthropic API key

    Returns:
        Tuple of (additional_issues, corrections_count)
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed, skipping AI validation")
        return [], 0

    # Only validate items with issues
    issue_indices = {i["index"] for i in existing_issues}
    items_to_validate = [
        {"index": idx, **item}
        for idx, item in enumerate(items)
        if idx in issue_indices
    ]

    if not items_to_validate:
        return [], 0

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are validating pay items extracted from a Florida drainage construction plan.

Items to validate:
{json.dumps(items_to_validate, indent=2)}

Known issues found:
{json.dumps(existing_issues, indent=2)}

For each item, verify:
1. SIZE VALIDITY: Pipe sizes should be 12-96" (round) or standard elliptical (14x23, 19x30, etc.)
2. QUANTITY REASONABLENESS:
   - LF (linear feet): typically 10-2000 for pipes
   - EA (each): typically 1-50 for structures
3. SWAPPED VALUES: If size > 96, it might be the quantity
4. DESCRIPTION ACCURACY: Does the description match what the numbers suggest?

Return ONLY valid JSON:
{{
  "additional_issues": [
    {{"index": 0, "issue": "description of issue", "severity": "high|medium|low", "suggested_correction": "..."}}
  ],
  "corrections_applied": 0
}}
"""

    try:
        response = client.messages.create(
            model=_get_ai_model(),
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
            timeout=60.0
        )

        # Safety check for empty response
        if not response.content or len(response.content) == 0:
            logger.warning("AI validation returned empty response")
            return [], 0

        response_text = response.content[0].text.strip()

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                return [], 0

        additional_issues = result.get("additional_issues") or []
        corrections = result.get("corrections_applied", 0) or 0

        return additional_issues, corrections

    except Exception as e:
        logger.error(f"AI validation failed: {e}")
        return [], 0


def _extract_size_from_description(desc: str) -> int:
    """Extract pipe size from description string."""
    import re

    # Try to find size pattern like 18" or 18 INCH
    match = re.search(r'(\d+)\s*(?:"|INCH)', desc)
    if match:
        return int(match.group(1))

    # Try elliptical pattern like 14x23 or 14"x23"
    match = re.search(r'(\d+)\s*[xX×]\s*(\d+)', desc)
    if match:
        return int(match.group(1))  # Return rise dimension

    return 0


def _is_elliptical(desc: str) -> bool:
    """Check if description refers to elliptical pipe."""
    import re
    return bool(re.search(r'(\d+)\s*[xX×]\s*(\d+)', desc)) or 'ELLIP' in desc or 'ERCP' in desc
