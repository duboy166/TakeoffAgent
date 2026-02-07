"""
Product Matcher - Product-first matching for AutoWork

Searches plan text for products from the catalog, extracts quantities,
and returns matches with confidence scores.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set
from .product_catalog import Product, ProductCatalog, load_catalog


@dataclass
class Match:
    """A matched product with quantity and confidence."""
    product: Product
    quantity: float
    confidence: float  # 0.0 to 1.0
    source_text: str   # The text that was matched
    pattern_used: str  # Which pattern matched
    unit: str          # Extracted or inferred unit
    
    @property
    def extended_price(self) -> float:
        """Calculate extended price (quantity × unit price)."""
        if self.product.price == 0.0:
            return 0.0
        return self.quantity * self.product.price
    
    def __repr__(self):
        return (f"Match({self.product.id}, qty={self.quantity}, "
                f"conf={self.confidence:.2f}, unit={self.unit})")


@dataclass
class FoundItem:
    """Intermediate result from pattern search."""
    quantity: float
    confidence: float
    raw_match: str
    pattern: str
    unit: str


def find_products_in_text(
    text: str, 
    catalog: ProductCatalog,
    context_window: int = 150
) -> List[Match]:
    """
    Find all products from catalog that appear in the text.
    
    Args:
        text: The plan text to search
        catalog: Product catalog to match against
        context_window: Characters to search for quantity (before/after match)
    
    Returns:
        List of Match objects with products, quantities, and confidence scores
    """
    matches = []
    text_upper = text.upper()
    
    # Quick filter: only check products whose keywords appear
    candidates = catalog.get_candidates(text)
    
    # Track matched text spans to avoid duplicates
    matched_spans: Set[Tuple[int, int]] = set()
    
    for product in candidates:
        # Try patterns in order (more specific first)
        for i, pattern in enumerate(product.search_patterns):
            found_items = search_with_quantity(
                text, text_upper, pattern, product.unit, context_window
            )
            
            for found in found_items:
                # Check for overlap with existing matches
                # (We'll deduplicate later, but skip obvious duplicates)
                match = Match(
                    product=product,
                    quantity=found.quantity,
                    confidence=found.confidence,
                    source_text=found.raw_match,
                    pattern_used=pattern,
                    unit=found.unit
                )
                
                # Adjust confidence based on pattern specificity
                # Earlier patterns are more specific
                specificity_bonus = max(0, 0.1 - (i * 0.02))
                match.confidence = min(1.0, match.confidence + specificity_bonus)
                
                matches.append(match)
    
    # Deduplicate and consolidate matches
    return deduplicate_matches(matches)


def search_with_quantity(
    text: str,
    text_upper: str, 
    product_pattern: str, 
    expected_unit: str,
    context_window: int = 150
) -> List[FoundItem]:
    """
    Find product pattern in text and extract nearby quantity.
    
    Handles various quantity formats:
    - "18" RCP - 150 LF"
    - "150 LF 18" RCP"  
    - "PIPE 18" (RCP) .......... 150 LF"
    - Table formats with quantities in columns
    
    Returns list of FoundItem for all matches found.
    """
    results = []
    
    try:
        pattern_re = re.compile(product_pattern, re.IGNORECASE)
    except re.error:
        # Invalid regex pattern
        return results
    
    # Find all matches of the product pattern
    for product_match in pattern_re.finditer(text):
        # Get context around the match
        start = max(0, product_match.start() - context_window)
        end = min(len(text), product_match.end() + context_window)
        context = text[start:end]
        context_upper = context.upper()
        
        # Calculate position of match within context
        match_start_in_context = product_match.start() - start
        match_end_in_context = product_match.end() - start
        
        # Try to extract quantity with unit
        found = extract_quantity_with_unit(
            context, context_upper, expected_unit,
            match_start_in_context, match_end_in_context
        )
        
        if found:
            results.append(FoundItem(
                quantity=found[0],
                confidence=found[1],
                raw_match=context.strip(),
                pattern=product_pattern,
                unit=found[2]
            ))
        else:
            # No quantity found - still record match but low confidence
            # (Useful for detecting product presence)
            results.append(FoundItem(
                quantity=0.0,
                confidence=0.3,
                raw_match=context.strip(),
                pattern=product_pattern,
                unit=expected_unit
            ))
    
    return results


def extract_quantity_with_unit(
    context: str,
    context_upper: str,
    expected_unit: str,
    match_start: int,
    match_end: int
) -> Optional[Tuple[float, float, str]]:
    """
    Extract quantity from context around a product match.
    
    Prioritizes quantities on the same line as the product mention.
    
    Returns: (quantity, confidence, unit) or None
    """
    # Find the line containing the match
    line_start = context.rfind('\n', 0, match_start) + 1
    line_end = context.find('\n', match_end)
    if line_end == -1:
        line_end = len(context)
    
    same_line = context[line_start:line_end]
    same_line_upper = same_line.upper()
    
    # Adjust match positions relative to line
    match_start_in_line = match_start - line_start
    match_end_in_line = match_end - line_start
    
    # Unit patterns - match expected unit and common variants
    unit_patterns = get_unit_patterns(expected_unit)
    
    # Strategy 1: Look for quantity + unit on the SAME LINE (highest priority)
    for unit_pat, unit_name in unit_patterns:
        qty_unit_pattern = rf'(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*{unit_pat}'
        
        for match in re.finditer(qty_unit_pattern, same_line_upper):
            qty = parse_number(match.group(1))
            if is_valid_quantity(qty, expected_unit):
                return (qty, 0.95, unit_name)
    
    # Strategy 2: Look for quantity + unit pattern anywhere in context
    for unit_pat, unit_name in unit_patterns:
        qty_unit_pattern = rf'(\d+(?:,\d{{3}})*(?:\.\d+)?)\s*{unit_pat}'
        
        for match in re.finditer(qty_unit_pattern, context_upper):
            qty = parse_number(match.group(1))
            if is_valid_quantity(qty, expected_unit):
                # Check if quantity is reasonably close to product mention
                distance = min(
                    abs(match.start() - match_end),
                    abs(match.end() - match_start)
                )
                # Closer quantities get higher confidence
                if distance < 30:
                    confidence = 0.90
                elif distance < 80:
                    confidence = 0.75
                else:
                    confidence = 0.60
                
                return (qty, confidence, unit_name)
        
        # Unit followed by number (less common)
        unit_qty_pattern = rf'{unit_pat}\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)'
        for match in re.finditer(unit_qty_pattern, context_upper):
            qty = parse_number(match.group(1))
            if is_valid_quantity(qty, expected_unit):
                return (qty, 0.75, unit_name)
    
    # Strategy 3: Look for tabular format (dotted lines, equals signs)
    # Example: "18" RCP CL III .......... 450"
    tabular_pattern = rf'[\.=\-_]{{3,}}\s*(\d+(?:,\d{{3}})*(?:\.\d+)?)'
    for match in re.finditer(tabular_pattern, same_line):
        qty = parse_number(match.group(1))
        if is_valid_quantity(qty, expected_unit):
            return (qty, 0.85, expected_unit)
    
    # Strategy 4: Look for any number on same line (after product mention)
    after_match_line = same_line[match_end_in_line:]
    number_pattern = r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b'
    for match in re.finditer(number_pattern, after_match_line):
        qty = parse_number(match.group(1))
        if is_valid_quantity(qty, expected_unit):
            return (qty, 0.65, expected_unit)
    
    # Strategy 5: Look for any number in full context (lower confidence)
    after_match = context[match_end:]
    for match in re.finditer(number_pattern, after_match):
        qty = parse_number(match.group(1))
        if is_valid_quantity(qty, expected_unit):
            return (qty, 0.50, expected_unit)
    
    before_match = context[:match_start]
    for match in re.finditer(number_pattern, before_match):
        qty = parse_number(match.group(1))
        if is_valid_quantity(qty, expected_unit):
            return (qty, 0.40, expected_unit)
    
    return None


def get_unit_patterns(unit: str) -> List[Tuple[str, str]]:
    """Get regex patterns for a unit and its variants."""
    if unit == 'LF':
        return [
            (r'L\.?F\.?', 'LF'),
            (r'LIN(?:EAR)?\s*(?:FT|FEET|FOOT)', 'LF'),
            (r"FEET|FT\.?|'", 'LF'),
        ]
    elif unit == 'EA':
        return [
            (r'EA\.?', 'EA'),
            (r'EACH', 'EA'),
            (r'EA?CH', 'EA'),
            (r'PCS?\.?', 'EA'),
            (r'PIECES?', 'EA'),
            (r'UNITS?', 'EA'),
        ]
    else:
        return [(re.escape(unit), unit)]


def parse_number(s: str) -> float:
    """Parse a number string (handles commas, decimals)."""
    try:
        return float(s.replace(',', ''))
    except ValueError:
        return 0.0


def is_valid_quantity(qty: float, unit: str) -> bool:
    """
    Check if quantity is reasonable for the unit type.
    
    Filters out obviously wrong values like:
    - Sizes mistaken for quantities (18, 24, 36)
    - Page numbers or other small integers
    - Impossibly large values
    """
    if qty <= 0:
        return False
    
    if unit == 'LF':
        # Linear feet: expect 10-5000 typically
        # Reject common pipe sizes
        if qty in [12, 15, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 84, 96]:
            # Could be pipe size, not quantity
            # Only accept if fairly large (>100)
            return qty > 100
        return 1 <= qty <= 50000
    
    elif unit == 'EA':
        # Each: expect 1-100 typically
        # Reject common pipe sizes
        if qty in [12, 15, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 84, 96]:
            # More likely a size than quantity for EA items
            return qty <= 10  # Small sizes could still be valid counts
        return 1 <= qty <= 500
    
    return 1 <= qty <= 100000


def deduplicate_matches(matches: List[Match]) -> List[Match]:
    """
    Deduplicate and consolidate matches.
    
    Strategy:
    1. Group by product ID
    2. For each product, keep the highest confidence match
    3. If multiple quantities, prefer the more specific one
    """
    if not matches:
        return []
    
    # Group by product ID
    by_product: dict = {}
    for match in matches:
        pid = match.product.id
        if pid not in by_product:
            by_product[pid] = []
        by_product[pid].append(match)
    
    # Select best match per product
    result = []
    for pid, product_matches in by_product.items():
        # Sort by confidence (descending), then by quantity presence
        product_matches.sort(
            key=lambda m: (m.confidence, m.quantity > 0),
            reverse=True
        )
        
        best = product_matches[0]
        
        # If best has no quantity, try to get from second-best
        if best.quantity == 0 and len(product_matches) > 1:
            for alt in product_matches[1:]:
                if alt.quantity > 0 and alt.confidence > 0.4:
                    # Use alt's quantity but keep best's confidence context
                    best = Match(
                        product=best.product,
                        quantity=alt.quantity,
                        confidence=min(best.confidence, alt.confidence),
                        source_text=best.source_text,
                        pattern_used=best.pattern_used,
                        unit=alt.unit
                    )
                    break
        
        result.append(best)
    
    # Sort by confidence for output
    result.sort(key=lambda m: m.confidence, reverse=True)
    return result


def match_text(text: str, catalog_path: Optional[str] = None) -> List[Match]:
    """
    Convenience function: load catalog and match text.
    
    Args:
        text: Plan text to search
        catalog_path: Optional path to price list CSV
        
    Returns:
        List of Match objects
    """
    from pathlib import Path
    
    if catalog_path:
        catalog = ProductCatalog.from_csv(Path(catalog_path))
    else:
        catalog = load_catalog()
    
    return find_products_in_text(text, catalog)


# Summary sheet detection
SUMMARY_INDICATORS = [
    "SUMMARY OF PAY ITEMS",
    "ESTIMATE OF QUANTITIES",
    "QUANTITY SUMMARY", 
    "PAY ITEM SUMMARY",
    "BID SCHEDULE",
    "SUMMARY OF ESTIMATED QUANTITIES",
    "ENGINEER'S ESTIMATE",
]


def is_summary_page(page_text: str) -> bool:
    """Check if a page is a summary/quantity page (higher quality data)."""
    text_upper = page_text.upper()
    return any(indicator in text_upper for indicator in SUMMARY_INDICATORS)
