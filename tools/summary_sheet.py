"""
Summary Sheet Detection and Parsing

Detects and extracts data from "Summary of Pay Items" / "Estimate of Quantities"
pages in construction PDFs. These summary pages contain cleaner, more structured
data than individual drawing pages.

Phase 3 of Product-First Matching Architecture.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set
from enum import Enum

logger = logging.getLogger(__name__)


class SummaryPageType(Enum):
    """Types of summary pages found in construction plans."""
    SUMMARY_OF_QUANTITIES = "summary_of_quantities"
    TABULATION_OF_QUANTITIES = "tabulation_of_quantities"
    ESTIMATE_OF_QUANTITIES = "estimate_of_quantities"
    PAY_ITEM_SUMMARY = "pay_item_summary"
    BID_SCHEDULE = "bid_schedule"
    ENGINEERS_ESTIMATE = "engineers_estimate"
    UNKNOWN = "unknown"


@dataclass
class SummaryItem:
    """A single item extracted from a summary sheet."""
    pay_item_no: str  # FDOT code like "430-175-118" or "N/A"
    description: str
    quantity: float
    unit: str
    
    # Optional fields
    bid_number: Optional[int] = None  # Line number in bid schedule
    sheet_references: List[str] = field(default_factory=list)  # Which drawing sheets
    
    # Metadata
    confidence: float = 0.95  # Summary sheets are high confidence
    source: str = "summary_sheet"
    raw_text: str = ""  # Original text that was parsed
    
    def __repr__(self):
        return f"SummaryItem({self.pay_item_no}, {self.description[:30]}..., {self.quantity} {self.unit})"


@dataclass
class SummaryPageResult:
    """Result of parsing a summary page."""
    page_num: int
    page_type: SummaryPageType
    items: List[SummaryItem]
    raw_text: str
    
    # Parsing metadata
    total_lines_parsed: int = 0
    items_extracted: int = 0
    parsing_errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Rate of successfully parsed lines."""
        if self.total_lines_parsed == 0:
            return 0.0
        return self.items_extracted / self.total_lines_parsed


# ============================================================================
# Summary Page Detection
# ============================================================================

# Primary indicators (high confidence)
SUMMARY_INDICATORS_PRIMARY = [
    "SUMMARY OF QUANTITIES",
    "SUMMARY OF PAY ITEMS",
    "TABULATION OF QUANTITIES",
    "ESTIMATE OF QUANTITIES",
    "PAY ITEM SUMMARY",
    "BID SCHEDULE",
    "ENGINEER'S ESTIMATE",
    "SUMMARY OF ESTIMATED QUANTITIES",
]

# Secondary indicators (support detection, lower confidence alone)
SUMMARY_INDICATORS_SECONDARY = [
    "QUANTITY SUMMARY",
    "BID ITEM",
    "PAY ITEM NO",
    "FDOT ITEM",
    "ITEM DESCRIPTION",
    "GRAND TOTAL",
    "TOTAL THIS SHEET",
]

# Table header patterns that confirm summary page structure
SUMMARY_TABLE_HEADERS = [
    r"PAY\s*ITEM\s*(?:NO\.?|NUMBER)",
    r"ITEM\s+DESCRIPTION",
    r"FDOT\s*(?:PAY\s*)?ITEM",
    r"BID\s*(?:ITEM\s*)?(?:NO\.?|NUMBER)",
    r"QUANTITY\s+UNIT",
    r"UNIT\s+QUANTITY",
]


def is_summary_page(page_text: str) -> bool:
    """
    Check if a page is a summary/quantity page.
    
    Summary pages are high-value targets because they contain clean,
    tabular data that's easier to parse than individual drawing pages.
    
    Args:
        page_text: Full text from a single page
        
    Returns:
        True if this appears to be a summary page
    """
    if not page_text or len(page_text) < 100:
        return False
    
    text_upper = page_text.upper()
    
    # Check primary indicators (one is enough)
    for indicator in SUMMARY_INDICATORS_PRIMARY:
        if indicator in text_upper:
            return True
    
    # Check for combination of secondary indicators (need 2+)
    secondary_matches = sum(1 for ind in SUMMARY_INDICATORS_SECONDARY if ind in text_upper)
    if secondary_matches >= 2:
        # Also verify it looks like a table (has units)
        has_units = any(u in text_upper for u in ['LF', 'EA', 'LS', 'SY', 'CY', 'SF', 'TON'])
        if has_units:
            return True
    
    return False


def detect_summary_page_type(page_text: str) -> SummaryPageType:
    """
    Determine the specific type of summary page.
    
    Different summary formats require different parsing strategies.
    """
    text_upper = page_text.upper()
    
    if "SUMMARY OF QUANTITIES" in text_upper:
        return SummaryPageType.SUMMARY_OF_QUANTITIES
    elif "TABULATION OF QUANTITIES" in text_upper:
        return SummaryPageType.TABULATION_OF_QUANTITIES
    elif "ESTIMATE OF QUANTITIES" in text_upper:
        return SummaryPageType.ESTIMATE_OF_QUANTITIES
    elif "PAY ITEM SUMMARY" in text_upper:
        return SummaryPageType.PAY_ITEM_SUMMARY
    elif "BID SCHEDULE" in text_upper:
        return SummaryPageType.BID_SCHEDULE
    elif "ENGINEER'S ESTIMATE" in text_upper or "ENGINEERS ESTIMATE" in text_upper:
        return SummaryPageType.ENGINEERS_ESTIMATE
    else:
        return SummaryPageType.UNKNOWN


# ============================================================================
# Summary Table Parsing
# ============================================================================

# FDOT pay item code pattern
FDOT_CODE_PATTERN = re.compile(r'\b(\d{3}-\d{1,3}(?:-\d{1,3})?)\b')

# Unit patterns
VALID_UNITS = {'LS', 'EA', 'LF', 'SY', 'CY', 'SF', 'TON', 'GAL', 'AC', 'TN', 'GM', 'DA', 'AS', 'PI'}
UNIT_PATTERN = re.compile(r'\b(' + '|'.join(VALID_UNITS) + r')\b', re.IGNORECASE)

# Quantity pattern (handles decimals and commas)
QUANTITY_PATTERN = re.compile(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b')


def extract_summary_table(page_text: str, page_num: int = 1) -> SummaryPageResult:
    """
    Extract structured data from a summary page.
    
    Handles multiple formats:
    - Native PDF text with separated fields
    - OCR text with space-separated fields
    - Multi-column formats with sheet references
    
    Args:
        page_text: Full text from the summary page
        page_num: Page number for reference
        
    Returns:
        SummaryPageResult with extracted items
    """
    page_type = detect_summary_page_type(page_text)
    
    # Try different parsing strategies based on page type
    if page_type == SummaryPageType.TABULATION_OF_QUANTITIES:
        items, errors, total = _parse_tabulation_format(page_text)
    else:
        items, errors, total = _parse_standard_format(page_text)
    
    return SummaryPageResult(
        page_num=page_num,
        page_type=page_type,
        items=items,
        raw_text=page_text,
        total_lines_parsed=total,
        items_extracted=len(items),
        parsing_errors=errors
    )


def _parse_standard_format(page_text: str) -> Tuple[List[SummaryItem], List[str], int]:
    """
    Parse standard "Summary of Quantities" format.
    
    Format (native PDF, fields on separate lines):
        25                                              <- bid number
        430-175-118                                     <- FDOT code
        REINFORCED CONCRETE PIPE (STORM & CROSS DRAIN) <- description
        160                                             <- quantity
        LF                                              <- unit
        
    Or inline format:
        430-175-118 PIPE CULVERT, ROUND, 18" 160 LF
    """
    items = []
    errors = []
    total_lines = 0
    
    lines = page_text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    # First pass: find all FDOT code positions
    fdot_positions = []
    for i, line in enumerate(lines):
        fdot_match = FDOT_CODE_PATTERN.match(line)  # Must be at start of line
        if fdot_match and len(line) < 20:  # FDOT codes are short lines
            fdot_positions.append((i, fdot_match.group(1)))
    
    # Process each FDOT code
    for idx, (pos, fdot_code) in enumerate(fdot_positions):
        total_lines += 1
        
        # Check if previous line is a bid number
        bid_number = None
        if pos > 0:
            prev_line = lines[pos - 1].strip()
            if prev_line.isdigit() and len(prev_line) <= 2:
                bid_number = int(prev_line)
        
        # Get description from next line
        desc_idx = pos + 1
        if desc_idx >= len(lines):
            errors.append(f"No description after FDOT code at line {pos}: {fdot_code}")
            continue
        
        description = lines[desc_idx].strip()
        
        # Skip if description looks wrong
        if description.replace(',', '').replace('.', '').isdigit():
            errors.append(f"Description looks like number at line {pos}: {fdot_code}")
            continue
        if description.upper() in VALID_UNITS:
            errors.append(f"Description looks like unit at line {pos}: {fdot_code}")
            continue
        
        # Get quantity and unit from following lines
        quantity = 0.0
        unit = ""
        
        # Look at next 2-3 lines for quantity and unit
        for j in range(desc_idx + 1, min(desc_idx + 4, len(lines))):
            line = lines[j].strip()
            
            # Stop if we hit the next FDOT code
            if FDOT_CODE_PATTERN.match(line):
                break
            # Stop if we hit a new bid number (next item)
            if line.isdigit() and len(line) <= 2 and j + 1 < len(lines):
                if FDOT_CODE_PATTERN.match(lines[j + 1].strip()):
                    break
            
            # Check for unit
            if line.upper() in VALID_UNITS:
                unit = line.upper()
                continue
            
            # Check for quantity
            qty_clean = line.replace(',', '')
            try:
                qty = float(qty_clean)
                if 0 < qty < 1000000:
                    quantity = qty
                    continue
            except ValueError:
                pass
        
        if quantity == 0:
            errors.append(f"Could not find quantity for {fdot_code}")
            continue
        
        items.append(SummaryItem(
            pay_item_no=fdot_code,
            description=description,
            quantity=quantity,
            unit=unit or "EA",
            bid_number=bid_number,
            confidence=0.95,
            raw_text=f"{fdot_code}\n{description}\n{quantity}\n{unit}"
        ))
    
    return items, errors, total_lines


def _try_parse_multiline(lines: List[str], start_idx: int, fdot_code: str) -> Optional[SummaryItem]:
    """
    Try to parse item from multi-line format (native PDF extraction).
    
    Expected format (common in FDOT plans):
        Line 0: Bid number (single integer 1-99)
        Line 1: FDOT code (e.g., 430-175-118)
        Line 2: Description (e.g., REINFORCED CONCRETE PIPE...)
        Line 3: Quantity (e.g., 160 or 2,000)
        Line 4: Unit (e.g., LF, EA)
    """
    if start_idx + 3 >= len(lines):
        return None
    
    curr_line = lines[start_idx].strip()
    
    # Check if current line is the bid number
    if curr_line.isdigit() and len(curr_line) <= 3 and int(curr_line) <= 99:
        # Pattern: bid_number -> fdot_code -> description -> quantity -> unit
        bid_number = int(curr_line)
        fdot_idx = start_idx + 1
    else:
        # Pattern: fdot_code -> description -> quantity -> unit
        bid_number = None
        fdot_idx = start_idx
    
    # Verify FDOT code is on expected line
    if fdot_idx >= len(lines):
        return None
    
    fdot_line = lines[fdot_idx].strip()
    if fdot_code not in fdot_line:
        return None
    
    # For 5-line format: lines are bid, fdot, desc, qty, unit
    # For 4-line format: lines are fdot, desc, qty, unit
    desc_idx = fdot_idx + 1
    
    if desc_idx >= len(lines):
        return None
    
    description = lines[desc_idx].strip()
    
    # Skip if description looks like a number or unit (parsing error)
    if description.replace(',', '').replace('.', '').isdigit():
        return None
    if description.upper() in VALID_UNITS:
        return None
    
    # Look for quantity and unit in next 2-3 lines
    quantity = 0.0
    unit = ""
    
    for j in range(desc_idx + 1, min(desc_idx + 4, len(lines))):
        line = lines[j].strip()
        line_upper = line.upper()
        
        # Skip if it's another FDOT code (next item started)
        if FDOT_CODE_PATTERN.match(line):
            break
        
        # Skip if it's another bid number (next item started)
        if line.isdigit() and len(line) <= 2:
            next_idx = j + 1
            if next_idx < len(lines) and FDOT_CODE_PATTERN.match(lines[next_idx].strip()):
                break
        
        # Check for unit (exact match to valid units)
        if line_upper in VALID_UNITS:
            unit = line_upper
            continue
        
        # Check for quantity (number, possibly with commas)
        qty_clean = line.replace(',', '')
        try:
            qty = float(qty_clean)
            if 0 < qty < 1000000:  # Sanity check
                quantity = qty
                continue
        except ValueError:
            pass
        
        # If it's text and we don't have description yet, it might be description continuation
        if not description:
            description = line
    
    if not description or quantity == 0:
        return None
    
    return SummaryItem(
        pay_item_no=fdot_code,
        description=description.strip(),
        quantity=quantity,
        unit=unit or "EA",
        bid_number=bid_number,
        confidence=0.95,
        raw_text="\n".join(lines[start_idx:min(fdot_idx + 5, len(lines))])
    )


def _try_parse_inline(line: str, fdot_code: str) -> Optional[SummaryItem]:
    """
    Parse item from single line format.
    
    Format: 430-175-118 PIPE CULVERT, ROUND, 18" 160 LF
    """
    # Remove FDOT code from line
    rest = line.replace(fdot_code, '').strip()
    
    # Find unit
    unit_match = UNIT_PATTERN.search(rest)
    if not unit_match:
        return None
    unit = unit_match.group(1).upper()
    
    # Find quantity (number near the unit)
    # Look for number before the unit
    before_unit = rest[:unit_match.start()].strip()
    after_unit = rest[unit_match.end():].strip()
    
    quantity = 0.0
    description = ""
    
    # Check if quantity is right before unit
    qty_before = QUANTITY_PATTERN.findall(before_unit)
    if qty_before:
        quantity = _parse_number(qty_before[-1])
        # Description is everything before the quantity
        qty_pos = before_unit.rfind(qty_before[-1])
        description = before_unit[:qty_pos].strip()
    elif after_unit:
        # Sometimes quantity comes after unit
        qty_after = QUANTITY_PATTERN.findall(after_unit)
        if qty_after:
            quantity = _parse_number(qty_after[0])
        description = before_unit.strip()
    
    if quantity == 0 or not description:
        return None
    
    # Clean up description
    description = re.sub(r'\s+', ' ', description).strip()
    description = description.rstrip(',.-')
    
    return SummaryItem(
        pay_item_no=fdot_code,
        description=description,
        quantity=quantity,
        unit=unit,
        raw_text=line
    )


def _try_parse_nearby(lines: List[str], start_idx: int, fdot_code: str) -> Optional[SummaryItem]:
    """
    Fallback parser: look for quantity/unit in nearby lines.
    """
    curr_line = lines[start_idx]
    
    # Build context from nearby lines
    context_start = max(0, start_idx - 2)
    context_end = min(len(lines), start_idx + 4)
    context = '\n'.join(lines[context_start:context_end])
    
    # Find all quantities in context
    quantities = QUANTITY_PATTERN.findall(context)
    units = UNIT_PATTERN.findall(context)
    
    if not quantities or not units:
        return None
    
    # Use the largest quantity (most likely to be the actual qty, not a size)
    quantity = max(_parse_number(q) for q in quantities)
    unit = units[0].upper()
    
    # Build description from non-numeric parts
    description = curr_line
    if fdot_code in description:
        description = description.replace(fdot_code, '').strip()
    # Remove numbers and units
    description = QUANTITY_PATTERN.sub('', description)
    description = UNIT_PATTERN.sub('', description)
    description = re.sub(r'\s+', ' ', description).strip()
    
    if not description:
        return None
    
    return SummaryItem(
        pay_item_no=fdot_code,
        description=description,
        quantity=quantity,
        unit=unit,
        confidence=0.75,  # Lower confidence for fallback parsing
        raw_text=context
    )


def _parse_tabulation_format(page_text: str) -> Tuple[List[SummaryItem], List[str], int]:
    """
    Parse "Tabulation of Quantities" format with multiple sheet columns.
    
    Common format (each field on separate line):
        430-175-118                         <- FDOT code
        PIPE CULVERT, OPTIONAL MATERIAL...  <- Description
        LF                                  <- Unit
        391                                 <- Quantity (may repeat for each sheet)
        391                                 <- Total
    """
    items = []
    errors = []
    total_lines = 0
    
    lines = page_text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    # Find all FDOT code positions (must be alone on their line)
    fdot_positions = []
    for i, line in enumerate(lines):
        # FDOT code should be alone on the line (maybe with minimal extra)
        if len(line) < 20:
            fdot_match = FDOT_CODE_PATTERN.match(line)
            if fdot_match:
                fdot_positions.append((i, fdot_match.group(1)))
    
    # Parse each FDOT code block
    for idx, (pos, fdot_code) in enumerate(fdot_positions):
        total_lines += 1
        
        # Find the next FDOT code position to know our boundary
        next_fdot_pos = fdot_positions[idx + 1][0] if idx + 1 < len(fdot_positions) else len(lines)
        
        # Collect lines between current FDOT and next
        block_lines = lines[pos + 1:next_fdot_pos]
        
        if not block_lines:
            errors.append(f"No data after FDOT code {fdot_code}")
            continue
        
        description = ""
        unit = ""
        quantities = []
        
        for line in block_lines:
            line_upper = line.upper().strip()
            
            # Check for unit (exact match)
            if line_upper in VALID_UNITS:
                unit = line_upper
                continue
            
            # Check for quantity (number only)
            qty_clean = line.replace(',', '').strip()
            try:
                qty = float(qty_clean)
                if 0 < qty < 1000000:
                    quantities.append(qty)
                    continue
            except ValueError:
                pass
            
            # Otherwise it's likely description
            if not description:
                description = line.strip()
        
        # Use the last quantity (usually the total)
        quantity = quantities[-1] if quantities else 0.0
        
        if not description:
            errors.append(f"No description for FDOT code {fdot_code}")
            continue
        
        if quantity == 0:
            errors.append(f"No quantity for FDOT code {fdot_code}")
            continue
        
        items.append(SummaryItem(
            pay_item_no=fdot_code,
            description=description,
            quantity=quantity,
            unit=unit or "EA",
            confidence=0.95,
            raw_text=f"{fdot_code}\n{description}\n{unit}\n{quantity}"
        ))
    
    return items, errors, total_lines


def _parse_number(s: str) -> float:
    """Parse a number string (handles commas, decimals)."""
    try:
        return float(s.replace(',', ''))
    except (ValueError, TypeError):
        return 0.0


# ============================================================================
# Drainage Item Filtering
# ============================================================================

# FDOT codes for drainage items (430-xxx for pipe, 425-xxx for structures)
DRAINAGE_CODE_PREFIXES = ('430-', '425-')

# Keywords indicating drainage items
DRAINAGE_KEYWORDS = {
    'PIPE', 'RCP', 'PVC', 'HDPE', 'CMP', 'CULVERT', 'DRAIN', 'STORM',
    'INLET', 'MANHOLE', 'CATCH BASIN', 'JUNCTION', 'ENDWALL', 'MES',
    'MITERED END', 'FLARED END', 'STRUCTURE'
}


def filter_drainage_items(items: List[SummaryItem]) -> List[SummaryItem]:
    """
    Filter summary items to only drainage-related pay items.
    
    We only price drainage items, so non-drainage items are extracted
    but not matched to the price list.
    """
    drainage_items = []
    
    for item in items:
        # Check FDOT code prefix
        if item.pay_item_no.startswith(DRAINAGE_CODE_PREFIXES):
            drainage_items.append(item)
            continue
        
        # Check description keywords
        desc_upper = item.description.upper()
        if any(kw in desc_upper for kw in DRAINAGE_KEYWORDS):
            drainage_items.append(item)
    
    return drainage_items


# ============================================================================
# Multi-Page Summary Handling
# ============================================================================

def find_summary_pages(page_texts: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    Find all summary pages in a document.
    
    Args:
        page_texts: List of (page_num, text) tuples
        
    Returns:
        List of (page_num, text) for summary pages
    """
    summary_pages = []
    
    for page_num, text in page_texts:
        if is_summary_page(text):
            summary_pages.append((page_num, text))
    
    return summary_pages


def extract_all_summary_items(
    page_texts: List[Tuple[int, str]]
) -> Tuple[List[SummaryItem], List[SummaryPageResult]]:
    """
    Extract items from all summary pages in a document.
    
    Deduplicates items that appear on multiple pages.
    
    Args:
        page_texts: List of (page_num, text) tuples
        
    Returns:
        Tuple of (deduplicated items, all page results)
    """
    all_items: Dict[str, SummaryItem] = {}  # Keyed by fdot_code + quantity
    page_results = []
    
    summary_pages = find_summary_pages(page_texts)
    
    for page_num, text in summary_pages:
        result = extract_summary_table(text, page_num)
        page_results.append(result)
        
        for item in result.items:
            # Dedupe key: FDOT code + quantity (same item on multiple pages)
            key = f"{item.pay_item_no}_{item.quantity}"
            
            if key not in all_items:
                all_items[key] = item
            else:
                # Keep the one with higher confidence
                if item.confidence > all_items[key].confidence:
                    all_items[key] = item
    
    return list(all_items.values()), page_results


# ============================================================================
# Conversion to Pay Item Format
# ============================================================================

def summary_item_to_pay_item(item: SummaryItem) -> Dict:
    """
    Convert a SummaryItem to the standard pay_item dict format.
    
    This allows summary sheet data to be used directly in the
    existing pipeline alongside regex-extracted items.
    """
    return {
        'pay_item_no': item.pay_item_no,
        'description': item.description,
        'unit': item.unit,
        'quantity': item.quantity,
        'matched': False,  # Will be matched by price lookup
        'unit_price': None,
        'line_cost': None,
        'source': 'summary_sheet',
        'confidence': 'high' if item.confidence >= 0.9 else 'medium',
        'confidence_score': item.confidence,
        'needs_verification': item.quantity == 0 or item.confidence < 0.7,
        'bid_number': item.bid_number,
    }


# ============================================================================
# Testing / CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python summary_sheet.py <pdf_file>")
        print("\nDetects and extracts data from summary/quantity pages.")
        sys.exit(1)
    
    import fitz
    
    pdf_path = sys.argv[1]
    doc = fitz.open(pdf_path)
    
    print(f"\nAnalyzing: {pdf_path}")
    print(f"Total pages: {len(doc)}")
    print("=" * 60)
    
    # Find summary pages
    page_texts = [(i + 1, doc[i].get_text()) for i in range(len(doc))]
    summary_pages = find_summary_pages(page_texts)
    
    print(f"\nFound {len(summary_pages)} summary page(s)")
    
    for page_num, text in summary_pages:
        print(f"\n{'='*60}")
        print(f"Page {page_num}: {detect_summary_page_type(text).value}")
        print("-" * 60)
        
        result = extract_summary_table(text, page_num)
        
        print(f"Items extracted: {result.items_extracted}")
        print(f"Parsing errors: {len(result.parsing_errors)}")
        
        # Show drainage items
        drainage = filter_drainage_items(result.items)
        print(f"Drainage items: {len(drainage)}")
        
        for item in drainage[:10]:  # Show first 10
            print(f"  {item.pay_item_no}: {item.description[:40]}... {item.quantity} {item.unit}")
        
        if len(drainage) > 10:
            print(f"  ... and {len(drainage) - 10} more")
        
        if result.parsing_errors:
            print(f"\nParsing errors:")
            for err in result.parsing_errors[:5]:
                print(f"  - {err}")
    
    doc.close()
