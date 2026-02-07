#!/usr/bin/env python3
"""
Construction Takeoff Analyzer
Parses extracted text from construction plans and matches to FL 2025 price list.

Supports two matching modes:
- Product-First (default): Uses ProductCatalog to search for known products
- Legacy Regex: Falls back to regex patterns for items product matcher misses
"""

import csv
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Import validation gates
try:
    from .validation_gates import (
        VALID_PIPE_SIZES, VALID_ELLIPTICAL_SIZES, MAX_PIPE_SIZE,
        validate_quantity, validate_description, validate_pipe_size,
        validate_all_items, ValidationReport
    )
except ImportError:
    # Fallback for direct script execution
    from validation_gates import (
        VALID_PIPE_SIZES, VALID_ELLIPTICAL_SIZES, MAX_PIPE_SIZE,
        validate_quantity, validate_description, validate_pipe_size,
        validate_all_items, ValidationReport
    )

# Import product-first matching modules (Phase 2)
try:
    from .product_catalog import ProductCatalog, load_catalog, DEFAULT_CATALOG_PATH
    from .product_matcher import find_products_in_text, Match, is_summary_page
    PRODUCT_FIRST_AVAILABLE = True
except ImportError:
    try:
        from product_catalog import ProductCatalog, load_catalog, DEFAULT_CATALOG_PATH
        from product_matcher import find_products_in_text, Match, is_summary_page
        PRODUCT_FIRST_AVAILABLE = True
    except ImportError:
        PRODUCT_FIRST_AVAILABLE = False
        ProductCatalog = None

# Import structure matcher for inlets, manholes, etc.
try:
    from .structure_matcher import match_structure, StructureMatch
    STRUCTURE_MATCHER_AVAILABLE = True
except ImportError:
    try:
        from structure_matcher import match_structure, StructureMatch
        STRUCTURE_MATCHER_AVAILABLE = True
    except ImportError:
        STRUCTURE_MATCHER_AVAILABLE = False
        match_structure = None
        StructureMatch = None

# Import summary sheet extraction (Phase 3)
try:
    from .summary_sheet import (
        extract_summary_table, 
        filter_drainage_items,
        summary_item_to_pay_item,
        SummaryItem,
        SummaryPageResult,
    )
    SUMMARY_SHEET_AVAILABLE = True
except ImportError:
    try:
        from summary_sheet import (
            extract_summary_table,
            filter_drainage_items,
            summary_item_to_pay_item,
            SummaryItem,
            SummaryPageResult,
        )
        SUMMARY_SHEET_AVAILABLE = True
    except ImportError:
        SUMMARY_SHEET_AVAILABLE = False

logger = logging.getLogger(__name__)

# Re-export for backward compatibility (sizes now come from validation_gates)
# VALID_PIPE_SIZES, VALID_ELLIPTICAL_SIZES, MAX_PIPE_SIZE are imported above

# Structure types for endwalls and end sections
STRUCTURE_TYPES = {
    'STRAIGHT_ENDWALL': ['straight endwall', 'straight end wall', 'straight concrete endwall'],
    'WINGED_ENDWALL': ['winged endwall', 'winged end wall', 'wing endwall', 'wing wall'],
    'U_TYPE_ENDWALL': ['u-type endwall', 'u type endwall', 'u-type end wall', 'utype'],
    'MES': ['mes', 'mitered end section', 'mitered end', 'mitered'],
    'FLARED_END': ['flared end', 'flared end section', 'fes'],
    'PIPE_CRADLE': ['pipe cradle', 'cradle']
}

# Valid U-Type Endwall slope ratios from FL 2025 catalog
VALID_SLOPE_RATIOS = {'2:1', '3:1', '4:1', '6:1'}

# Valid MES run configurations
VALID_MES_CONFIGS = {'SINGLE RUN', 'DOUBLE RUN', 'TRIPLE RUN'}

# Valid MES frame options
VALID_MES_FRAME_OPTIONS = {'WITH FRAME', 'NO FRAME'}

# =============================================================================
# Shared Constants for Product Detection
# Used by page quality analysis, material summary, and item classification
# =============================================================================

# Pipe material keywords (canonical set for classification)
PIPE_MATERIALS = {'RCP', 'PVC', 'HDPE', 'CMP', 'DIP', 'SRCP', 'ERCP', 'PIPE', 'CULVERT'}

# Structure type keywords (canonical set for classification)
STRUCTURE_TYPE_KEYWORDS = (
    'INLET', 'MANHOLE', 'CATCH BASIN', 'JUNCTION BOX',
    'ENDWALL', 'MES', 'FLARED END', 'CRADLE'
)

# Combined product keywords for page quality analysis
# Includes materials, structures, abbreviations, and general terms
PRODUCT_KEYWORDS = (
    PIPE_MATERIALS |
    set(STRUCTURE_TYPE_KEYWORDS) |
    {'MH', 'CB', 'JB', 'FES'} |  # Common abbreviations
    {'STORM', 'DRAIN', 'SEWER'}   # General drainage terms
)

# Annotation/callout indicators suggesting complex layouts
ANNOTATION_KEYWORDS = {'SEE', 'NOTE', 'TYP', 'TYPICAL', 'DETAIL', 'SCHEDULE', 'TABLE'}


@dataclass
class PageProductAnalysis:
    """Analysis of product detection quality for a single page."""
    page_num: int
    product_keywords_found: int = 0
    complete_items_found: int = 0
    incomplete_items_found: int = 0
    callouts_without_quantity: int = 0
    has_complex_tables: bool = False
    has_pipe_schedule: bool = False
    needs_vision: bool = False
    reasons: List[str] = field(default_factory=list)


def analyze_page_product_quality(page_text: str, page_num: int = 1) -> PageProductAnalysis:
    """
    Analyze a page's text to determine product detection quality.

    This is used in hybrid mode to decide whether a page needs Vision API
    for better extraction, based on product detection quality rather than
    just OCR quality metrics.

    Args:
        page_text: The OCR-extracted text from a single page
        page_num: Page number for reference

    Returns:
        PageProductAnalysis with detection metrics and needs_vision flag
    """
    analysis = PageProductAnalysis(page_num=page_num)
    text_upper = page_text.upper()

    # Count product keywords found
    for keyword in PRODUCT_KEYWORDS:
        if keyword in text_upper:
            analysis.product_keywords_found += 1

    # Quick patterns for complete items (has size AND quantity)
    # These match items that have both a size and a quantity
    complete_patterns = [
        # FDOT format: 430-175-118 PIPE 18" LF 98
        r'\d{3}-\d{1,3}-\d{1,3}\s+.+?\s+(?:LS|EA|LF|SY|CY|SF|TON)\s+\d+',
        # Quantity-first: 51 LF 18" RCP
        r'\d+\s*(?:LF|EA|SF|SY)\s+\d+\s*["\u201d]?\s*(?:RCP|PVC|HDPE|CMP)',
        # Size-first with quantity: 18" RCP - 51 LF
        r'\d+\s*["\u201d]\s*(?:RCP|PVC|HDPE|CMP).*?\d+\s*(?:LF|EA)',
    ]

    for pattern in complete_patterns:
        matches = re.findall(pattern, text_upper, re.IGNORECASE)
        analysis.complete_items_found += len(matches)

    # Patterns for incomplete items (keyword but missing size or quantity)
    # Callouts like "RCP 24" without a length
    callout_pattern = r'\b(?:RCP|PVC|HDPE|CMP|DIP|SRCP)\s*-?\s*(\d{1,2})\b'
    callout_matches = re.findall(callout_pattern, text_upper)

    # Check if these callouts have quantities nearby
    for match in callout_matches:
        # Look for quantity pattern near the callout (within ~50 chars)
        context_pattern = rf'(?:RCP|PVC|HDPE|CMP|DIP|SRCP)\s*-?\s*{match}.*?(\d+)\s*(?:LF|EA)'
        if not re.search(context_pattern, text_upper):
            analysis.callouts_without_quantity += 1

    # Check for incomplete product mentions (keywords without full details)
    incomplete_patterns = [
        # Size mentioned but no quantity
        r'\b\d{1,2}\s*["\u201d]\s*(?:RCP|PVC|HDPE|CMP|DIP|SRCP)\b(?!.*\d+\s*(?:LF|EA))',
        # INLET/MANHOLE without ID number
        r'\b(?:INLET|MANHOLE|CATCH\s*BASIN)\b(?!\s*[#-]?\s*\d)',
    ]

    for pattern in incomplete_patterns:
        matches = re.findall(pattern, text_upper[:500], re.IGNORECASE)  # Check first 500 chars
        analysis.incomplete_items_found += len(matches)

    # Detect complex tables (pipe schedules)
    table_indicators = [
        'PIPE SCHEDULE', 'STORM SCHEDULE', 'DRAINAGE SCHEDULE',
        'QTY', 'QUAN', 'SIZE', 'LENGTH', 'STATION'
    ]
    table_count = sum(1 for ind in table_indicators if ind in text_upper)
    if table_count >= 2:
        analysis.has_complex_tables = True
        analysis.has_pipe_schedule = True

    # Check for annotation references that suggest more data elsewhere
    for ann in ANNOTATION_KEYWORDS:
        if ann in text_upper:
            analysis.has_complex_tables = True
            break

    # Determine if Vision is needed based on product detection issues
    reasons = []

    # Flag if products detected but items are incomplete
    if analysis.product_keywords_found > 0 and analysis.complete_items_found == 0:
        reasons.append(f"product_keywords_found ({analysis.product_keywords_found}) but no complete items")
        analysis.needs_vision = True

    # Flag if CAD callouts without quantities
    if analysis.callouts_without_quantity > 0:
        reasons.append(f"callouts_without_quantity ({analysis.callouts_without_quantity})")
        analysis.needs_vision = True

    # Flag if incomplete items exceed complete items
    if analysis.incomplete_items_found > analysis.complete_items_found and analysis.product_keywords_found > 2:
        reasons.append(f"incomplete_items ({analysis.incomplete_items_found}) > complete_items ({analysis.complete_items_found})")
        analysis.needs_vision = True

    # Flag if complex tables/schedules detected (Vision better at reading these)
    if analysis.has_pipe_schedule:
        reasons.append("pipe_schedule_detected")
        analysis.needs_vision = True

    analysis.reasons = reasons
    return analysis


def generate_material_summary(pay_items: List[Dict]) -> Dict:
    """
    Generate a material takeoff summary from pay items.

    Groups items by type and size to provide totals:
    - Pipe summary: total LF per size/material
    - Structure summary: counts by type
    - Grand totals

    Args:
        pay_items: List of pay item dictionaries

    Returns:
        Material summary dictionary
    """
    pipe_summary = {}  # key: "18\"_RCP" -> {"total_lf": X, "count": Y}
    structure_summary = {}  # key: "INLET" -> {"count": X}

    for item in pay_items:
        desc = item.get('description', '').upper()
        qty = item.get('quantity', 0) or 0
        unit = item.get('unit', '').upper()

        # Classify as pipe or structure using shared constants
        is_pipe = any(mat in desc for mat in PIPE_MATERIALS)
        is_structure = any(struct in desc for struct in STRUCTURE_TYPE_KEYWORDS)

        if is_pipe and unit == 'LF':
            # Extract size and material for grouping
            size_match = re.search(r'(\d{1,2})\s*["\u201d]', desc)
            mat_match = re.search(r'\b(RCP|PVC|HDPE|CMP|DIP|SRCP|ERCP)\b', desc)

            if size_match:
                size = size_match.group(1)
                material = mat_match.group(1) if mat_match else 'PIPE'
                key = f'{size}"_{material}'

                if key not in pipe_summary:
                    pipe_summary[key] = {'total_lf': 0, 'count': 0, 'size': size, 'material': material}
                pipe_summary[key]['total_lf'] += qty
                pipe_summary[key]['count'] += 1

        elif is_structure or unit == 'EA':
            # Determine structure type using shared constant
            struct_type = 'OTHER'
            for st in STRUCTURE_TYPE_KEYWORDS:
                if st in desc:
                    struct_type = st
                    break

            if struct_type not in structure_summary:
                structure_summary[struct_type] = {'count': 0}
            structure_summary[struct_type]['count'] += int(qty) if qty else 1

    # Calculate totals
    total_pipe_lf = sum(p['total_lf'] for p in pipe_summary.values())
    total_structures = sum(s['count'] for s in structure_summary.values())

    return {
        'pipe_summary': pipe_summary,
        'structure_summary': structure_summary,
        'totals': {
            'total_pipe_lf': total_pipe_lf,
            'total_pipe_sizes': len(pipe_summary),
            'total_structures': total_structures,
            'total_structure_types': len(structure_summary)
        }
    }


class TakeoffAnalyzer:
    """Analyzes construction plan text and generates takeoff reports."""

    def __init__(self, price_list_path: str = None, use_product_first: bool = True):
        """
        Initialize with optional price list path.
        
        Args:
            price_list_path: Path to FL 2025 price list CSV
            use_product_first: If True (default), use product-first matching as primary.
                               Falls back to regex patterns for missed items.
        """
        self.price_list = {}
        self.price_list_by_fdot = {}  # Index by FDOT code for fast lookup
        self.price_list_loaded = False
        self._text_blocks = []  # OCR text blocks with locations
        
        # Product-first matching (Phase 2)
        self.use_product_first = use_product_first and PRODUCT_FIRST_AVAILABLE
        self.product_catalog = None
        
        if price_list_path:
            self.load_price_list(price_list_path)
            # Also load ProductCatalog from same path for product-first matching
            if self.use_product_first:
                try:
                    self.product_catalog = ProductCatalog.from_csv(Path(price_list_path))
                    logger.info(f"Loaded ProductCatalog with {len(self.product_catalog)} products")
                except Exception as e:
                    logger.warning(f"Could not load ProductCatalog: {e}")
                    self.product_catalog = None

    def _find_source_location(self, match_text: str, description: str = None) -> Optional[Dict]:
        """
        Find the source location for a matched text in the text blocks.

        Args:
            match_text: The text that was matched by regex
            description: Optional additional text for matching

        Returns:
            Source location dict with page, bbox, and text_context, or None
        """
        if not self._text_blocks:
            return None

        # Normalize the match text for comparison
        match_text_lower = match_text.lower().strip()
        desc_lower = description.lower().strip() if description else ""

        best_match = None
        best_score = 0

        for block in self._text_blocks:
            block_text = block.get("text", "").lower()

            # Calculate match score
            score = 0

            # Check if the block contains key parts of the match
            if match_text_lower in block_text:
                score += 100
            elif any(part in block_text for part in match_text_lower.split() if len(part) > 2):
                # Partial match - some words match
                matching_words = sum(1 for part in match_text_lower.split() if part in block_text and len(part) > 2)
                score += matching_words * 20

            # Check description match
            if desc_lower:
                if desc_lower in block_text:
                    score += 50
                elif any(part in block_text for part in desc_lower.split() if len(part) > 2):
                    matching_words = sum(1 for part in desc_lower.split() if part in block_text and len(part) > 2)
                    score += matching_words * 10

            if score > best_score:
                best_score = score
                best_match = block

        if best_match and best_score >= 20:
            return {
                "page": best_match.get("page", 1),
                "bbox": best_match.get("bbox", []),
                "text_context": best_match.get("text", "")[:100],
                "ocr_confidence": best_match.get("confidence", 0.0)
            }

        return None

    def _attach_source_location(self, item: Dict, match_text: str) -> Dict:
        """
        Attach source location to an item if text_blocks are available.

        Args:
            item: Pay item dict
            match_text: The text that was matched

        Returns:
            Item with source_location attached if found
        """
        if self._text_blocks:
            location = self._find_source_location(
                match_text,
                item.get("description", "")
            )
            if location:
                item["source_location"] = location
        return item

    def load_price_list(self, path: str) -> bool:
        """
        Load Florida 2025 price list from CSV.

        Builds two indexes:
        - price_list: keyed by size_configuration_product_type
        - price_list_by_fdot: keyed by normalized FDOT code (e.g., "430-030")

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    product = {
                        'product_type': row.get('Product_Type', ''),
                        'size': row.get('Size', ''),
                        'configuration': row.get('Configuration', ''),
                        'price': self._parse_price(row.get('Price', '$0')),
                        'unit': row.get('Unit', ''),
                        'fdot_code': row.get('FDOT_Code', '')
                    }

                    # Primary index by size/config/type
                    key = f"{product['size']}_{product['configuration']}_{product['product_type']}"
                    self.price_list[key] = product

                    # Secondary index by FDOT code (normalized)
                    fdot_raw = row.get('FDOT_Code', '')
                    if fdot_raw:
                        # Normalize: "FDOT 430-030" -> "430-030"
                        fdot_normalized = fdot_raw.replace('FDOT ', '').replace('FDOT', '').strip()
                        if fdot_normalized:
                            # Group by FDOT code (multiple products may share same code)
                            if fdot_normalized not in self.price_list_by_fdot:
                                self.price_list_by_fdot[fdot_normalized] = []
                            self.price_list_by_fdot[fdot_normalized].append(product)

            self.price_list_loaded = True
            return True
        except Exception as e:
            print(f"Warning: Could not load price list: {e}")
            return False

    def _parse_price(self, price_str: str) -> float:
        """Parse price string to float."""
        try:
            return float(price_str.replace('$', '').replace(',', ''))
        except (ValueError, TypeError, AttributeError):
            return 0.0

    def extract_pay_items(self, text: str, text_blocks: List[Dict] = None) -> List[Dict]:
        """
        Extract pay items from plan text using multiple pattern strategies.

        Extraction priority (Phase 3):
        1. Summary sheet detection - highest quality, use as primary when found
        2. Product-first matching - scan for known products
        3. Regex patterns - fallback for items missed by above

        Args:
            text: Extracted text from construction plans
            text_blocks: Optional list of OCR text blocks with bounding boxes
                         for source location tracking

        Returns:
            List of pay item dictionaries with optional source_location
        """
        # Store text_blocks for location tracking
        self._text_blocks = text_blocks or []

        seen = set()  # Avoid duplicates across all strategies
        
        # Phase 3: Check for summary page first (highest quality data)
        if SUMMARY_SHEET_AVAILABLE and is_summary_page(text):
            logger.info("Summary page detected - using summary sheet extraction as primary")
            return self._extract_from_summary_page(text, seen)
        
        # Product-first matching (Phase 2)
        if self.use_product_first and self.product_catalog:
            return self._extract_with_product_first(text, seen)
        
        # Legacy regex-only path
        return self._extract_items_from_page(text, seen)
    
    def _extract_from_summary_page(self, text: str, seen: set) -> List[Dict]:
        """
        Extract pay items from a summary page (Phase 3).
        
        Summary pages contain cleaner, tabular data with FDOT codes,
        descriptions, quantities, and units in a structured format.
        
        This is the highest-confidence extraction source.
        
        Args:
            text: Summary page text
            seen: Set of already-seen items for deduplication
            
        Returns:
            List of pay item dictionaries
        """
        items = []
        
        # Parse the summary table
        result = extract_summary_table(text)
        logger.info(f"Summary sheet parsed: {result.items_extracted} items extracted, "
                   f"{len(result.parsing_errors)} errors")
        
        # Convert summary items to pay_item format
        for summary_item in result.items:
            item_key = f"summary_{summary_item.pay_item_no}_{summary_item.quantity}"
            if item_key in seen:
                continue
            seen.add(item_key)
            
            pay_item = summary_item_to_pay_item(summary_item)
            
            # Try to match to price list by FDOT code
            if self.price_list_loaded:
                fdot_code = summary_item.pay_item_no
                if fdot_code in self.price_list_by_fdot:
                    products = self.price_list_by_fdot[fdot_code]
                    if products:
                        # Use first matching product (usually only one per FDOT code)
                        product = products[0]
                        pay_item['matched'] = True
                        pay_item['unit_price'] = product['price']
                        pay_item['line_cost'] = summary_item.quantity * product['price']
            
            items.append(pay_item)
        
        # Also run product-first matching to catch items summary parser might have missed
        # (especially items without FDOT codes)
        if self.use_product_first and self.product_catalog:
            product_matches = find_products_in_text(text, self.product_catalog)
            
            for match in product_matches:
                item_key = f"product_{match.product.id}_{match.quantity}"
                if item_key not in seen:
                    # Check if we already have this FDOT code from summary
                    already_have_fdot = any(
                        match.product.fdot_code and 
                        match.product.fdot_code in item.get('pay_item_no', '')
                        for item in items
                    )
                    if not already_have_fdot:
                        seen.add(item_key)
                        item = self._match_to_pay_item(match)
                        item['source'] = 'product_first_supplement'  # Indicate it supplemented summary
                        items.append(item)
        
        return items
    
    def _extract_with_product_first(self, text: str, seen: set) -> List[Dict]:
        """
        Extract pay items using product-first matching with regex fallback.
        
        Strategy:
        1. Use ProductCatalog to find known products in text
        2. Convert matches to pay_item format with confidence scores
        3. Run legacy regex patterns to catch items product matcher missed
        4. Deduplicate, preferring product-first matches (higher confidence)
        
        Returns:
            List of pay item dictionaries
        """
        items = []
        product_matched_sizes = set()  # Track what product matcher found
        
        # Step 1: Product-first matching
        matches = find_products_in_text(text, self.product_catalog)
        
        for match in matches:
            item = self._match_to_pay_item(match)
            item_key = f"product_{match.product.id}_{match.quantity}"
            if item_key not in seen:
                seen.add(item_key)
                items.append(item)
                
                # Track matched sizes for fallback dedup
                size_match = re.search(r'(\d+)', match.product.size)
                if size_match:
                    product_matched_sizes.add(size_match.group(1))
        
        # Step 2: Fall back to regex patterns for items product matcher missed
        regex_items = self._extract_items_from_page(text, seen.copy())
        
        # Step 3: Add regex items that don't duplicate product-first matches
        for item in regex_items:
            # Check if this is likely a duplicate of a product match
            desc = item.get('description', '').upper()
            pay_item_no = item.get('pay_item_no', '')
            
            # Extract size from item
            size_match = re.search(r'(\d+)', pay_item_no)
            item_size = size_match.group(1) if size_match else None
            
            # Skip if same size already matched by product-first (for same product type)
            is_duplicate = False
            if item_size and item_size in product_matched_sizes:
                # Check if same product category
                for existing in items:
                    existing_desc = existing.get('description', '').upper()
                    # Same size and similar product type = likely duplicate
                    if item_size in existing.get('pay_item_no', ''):
                        if ('RCP' in desc and 'RCP' in existing_desc) or \
                           ('ENDWALL' in desc and 'ENDWALL' in existing_desc) or \
                           ('MES' in desc and 'MES' in existing_desc):
                            is_duplicate = True
                            break
            
            if not is_duplicate:
                # Mark as fallback source with lower confidence
                if item.get('confidence') in ('high', 'medium'):
                    item['confidence_score'] = 0.6  # Downgrade regex matches
                items.append(item)
        
        return items
    
    def _match_to_pay_item(self, match: 'Match') -> Dict:
        """
        Convert a product_matcher.Match to a pay_item dictionary.
        
        Preserves confidence scores and adds product catalog metadata.
        """
        product = match.product
        
        # Build pay item number from product
        pay_item_no = product.fdot_code or f"{product.category}-{product.size}"
        
        # Build description
        desc_parts = [product.size, product.product_type]
        if product.configuration:
            desc_parts.append(product.configuration)
        description = ' '.join(filter(None, desc_parts))
        
        item = {
            'pay_item_no': pay_item_no,
            'description': description,
            'unit': match.unit or product.unit,
            'quantity': match.quantity,
            'matched': True,  # Already matched to catalog
            'unit_price': product.price if product.price > 0 else None,
            'line_cost': match.quantity * product.price if product.price > 0 else None,
            'source': 'product_first',
            'confidence': self._confidence_to_label(match.confidence),
            'confidence_score': match.confidence,  # Numeric 0.0-1.0
            'needs_verification': match.quantity == 0 or match.confidence < 0.6,
            'matched_product': {
                'product_type': product.product_type,
                'size': product.size,
                'configuration': product.configuration,
                'price': product.price,
                'unit': product.unit,
                'fdot_code': product.fdot_code,
                'category': product.category,
            }
        }
        
        # Attach source location if available
        if match.source_text:
            item['source_text'] = match.source_text[:200]
        
        return item
    
    def _confidence_to_label(self, confidence: float) -> str:
        """Convert numeric confidence to label."""
        if confidence >= 0.85:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        else:
            return 'low'

    def _extract_items_from_page(self, page_text: str, seen: set) -> List[Dict]:
        """
        Extract all pay items from text.

        Args:
            page_text: Text content to extract from
            seen: Set of already-seen items (for deduplication)

        Returns:
            List of pay items
        """
        items = []

        # Strategy 1: FDOT pay item codes (highest confidence)
        items.extend(self._extract_fdot_items(page_text, seen))

        # Strategy 1b: Multi-line FDOT pay item tables (native PDF extraction format)
        items.extend(self._extract_fdot_multiline_items(page_text, seen))

        # Strategy 2: Quantity-first format (federal/military style)
        items.extend(self._extract_quantity_first_items(page_text, seen))

        # Strategy 3: Elliptical pipe (14"x23" RCP HE, ERCP, etc.)
        items.extend(self._extract_elliptical_pipe_items(page_text, seen))

        # Strategy 4: Elliptical accessories (Flared End, MES for elliptical)
        items.extend(self._extract_elliptical_accessories(page_text, seen))

        # Strategy 5: Structures (endwalls, MES, flared ends, pipe cradles)
        items.extend(self._extract_structure_items(page_text, seen))

        # Strategy 6: Galvanized Steel MES (round and elliptical)
        items.extend(self._extract_galvanized_mes_items(page_text, seen))

        # Strategy 7: Table extraction (pipe schedules, quantity tables)
        items.extend(self._extract_table_items(page_text, seen))

        # Strategy 8: Pipe callouts (CAD annotations)
        items.extend(self._extract_callout_items(page_text, seen))

        # Strategy 9: Drainage structure labels
        items.extend(self._extract_drainage_labels(page_text, seen))

        # Post-processing: Remove redundant callouts that duplicate FDOT items
        items = self._dedupe_callouts_against_fdot(items)

        return items

    def _dedupe_callouts_against_fdot(self, items: List[Dict]) -> List[Dict]:
        """
        Remove callout items that are redundant with FDOT items.
        
        When an FDOT item like "430-175-118 PIPE CULVERT 18" RCP" exists,
        we don't need a separate "RCP-18 (VERIFY QTY)" callout item.
        """
        import re
        
        # Build set of pipe sizes from high-confidence items
        fdot_pipe_sizes = set()
        for item in items:
            if item.get('source') in ('fdot', 'quantity_first', 'table'):
                desc = item.get('description', '').upper()
                pay_item = item.get('pay_item_no', '').upper()
                
                # Extract size from various description formats:
                # - "18" RCP", "24" HDPE" (size before material)
                # - "ROUND 18"", "ROUND 24"" (size after ROUND)
                # - "SRCP-CLASS III, ROUND 18"" (FDOT format)
                size_patterns = [
                    r'(\d+)["\u201d]?\s*(RCP|PVC|HDPE|CMP|DIP|SRCP)',  # 18" RCP
                    r'(RCP|PVC|HDPE|CMP|DIP|SRCP)\s*-?\s*(\d+)',       # RCP-18
                    r'ROUND\s+(\d+)["\u201d]',                          # ROUND 18"
                ]
                
                for pattern in size_patterns:
                    match = re.search(pattern, desc)
                    if match:
                        groups = match.groups()
                        # Find the numeric group (the size)
                        for g in groups:
                            if g and g.isdigit():
                                fdot_pipe_sizes.add(g)
                                break
                        break
                
                # Also check pay item codes like 430-175-118 where last 3 digits encode size
                if pay_item.startswith('430-17'):
                    # FDOT pipe culvert codes: 430-17X-YYY where YYY is size code
                    # 118 = 18", 124 = 24", 130 = 30", etc.
                    size_code = pay_item.split('-')[-1] if '-' in pay_item else ''
                    if size_code.isdigit() and len(size_code) == 3:
                        # Extract size: 118 -> 18, 124 -> 24, 130 -> 30
                        size = size_code[1:]  # Remove first digit
                        if int(size) in VALID_PIPE_SIZES:
                            fdot_pipe_sizes.add(size)
        
        # Filter out callouts that match FDOT pipe sizes
        filtered_items = []
        for item in items:
            if item.get('source') == 'callout':
                # Extract size from pay_item_no like "RCP-18" -> "18"
                pay_item = item.get('pay_item_no', '')
                parts = pay_item.split('-')
                if len(parts) == 2:
                    size = parts[1]
                    if size in fdot_pipe_sizes:
                        # Skip - already covered by FDOT item with this size
                        continue
            filtered_items.append(item)
        
        return filtered_items

    def _extract_fdot_items(self, text: str, seen: set) -> List[Dict]:
        """Extract FDOT pay item codes (e.g., 430-175-118)."""
        items = []
        patterns = [
            # Standard format: 430-175-118 or 425-1-549
            r'(\d{3}-\d{1,3}-\d{1,3})\s+(.+?)\s+(LS|EA|LF|SY|CY|SF|TON|GAL|AC)\s+(\d+(?:\.\d+)?)',
            # Alternative: 101-1
            r'(\d{3}-\d{1,2})\s+(.+?)\s+(LS|EA|LF|SY|CY|SF|TON|GAL|AC)\s+(\d+(?:\.\d+)?)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                groups = match.groups()
                description = groups[1].strip()
                quantity = float(groups[3])
                unit = groups[2].upper()
                
                # VALIDATION GATE: Description validation
                # Reject items with invalid descriptions (numeric-only, too short, just units)
                desc_result, desc_warnings = validate_description(
                    description,
                    pay_item_no=groups[0]
                )
                if desc_result is None:
                    logger.debug(f"Rejected FDOT item {groups[0]}: invalid description '{description}'")
                    continue
                
                # VALIDATION GATE: Quantity validation
                # Reject quantities outside valid ranges for the unit type
                qty_result, qty_warnings = validate_quantity(
                    quantity, unit,
                    pay_item_no=groups[0]
                )
                if qty_result is None:
                    logger.debug(f"Rejected FDOT item {groups[0]}: invalid quantity {quantity} {unit}")
                    continue
                
                item_key = f"fdot_{groups[0]}_{groups[3]}"
                if item_key in seen:
                    continue
                seen.add(item_key)

                item = {
                    'pay_item_no': groups[0],
                    'description': description,
                    'unit': unit,
                    'quantity': quantity,
                    'matched': False,
                    'unit_price': None,
                    'line_cost': None,
                    'source': 'fdot',
                    'confidence': 'high'
                }
                
                # Add validation warnings to item if any
                if desc_warnings or qty_warnings:
                    item['validation_warnings'] = [
                        w.to_dict() for w in (desc_warnings + qty_warnings)
                        if w.severity.value == 'warning'
                    ]

                # Attach source location if available
                self._attach_source_location(item, match.group(0))
                items.append(item)

        return items

    def _extract_fdot_multiline_items(self, text: str, seen: set) -> List[Dict]:
        """
        Extract FDOT pay items from multi-line table format (native PDF extraction).

        Native PDF text extraction often produces each table cell on a separate line:
            Line N:   430-175-118                              (FDOT code)
            Line N+1: REINFORCED CONCRETE PIPE ... (18")       (description)
            Line N+2: 160                                      (quantity)
            Line N+3: LF                                       (unit)

        This handles the following FDOT code patterns:
        - Pipe culverts: 430-175-XXX
        - Inlets: 425-1-XXX
        - Manholes: 425-2-XXX
        - Endwalls: 430-030-XXX, 430-040-XXX
        - MES: 430-982-XXX

        Args:
            text: Page text to search
            seen: Set of already-seen items

        Returns:
            List of extracted pay items
        """
        items = []
        lines = text.split('\n')

        # FDOT code patterns for drainage items
        fdot_code_pattern = re.compile(
            r'^(\d{3}-\d{1,3}-\d{1,3})\s*$'  # Just the code on a line by itself
        )

        # Valid units
        valid_units = {'LS', 'EA', 'LF', 'SY', 'CY', 'SF', 'TON', 'GAL', 'AC', 'AS', 'GM'}

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check if this line is an FDOT code
            code_match = fdot_code_pattern.match(line)
            if code_match:
                fdot_code = code_match.group(1)

                # Only process drainage-related codes
                # 430-175 = pipes, 425-1 = inlets, 425-2 = manholes, 430-030/040 = endwalls, 430-982 = MES
                if not (fdot_code.startswith('430-') or fdot_code.startswith('425-')):
                    i += 1
                    continue

                # Look ahead for description, quantity, unit
                description = None
                quantity = None
                unit = None

                # Check next 1-4 lines for the components
                for j in range(1, 5):
                    if i + j >= len(lines):
                        break

                    next_line = lines[i + j].strip()
                    if not next_line:
                        continue

                    # Check if it's a unit
                    if next_line.upper() in valid_units:
                        unit = next_line.upper()
                        continue

                    # Check if it's a quantity (numeric, possibly with decimal)
                    try:
                        test_qty = float(next_line.replace(',', ''))
                        # If we already have a quantity and this looks like a bid item number, skip
                        if quantity is None and test_qty < 100000:
                            quantity = test_qty
                            continue
                    except ValueError:
                        pass

                    # Check if this is a description (not a number, not a unit, not another FDOT code)
                    if (description is None and
                        not fdot_code_pattern.match(next_line) and
                        not next_line.upper() in valid_units and
                        len(next_line) > 5):
                        # Skip if it looks like a bid item number (just digits)
                        if not re.match(r'^\d{1,3}$', next_line):
                            description = next_line

                # Validate we found all required components
                if description and quantity is not None and unit:
                    # Validate using gates
                    desc_result, desc_warnings = validate_description(
                        description,
                        pay_item_no=fdot_code
                    )
                    if desc_result is None:
                        logger.debug(f"Rejected multiline FDOT item {fdot_code}: invalid description '{description}'")
                        i += 1
                        continue

                    qty_result, qty_warnings = validate_quantity(
                        quantity, unit,
                        pay_item_no=fdot_code
                    )
                    if qty_result is None:
                        logger.debug(f"Rejected multiline FDOT item {fdot_code}: invalid quantity {quantity} {unit}")
                        i += 1
                        continue

                    item_key = f"fdot_ml_{fdot_code}_{quantity}"
                    if item_key in seen:
                        i += 1
                        continue
                    seen.add(item_key)

                    item = {
                        'pay_item_no': fdot_code,
                        'description': description,
                        'unit': unit,
                        'quantity': quantity,
                        'matched': False,
                        'unit_price': None,
                        'line_cost': None,
                        'source': 'fdot_multiline',
                        'confidence': 'high'
                    }

                    # Add validation warnings if any
                    if desc_warnings or qty_warnings:
                        item['validation_warnings'] = [
                            w.to_dict() for w in (desc_warnings + qty_warnings)
                            if w.severity.value == 'warning'
                        ]

                    # Attach source location if available
                    self._attach_source_location(item, f"{fdot_code} {description}")
                    items.append(item)

            i += 1

        return items

    def _extract_quantity_first_items(self, text: str, seen: set) -> List[Dict]:
        """
        Extract items in various quantity-first formats.

        Handles multiple common formats found in construction plans:
        - Federal/military: "51 LF 15" RCP CLASS V"
        - Size-first: "18" RCP - 51 LF"
        - Parenthetical: "18" RCP (51 LF)"
        - SDR pipe: "125 LF 6" PVC SDR 35"

        Uses validation to catch swapped size/quantity values.
        """
        items = []

        # Pattern definitions: (pattern, qty_group, unit_group, size_group, material_group, spec_group_or_none)
        # Using re.MULTILINE | re.IGNORECASE for all patterns
        patterns = [
            # Format 1: QTY UNIT SIZE" MATERIAL [CLASS/SDR SPEC]
            # Example: "51 LF 15" RCP CLASS V @ 0.5%"
            (r'(\d+(?:\.\d+)?)\s*(LF|EA|SF|SY|CY)\s+(\d+)\s*["\u201d\u2033]?\s*(RCP|PVC|HDPE|CMP|DIP|SRCP)(?:\s*(?:CLASS\s*)?([IVX]+))?',
             1, 2, 3, 4, 5),

            # Format 2: QTY UNIT SIZE" MATERIAL SDR ##
            # Example: "125 LF 6" PVC SDR 35"
            (r'(\d+(?:\.\d+)?)\s*(LF|EA)\s+(\d+)\s*["\u201d\u2033]?\s*(PVC|HDPE)\s*(?:SDR\s*)?(\d+)?',
             1, 2, 3, 4, 5),

            # Format 3: SIZE" MATERIAL - QTY UNIT (with dash separator)
            # Example: "18" RCP - 51 LF"
            (r'(\d+)\s*["\u201d\u2033]\s*(RCP|PVC|HDPE|CMP|DIP|SRCP)\s*[-\u2013\u2014]\s*(\d+(?:\.\d+)?)\s*(LF|EA|SF|SY)',
             3, 4, 1, 2, None),

            # Format 4: SIZE" MATERIAL (QTY UNIT) (parenthetical)
            # Example: "18" RCP (51 LF)"
            (r'(\d+)\s*["\u201d\u2033]\s*(RCP|PVC|HDPE|CMP|DIP|SRCP)\s*\(\s*(\d+(?:\.\d+)?)\s*(LF|EA|SF|SY)\s*\)',
             3, 4, 1, 2, None),

            # Format 5: QTY UNIT of SIZE" MATERIAL
            # Example: "51 LF of 18" RCP"
            (r'(\d+(?:\.\d+)?)\s*(LF|EA|SF|SY|CY)\s+(?:of\s+)?(\d+)\s*["\u201d\u2033]\s*(RCP|PVC|HDPE|CMP|DIP|SRCP)',
             1, 2, 3, 4, None),

            # Format 6: MATERIAL SIZE" @ QTY UNIT or MATERIAL SIZE" = QTY UNIT
            # Example: "RCP 18" @ 51 LF" or "RCP 18" = 51 LF"
            (r'(RCP|PVC|HDPE|CMP|DIP|SRCP)\s*(\d+)\s*["\u201d\u2033]?\s*[@=]\s*(\d+(?:\.\d+)?)\s*(LF|EA|SF|SY)',
             3, 4, 2, 1, None),

            # Format 7: Simple quantity with pipe - "51 LF RCP 18" or "51 LF 18 RCP"
            # Example: "51 LF RCP 18""
            (r'(\d+(?:\.\d+)?)\s*(LF|EA)\s+(RCP|PVC|HDPE|CMP|DIP|SRCP)\s*(\d+)\s*["\u201d\u2033]?',
             1, 2, 4, 3, None),

            # Format 8: SIZE" MATERIAL UNIT QTY (Vision API output format)
            # Example: "12" RCP LF 85" or "18" SRCP LF 200"
            (r'(\d+)\s*["\u201d\u2033]\s*(RCP|PVC|HDPE|CMP|DIP|SRCP)\s+(LF|EA|SF|SY)\s+(\d+(?:\.\d+)?)',
             4, 3, 1, 2, None),
        ]

        for pattern_tuple in patterns:
            pattern = pattern_tuple[0]
            qty_grp = pattern_tuple[1]
            unit_grp = pattern_tuple[2]
            size_grp = pattern_tuple[3]
            mat_grp = pattern_tuple[4]
            spec_grp = pattern_tuple[5]

            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                try:
                    raw_qty = float(match.group(qty_grp))
                    unit = match.group(unit_grp).upper()
                    raw_size = int(match.group(size_grp))
                    material = match.group(mat_grp).upper()
                    spec = ""
                    if spec_grp and len(match.groups()) >= spec_grp and match.group(spec_grp):
                        spec = match.group(spec_grp)

                    # VALIDATION: Check if size and quantity appear swapped
                    size, qty = self._validate_and_swap_size_qty(raw_size, raw_qty)

                    # Skip if size is still invalid after validation
                    if not self._is_valid_pipe_size(size):
                        # Log for debugging but don't skip - might be valid non-standard size
                        pass

                    # Build description
                    desc = f'{size}" {material}'
                    if spec:
                        if material == 'RCP' or material == 'SRCP':
                            desc += f' CLASS {spec}'
                        else:
                            desc += f' SDR {spec}'

                    # VALIDATION GATE: Quantity validation
                    qty_result, qty_warnings = validate_quantity(
                        qty, unit,
                        pay_item_no=f'{material}-{size}'
                    )
                    if qty_result is None:
                        logger.debug(f"Rejected qty-first item {material}-{size}: invalid quantity {qty} {unit}")
                        continue

                    item_key = f"qty_{material}_{size}_{qty}"
                    if item_key in seen:
                        continue
                    seen.add(item_key)

                    item = {
                        'pay_item_no': f'{material}-{size}',
                        'description': desc,
                        'unit': unit,
                        'quantity': qty,
                        'matched': False,
                        'unit_price': None,
                        'line_cost': None,
                        'source': 'quantity_first',
                        'confidence': 'medium'
                    }
                    
                    # Add validation warnings if any
                    if qty_warnings:
                        item['validation_warnings'] = [
                            w.to_dict() for w in qty_warnings
                            if w.severity.value == 'warning'
                        ]

                    # Attach source location if available
                    self._attach_source_location(item, match.group(0))
                    items.append(item)
                except (ValueError, IndexError):
                    # Skip malformed matches
                    continue

        return items

    def _extract_callout_items(self, text: str, seen: set) -> List[Dict]:
        """
        Extract pipe callouts like 'RCP 24' or '15" HDPE' from CAD annotations.

        NOTE: Callouts are annotations/labels on drawings, NOT actual quantities.
        Each unique callout type is recorded once with quantity=0 to flag it for
        manual verification. The 'mentions' field tracks how many times it appeared.

        VALIDATION: Only accepts sizes that match valid FDOT pipe diameters
        to avoid capturing quantities as sizes.
        """
        items = []
        callout_data = {}  # Track unique callouts and their mention counts

        patterns = [
            (r'\b(RCP|PVC|HDPE|CMP|DIP|SRCP)\s*-?\s*(\d+)\b', 1, 2),  # RCP 24, RCP-24
            (r'\b(\d+)["\s]*(RCP|PVC|HDPE|CMP|DIP|SRCP)\b', 2, 1),    # 24" RCP
        ]

        for pattern, mat_group, size_group in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                material = match.group(mat_group).upper()
                size_str = match.group(size_group)

                # VALIDATION: Only accept valid pipe sizes from catalog
                try:
                    size_int = int(size_str)
                    if not self._is_valid_pipe_size(size_int):
                        # Skip - likely a quantity, not a pipe size
                        continue
                except ValueError:
                    continue

                key = f"{material}_{size_str}"

                # Track unique callouts and count mentions (for reference only)
                if key not in callout_data:
                    callout_data[key] = {'material': material, 'size': size_str, 'mentions': 0}
                callout_data[key]['mentions'] += 1

        # Convert to items - each unique callout is ONE item needing verification
        for key, data in callout_data.items():
            item_key = f"callout_{key}"
            if item_key in seen:
                continue
            seen.add(item_key)

            item = {
                'pay_item_no': f"{data['material']}-{data['size']}",
                'description': f"{data['size']}\" {data['material']} PIPE (VERIFY QTY)",
                'unit': 'LF',  # Pipes are typically measured in LF, not EA
                'quantity': 0,  # Unknown - requires manual verification
                'matched': False,
                'unit_price': None,
                'line_cost': None,
                'source': 'callout',
                'confidence': 'low',
                'needs_verification': True,
                'mentions': data['mentions']  # How many times seen (for reference)
            }

            # Attach source location if available
            self._attach_source_location(item, f"{data['material']} {data['size']}")
            items.append(item)

        return items

    def _extract_drainage_labels(self, text: str, seen: set) -> List[Dict]:
        """
        Extract named drainage structures like 'STORM DRAIN MANHOLE #1'.

        NOTE: Only structures with explicit IDs (e.g., "#1", "-1") are counted as
        actual items. Generic mentions without IDs are recorded with quantity=0
        and flagged for manual verification to avoid inflating counts.
        """
        items = []
        structures_with_id = {}  # Structures with explicit IDs (reliable)
        generic_mentions = {}    # Generic mentions needing verification

        patterns = [
            # STORM DRAIN MANHOLE #1, SANITARY SEWER MANHOLE #2, MH-1
            r'(?:STORM\s*DRAIN|SANITARY\s*SEWER|SD|SS)?\s*(MANHOLE|MH)\s*[#-]?\s*(\d+)',
            # GRATE INLET #1, TYPE D INLET #2, INLET-1
            r'(?:GRATE|TYPE\s*[A-Z])?\s*(INLET)\s*[#-]?\s*(\d+)',
            # CATCH BASIN #1, CB-1
            r'(CATCH\s*BASIN|CB)\s*[#-]?\s*(\d+)',
            # JUNCTION BOX #1, JB-1
            r'(JUNCTION\s*BOX|JB)\s*[#-]?\s*(\d+)',
        ]

        # Patterns for generic mentions (without IDs)
        generic_patterns = [
            (r'\b(MANHOLE|MH)\b(?!\s*[#-]?\s*\d)', 'MANHOLE'),
            (r'\b(INLET)\b(?!\s*[#-]?\s*\d)', 'INLET'),
            (r'\b(CATCH\s*BASIN|CB)\b(?!\s*[#-]?\s*\d)', 'CATCH BASIN'),
            (r'\b(JUNCTION\s*BOX|JB)\b(?!\s*[#-]?\s*\d)', 'JUNCTION BOX'),
        ]

        # Extract structures with explicit IDs
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                struct_type = match.group(1).upper()
                struct_id = match.group(2)

                # Normalize structure type
                if struct_type in ('MH', 'MANHOLE'):
                    struct_type = 'MANHOLE'
                elif struct_type in ('CB', 'CATCH BASIN'):
                    struct_type = 'CATCH BASIN'
                elif struct_type in ('JB', 'JUNCTION BOX'):
                    struct_type = 'JUNCTION BOX'

                # Track unique structures by ID
                key = f"{struct_type}_{struct_id}"
                structures_with_id[key] = {
                    'type': struct_type,
                    'id': struct_id
                }

        # Track generic mentions (for reference, not as quantities)
        for pattern, struct_type in generic_patterns:
            mentions = len(re.findall(pattern, text, re.IGNORECASE))
            if mentions > 0:
                if struct_type not in generic_mentions:
                    generic_mentions[struct_type] = 0
                generic_mentions[struct_type] += mentions

        # Convert structures with IDs to items (reliable quantities)
        for key, data in structures_with_id.items():
            item_key = f"struct_{key}"
            if item_key in seen:
                continue
            seen.add(item_key)

            item = {
                'pay_item_no': data['type'],
                'description': f"{data['type']} #{data['id']}",
                'unit': 'EA',
                'quantity': 1,  # Each unique ID = 1 structure
                'matched': False,
                'unit_price': None,
                'line_cost': None,
                'source': 'drainage_label',
                'confidence': 'high'  # High confidence when ID is present
            }

            # Attach source location if available
            self._attach_source_location(item, f"{data['type']} #{data['id']}")
            items.append(item)

        # Record generic mentions needing verification (if not already counted via IDs)
        for struct_type, mention_count in generic_mentions.items():
            # Only add if we didn't find any with explicit IDs
            has_ids = any(d['type'] == struct_type for d in structures_with_id.values())
            if has_ids:
                continue  # Skip - we already have specific structures

            item_key = f"struct_generic_{struct_type}"
            if item_key in seen:
                continue
            seen.add(item_key)

            item = {
                'pay_item_no': struct_type,
                'description': f"{struct_type} (VERIFY QTY)",
                'unit': 'EA',
                'quantity': 0,  # Unknown - requires manual verification
                'matched': False,
                'unit_price': None,
                'line_cost': None,
                'source': 'drainage_label',
                'confidence': 'low',  # Low confidence for generic mentions
                'needs_verification': True,
                'mentions': mention_count  # How many times mentioned (for reference)
            }

            # Attach source location if available
            self._attach_source_location(item, struct_type)
            items.append(item)

        return items

    def _is_valid_elliptical_size(self, rise: int, span: int) -> bool:
        """Check if rise x span is a valid elliptical pipe size.

        Checks both orientations since dimensions may be swapped in OCR text.
        """
        return (rise, span) in VALID_ELLIPTICAL_SIZES or (span, rise) in VALID_ELLIPTICAL_SIZES

    def _extract_elliptical_pipe_items(self, text: str, seen: set) -> List[Dict]:
        """
        Extract elliptical pipe items from text.

        Elliptical pipes have two-dimensional sizes like:
        - 14"x23" RCP HE (Horizontal Elliptical)
        - 14 X 23 ERCP
        - ERCP 14x23
        - 19"X30" ELLIPTICAL RCP CLASS III

        Args:
            text: Page text to search
            seen: Set of already-seen items

        Returns:
            List of extracted elliptical pipe items
        """
        items = []

        # Patterns for elliptical pipe detection
        # Format variations: 14"x23", 14x23, 14" x 23", 14 X 23
        patterns = [
            # QTY UNIT SIZE ERCP/HE - e.g., "51 LF 14"x23" RCP HE"
            (r'(\d+(?:\.\d+)?)\s*(LF|EA)\s+(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?\s*(?:RCP|ERCP)?\s*(?:HE|ELLIP(?:TICAL)?)?(?:\s*(?:CL(?:ASS)?\s*)?([IVX]+))?',
             1, 2, 3, 4, 5),

            # SIZE ERCP/HE - QTY UNIT - e.g., "14"x23" ERCP - 51 LF"
            (r'(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?\s*(?:RCP\s*HE|ERCP|ELLIP(?:TICAL)?\s*RCP)\s*[-\u2013\u2014]\s*(\d+(?:\.\d+)?)\s*(LF|EA)',
             3, 4, 1, 2, None),

            # ERCP SIZE - e.g., "ERCP 14x23" or "RCP HE 14"x23""
            (r'(?:ERCP|RCP\s*HE|ELLIP(?:TICAL)?\s*RCP)\s*(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?',
             None, None, 1, 2, None),

            # SIZE ERCP with class - e.g., "14"X23" RCP HE CLASS III"
            (r'(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?\s*(?:RCP\s*HE|ERCP|ELLIP(?:TICAL)?\s*RCP)(?:\s*(?:CL(?:ASS)?\s*)?([IVX]+))?',
             None, None, 1, 2, 3),
        ]

        for pattern_tuple in patterns:
            pattern = pattern_tuple[0]
            qty_grp = pattern_tuple[1]
            unit_grp = pattern_tuple[2]
            rise_grp = pattern_tuple[3]
            span_grp = pattern_tuple[4]
            class_grp = pattern_tuple[5]

            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                try:
                    rise = int(match.group(rise_grp))
                    span = int(match.group(span_grp))

                    # Validate elliptical size
                    if not self._is_valid_elliptical_size(rise, span):
                        continue

                    # Get quantity and unit if available
                    if qty_grp and match.group(qty_grp):
                        qty = float(match.group(qty_grp))
                        unit = match.group(unit_grp).upper() if unit_grp else 'LF'
                    else:
                        qty = 0  # Callout only, needs verification
                        unit = 'LF'

                    # Get class if available
                    pipe_class = ""
                    if class_grp and len(match.groups()) >= class_grp and match.group(class_grp):
                        pipe_class = match.group(class_grp)

                    # Build description
                    size_str = f'{rise}"x{span}"'
                    desc = f'{size_str} ERCP'
                    if pipe_class:
                        desc += f' CLASS {pipe_class}'

                    item_key = f"ellip_{rise}x{span}_{qty}"
                    if item_key in seen:
                        continue
                    seen.add(item_key)

                    item = {
                        'pay_item_no': f'ERCP-{rise}x{span}',
                        'description': desc,
                        'unit': unit,
                        'quantity': qty,
                        'matched': False,
                        'unit_price': None,
                        'line_cost': None,
                        'source': 'elliptical_pipe',
                        'confidence': 'medium' if qty > 0 else 'low',
                        'pipe_type': 'elliptical',
                        'rise': rise,
                        'span': span,
                        'needs_verification': qty == 0
                    }

                    # Attach source location if available
                    self._attach_source_location(item, match.group(0))
                    items.append(item)
                except (ValueError, IndexError):
                    continue

        return items

    def _extract_elliptical_accessories(self, text: str, seen: set) -> List[Dict]:
        """
        Extract elliptical pipe accessories: Flared Ends and MES.

        These are separate from regular elliptical pipe detection because they're
        accessories/end treatments, not the pipe itself.

        Formats detected:
        - 14"x23" FLARED END, FLARED END 14x23, 14X23 FE
        - 14"x23" MES, MES 14x23, 14X23 MES 4:1

        Args:
            text: Page text to search
            seen: Set of already-seen items

        Returns:
            List of extracted elliptical accessory items
        """
        items = []

        # Patterns for elliptical FLARED END
        flared_patterns = [
            # SIZE FLARED END - e.g., "14"x23" FLARED END" or "14x23 FE"
            (r'(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?\s*(?:FLARED\s*END(?:\s*SECTION)?|FE|FES)',
             1, 2),
            # FLARED END SIZE - e.g., "FLARED END 14x23" or "FE 14"x23""
            (r'(?:FLARED\s*END(?:\s*SECTION)?|FE|FES)\s*(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?',
             1, 2),
        ]

        # Patterns for elliptical MES (concrete, not galvanized)
        mes_patterns = [
            # SIZE MES - e.g., "14"x23" MES 4:1" or "14x23 MES"
            (r'(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?\s*MES(?:\s*4:1)?',
             1, 2),
            # MES SIZE - e.g., "MES 14x23" or "MES 14"x23" 4:1"
            (r'MES\s*(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?(?:\s*4:1)?',
             1, 2),
            # MITERED END SECTION SIZE - e.g., "MITERED END SECTION 14x23"
            (r'MITERED\s*END(?:\s*SECTION)?\s*(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?',
             1, 2),
        ]

        # Process FLARED END patterns
        for pattern, rise_grp, span_grp in flared_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                try:
                    rise = int(match.group(rise_grp))
                    span = int(match.group(span_grp))

                    # Validate elliptical size
                    if not self._is_valid_elliptical_size(rise, span):
                        continue

                    size_str = f'{rise}"x{span}"'
                    item_key = f"ellip_flared_{rise}x{span}"

                    if item_key in seen:
                        continue
                    seen.add(item_key)

                    item = {
                        'pay_item_no': f'ERCP-FE-{rise}x{span}',
                        'description': f'{size_str} ELLIPTICAL FLARED END',
                        'unit': 'EA',
                        'quantity': 1,
                        'matched': False,
                        'unit_price': None,
                        'line_cost': None,
                        'source': 'elliptical_accessory',
                        'confidence': 'medium',
                        'structure_type': 'FLARED_END_ELLIPTICAL',
                        'rise': rise,
                        'span': span
                    }

                    # Attach source location if available
                    self._attach_source_location(item, match.group(0))
                    items.append(item)
                except (ValueError, IndexError):
                    continue

        # Process MES patterns (but avoid galvanized - those handled separately)
        for pattern, rise_grp, span_grp in mes_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                try:
                    # Skip if this looks like a galvanized MES (has GALV nearby)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].upper()
                    if 'GALV' in context or 'STEEL' in context:
                        continue

                    rise = int(match.group(rise_grp))
                    span = int(match.group(span_grp))

                    # Validate elliptical size
                    if not self._is_valid_elliptical_size(rise, span):
                        continue

                    size_str = f'{rise}"x{span}"'
                    item_key = f"ellip_mes_{rise}x{span}"

                    if item_key in seen:
                        continue
                    seen.add(item_key)

                    item = {
                        'pay_item_no': f'ERCP-MES-{rise}x{span}',
                        'description': f'{size_str} ELLIPTICAL MES 4:1',
                        'unit': 'EA',
                        'quantity': 1,
                        'matched': False,
                        'unit_price': None,
                        'line_cost': None,
                        'source': 'elliptical_accessory',
                        'confidence': 'medium',
                        'structure_type': 'MES_ELLIPTICAL',
                        'rise': rise,
                        'span': span
                    }

                    # Attach source location if available
                    self._attach_source_location(item, match.group(0))
                    items.append(item)
                except (ValueError, IndexError):
                    continue

        return items

    def _extract_structure_items(self, text: str, seen: set) -> List[Dict]:
        """
        Extract endwall and end section structures from text.

        Types:
        - Straight Endwalls (single, double, triple, quad)
        - Winged Endwalls (45 degree, U-type)
        - U-Type Endwalls
        - MES (Mitered End Sections)
        - Flared Ends

        Args:
            text: Page text to search
            seen: Set of already-seen items

        Returns:
            List of extracted structure items
        """
        items = []

        # Patterns for structure detection
        patterns = [
            # STRAIGHT ENDWALL with size and configuration
            # e.g., "18" STRAIGHT ENDWALL SINGLE" or "STRAIGHT CONCRETE ENDWALL 24" DOUBLE"
            (r'(\d+)\s*["\u201d]?\s*STRAIGHT\s*(?:CONCRETE\s*)?END\s*WALL[S]?\s*(SINGLE|DOUBLE|TRIPLE|QUAD)?',
             'STRAIGHT_ENDWALL', 1, 2),
            (r'STRAIGHT\s*(?:CONCRETE\s*)?END\s*WALL[S]?\s*(\d+)\s*["\u201d]?\s*(SINGLE|DOUBLE|TRIPLE|QUAD)?',
             'STRAIGHT_ENDWALL', 1, 2),

            # WINGED ENDWALL
            # e.g., "18" WINGED ENDWALL 45 DEG" or "WING WALL 24" U-TYPE"
            (r'(\d+)\s*["\u201d]?\s*(?:WINGED?|WING)\s*(?:END\s*)?WALL[S]?\s*(45\s*DEG(?:REE)?|U[-\s]?TYPE)?',
             'WINGED_ENDWALL', 1, 2),
            (r'(?:WINGED?|WING)\s*(?:END\s*)?WALL[S]?\s*(\d+)\s*["\u201d]?\s*(45\s*DEG(?:REE)?|U[-\s]?TYPE)?',
             'WINGED_ENDWALL', 1, 2),

            # U-TYPE ENDWALL with slope ratios (2:1, 3:1, 4:1, 6:1)
            # e.g., "U-TYPE ENDWALL 18"", "18" U-TYPE 2:1", "U-TYPE 24" 4:1 WITH GRATE"
            # Note: Longer alternatives must come first in regex alternation
            (r'(\d+)\s*["\u201d]?\s*U[-\s]?TYPE\s*(?:END\s*WALL)?\s*([2-6]:1)?(?:\s*(?:WITH\s*)?(GRATE\s*AND\s*BAFFLES|GRATE|BAFFLES))?',
             'U_TYPE_ENDWALL', 1, 2, 3),
            (r'U[-\s]?TYPE\s*(?:END\s*WALL)?\s*(\d+)\s*["\u201d]?\s*([2-6]:1)?(?:\s*(?:WITH\s*)?(GRATE\s*AND\s*BAFFLES|GRATE|BAFFLES))?',
             'U_TYPE_ENDWALL', 1, 2, 3),

            # PIPE CRADLE
            # e.g., "18" PIPE CRADLE", "PIPE CRADLE 24"", "24" CRADLE"
            (r'(\d+)\s*["\u201d]?\s*(?:PIPE\s*)?CRADLE',
             'PIPE_CRADLE', 1, None, None),
            (r'(?:PIPE\s*)?CRADLE\s*(\d+)\s*["\u201d]?',
             'PIPE_CRADLE', 1, None, None),

            # MES (Mitered End Section) for round pipe
            # e.g., "18" MES 4:1" or "MES 24"" or "MITERED END SECTION 18""
            (r'(\d+)\s*["\u201d]?\s*MES(?:\s*4:1)?',
             'MES', 1, None),
            (r'MES(?:\s*4:1)?\s*(\d+)\s*["\u201d]?',
             'MES', 1, None),
            (r'MITERED\s*END(?:\s*SECTION)?\s*(\d+)\s*["\u201d]?',
             'MES', 1, None),
            (r'(\d+)\s*["\u201d]?\s*MITERED\s*END(?:\s*SECTION)?',
             'MES', 1, None),

            # MES for elliptical pipe
            # e.g., "14"x23" MES" or "MES 14x23"
            (r'(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?\s*MES',
             'MES_ELLIPTICAL', 1, 2),
            (r'MES\s*(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?',
             'MES_ELLIPTICAL', 1, 2),

            # FLARED END
            # e.g., "FLARED END 18"" or "18" FLARED END SECTION"
            (r'(\d+)\s*["\u201d]?\s*FLARED\s*END(?:\s*SECTION)?',
             'FLARED_END', 1, None),
            (r'FLARED\s*END(?:\s*SECTION)?\s*(\d+)\s*["\u201d]?',
             'FLARED_END', 1, None),
        ]

        for pattern_tuple in patterns:
            pattern = pattern_tuple[0]
            struct_type = pattern_tuple[1]
            size_grp = pattern_tuple[2]
            config_grp = pattern_tuple[3]
            # Handle extended tuple format (5 elements) for U-TYPE with slope/option
            option_grp = pattern_tuple[4] if len(pattern_tuple) > 4 else None

            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                try:
                    # Handle elliptical MES separately (has rise x span)
                    if struct_type == 'MES_ELLIPTICAL':
                        rise = int(match.group(size_grp))
                        span = int(match.group(config_grp))
                        if not self._is_valid_elliptical_size(rise, span):
                            continue
                        size_str = f'{rise}"x{span}"'
                        config = ''
                        option = ''
                        item_key = f"struct_MES_{rise}x{span}"
                    else:
                        # Round pipe structures
                        size = int(match.group(size_grp))
                        if not self._is_valid_pipe_size(size):
                            continue
                        size_str = f'{size}"'

                        # Get configuration if available
                        config = ''
                        option = ''

                        if config_grp and len(match.groups()) >= config_grp and match.group(config_grp):
                            config = match.group(config_grp).upper()
                            # Normalize configuration - be specific to avoid false matches
                            if '45' in config and 'DEG' in config:
                                config = '45 DEGREE'
                            elif config.startswith('U') and ('TYPE' in config or config == 'U'):
                                config = 'U-TYPE'
                            # Keep SINGLE, DOUBLE, TRIPLE, QUAD, slope ratios (2:1, 3:1, etc.) as-is

                        # Get option (grate, baffles) for U-TYPE endwalls
                        if option_grp and len(match.groups()) >= option_grp and match.group(option_grp):
                            option = match.group(option_grp).upper()
                            # Normalize options
                            if 'GRATE' in option and 'BAFFLE' in option:
                                option = 'WITH GRATE AND BAFFLES'
                            elif 'GRATE' in option:
                                option = 'WITH GRATE'
                            elif 'BAFFLE' in option:
                                option = 'WITH BAFFLES'

                        item_key = f"struct_{struct_type}_{size}_{config}_{option}"

                    if item_key in seen:
                        continue
                    seen.add(item_key)

                    # Build description
                    type_name = struct_type.replace('_', ' ')
                    if struct_type == 'MES_ELLIPTICAL':
                        type_name = 'MES (ELLIPTICAL)'
                    elif struct_type == 'PIPE_CRADLE':
                        type_name = 'PIPE CRADLE'

                    desc = f'{size_str} {type_name}'
                    if config:
                        desc += f' {config}'
                    if option:
                        desc += f' {option}'

                    item = {
                        'pay_item_no': f'{struct_type}-{size_str}',
                        'description': desc,
                        'unit': 'EA',
                        'quantity': 1,  # Structures are typically counted
                        'matched': False,
                        'unit_price': None,
                        'line_cost': None,
                        'source': 'structure',
                        'confidence': 'medium',
                        'structure_type': struct_type,
                        'configuration': config if config else None,
                        'option': option if option else None
                    }

                    # Attach source location if available
                    self._attach_source_location(item, match.group(0))
                    items.append(item)
                except (ValueError, IndexError):
                    continue

        return items

    def _extract_galvanized_mes_items(self, text: str, seen: set) -> List[Dict]:
        """
        Extract Galvanized Steel 4:1 MES items.

        These are distinct from concrete MES and come in both round and elliptical
        sizes with various configurations:
        - Run configurations: Single Run, Double Run, Triple Run
        - Frame options: With Frame, No Frame

        Formats detected:
        - 18" GALVANIZED MES SINGLE RUN WITH FRAME
        - GALV STEEL MES 14x23 DOUBLE RUN NO FRAME
        - 24" GALV MES
        - GALVANIZED STEEL 4:1 MES 19X30

        Args:
            text: Page text to search
            seen: Set of already-seen items

        Returns:
            List of extracted galvanized MES items
        """
        items = []

        # Patterns for ROUND pipe Galvanized MES
        round_patterns = [
            # SIZE GALV MES [RUN CONFIG] [FRAME] - e.g., "18" GALVANIZED MES SINGLE RUN WITH FRAME"
            (r'(\d+)\s*["\u201d]?\s*(?:GALV(?:ANIZED)?(?:\s*STEEL)?)\s*(?:4:1\s*)?MES(?:\s*(SINGLE|DOUBLE|TRIPLE)\s*RUN)?(?:\s*(WITH|NO)\s*FRAME)?',
             1, 2, 3),
            # GALV MES SIZE [RUN CONFIG] [FRAME] - e.g., "GALV MES 24" DOUBLE RUN NO FRAME"
            (r'(?:GALV(?:ANIZED)?(?:\s*STEEL)?)\s*(?:4:1\s*)?MES\s*(\d+)\s*["\u201d]?(?:\s*(SINGLE|DOUBLE|TRIPLE)\s*RUN)?(?:\s*(WITH|NO)\s*FRAME)?',
             1, 2, 3),
        ]

        # Patterns for ELLIPTICAL pipe Galvanized MES
        elliptical_patterns = [
            # SIZE GALV MES [RUN CONFIG] [FRAME] - e.g., "14x23 GALVANIZED MES SINGLE RUN"
            (r'(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?\s*(?:GALV(?:ANIZED)?(?:\s*STEEL)?)\s*(?:4:1\s*)?MES(?:\s*(SINGLE|DOUBLE|TRIPLE)\s*RUN)?(?:\s*(WITH|NO)\s*FRAME)?',
             1, 2, 3, 4),
            # GALV MES SIZE [RUN CONFIG] [FRAME] - e.g., "GALV STEEL MES 14 X 23 DOUBLE RUN"
            (r'(?:GALV(?:ANIZED)?(?:\s*STEEL)?)\s*(?:4:1\s*)?MES\s*(\d+)\s*["\u201d]?\s*[xX×]\s*(\d+)\s*["\u201d]?(?:\s*(SINGLE|DOUBLE|TRIPLE)\s*RUN)?(?:\s*(WITH|NO)\s*FRAME)?',
             1, 2, 3, 4),
        ]

        # Process ROUND patterns
        for pattern, size_grp, run_grp, frame_grp in round_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                try:
                    size = int(match.group(size_grp))
                    if not self._is_valid_pipe_size(size):
                        continue

                    # Get run configuration
                    run_config = ''
                    if run_grp and len(match.groups()) >= run_grp and match.group(run_grp):
                        run_config = f'{match.group(run_grp).upper()} RUN'
                    else:
                        run_config = 'SINGLE RUN'  # Default

                    # Get frame option
                    frame_option = ''
                    if frame_grp and len(match.groups()) >= frame_grp and match.group(frame_grp):
                        frame_option = f'{match.group(frame_grp).upper()} FRAME'
                    else:
                        frame_option = 'NO FRAME'  # Default

                    size_str = f'{size}"'
                    item_key = f"galv_mes_{size}_{run_config}_{frame_option}"

                    if item_key in seen:
                        continue
                    seen.add(item_key)

                    desc = f'{size_str} GALVANIZED STEEL 4:1 MES {run_config} {frame_option}'

                    item = {
                        'pay_item_no': f'GALV-MES-{size}',
                        'description': desc,
                        'unit': 'EA',
                        'quantity': 1,
                        'matched': False,
                        'unit_price': None,
                        'line_cost': None,
                        'source': 'galvanized_mes',
                        'confidence': 'medium',
                        'structure_type': 'GALVANIZED_MES_ROUND',
                        'run_config': run_config,
                        'frame_option': frame_option
                    }

                    # Attach source location if available
                    self._attach_source_location(item, match.group(0))
                    items.append(item)
                except (ValueError, IndexError):
                    continue

        # Process ELLIPTICAL patterns
        for pattern, rise_grp, span_grp, run_grp, frame_grp in elliptical_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                try:
                    rise = int(match.group(rise_grp))
                    span = int(match.group(span_grp))
                    if not self._is_valid_elliptical_size(rise, span):
                        continue

                    # Get run configuration
                    run_config = ''
                    if run_grp and len(match.groups()) >= run_grp and match.group(run_grp):
                        run_config = f'{match.group(run_grp).upper()} RUN'
                    else:
                        run_config = 'SINGLE RUN'  # Default

                    # Get frame option
                    frame_option = ''
                    if frame_grp and len(match.groups()) >= frame_grp and match.group(frame_grp):
                        frame_option = f'{match.group(frame_grp).upper()} FRAME'
                    else:
                        frame_option = 'NO FRAME'  # Default

                    size_str = f'{rise}x{span}'
                    item_key = f"galv_mes_{rise}x{span}_{run_config}_{frame_option}"

                    if item_key in seen:
                        continue
                    seen.add(item_key)

                    desc = f'{rise}"x{span}" GALVANIZED STEEL 4:1 MES {run_config} {frame_option}'

                    item = {
                        'pay_item_no': f'GALV-MES-{rise}x{span}',
                        'description': desc,
                        'unit': 'EA',
                        'quantity': 1,
                        'matched': False,
                        'unit_price': None,
                        'line_cost': None,
                        'source': 'galvanized_mes',
                        'confidence': 'medium',
                        'structure_type': 'GALVANIZED_MES_ELLIPTICAL',
                        'rise': rise,
                        'span': span,
                        'run_config': run_config,
                        'frame_option': frame_option
                    }

                    # Attach source location if available
                    self._attach_source_location(item, match.group(0))
                    items.append(item)
                except (ValueError, IndexError):
                    continue

        return items

    def _extract_table_items(self, text: str, seen: set) -> List[Dict]:
        """
        Extract items from tabular data (pipe schedules, quantity takeoff tables).

        Handles table formats like:
        - Qty | Unit | Size | Material
        - Description | Size | Qty | Unit

        Tables typically use multiple spaces or tabs as delimiters.

        Args:
            text: Page text to search
            seen: Set of already-seen items
        """
        items = []

        # Split text into lines
        lines = text.split('\n')

        # Track consecutive table-like lines
        table_buffer = []

        for line in lines:
            # Check if line looks like a table row (has multiple delimiters)
            cells = re.split(r'\s{2,}|\t', line.strip())
            cells = [c.strip() for c in cells if c.strip()]

            if len(cells) >= 3:
                table_buffer.append(cells)
            else:
                # Process any accumulated table data
                if len(table_buffer) >= 2:
                    items.extend(self._parse_table_rows(table_buffer, seen))
                table_buffer = []

        # Don't forget last table
        if len(table_buffer) >= 2:
            items.extend(self._parse_table_rows(table_buffer, seen))

        return items

    def _parse_table_rows(self, rows: List[List[str]], seen: set) -> List[Dict]:
        """
        Parse table rows to extract pay items.

        Tries to identify columns containing:
        - Quantity (numbers)
        - Unit (LF, EA, etc.)
        - Size (pipe diameter)
        - Material (RCP, PVC, etc.)

        Args:
            rows: List of table rows (each row is a list of cell values)
            seen: Set of already-seen items
        """
        items = []

        # Material and unit keywords
        materials = {'RCP', 'PVC', 'HDPE', 'CMP', 'DIP', 'SRCP', 'ERCP', 'ADS'}
        units = {'LF', 'EA', 'SF', 'SY', 'CY', 'TON', 'GAL'}

        for row in rows:
            qty = None
            unit = None
            size = None
            material = None
            description_parts = []

            for cell in row:
                cell_upper = cell.upper().strip()

                # Check for material
                for mat in materials:
                    if mat in cell_upper:
                        material = mat
                        break

                # Check for unit
                for u in units:
                    if cell_upper == u or cell_upper.endswith(f' {u}'):
                        unit = u
                        break

                # Check for size (number followed by " or number in typical pipe sizes)
                size_match = re.search(r'^(\d{1,2})\s*["\u201d\u2033]?\s*$', cell)
                if size_match:
                    potential_size = int(size_match.group(1))
                    if self._is_valid_pipe_size(potential_size):
                        size = potential_size

                # Check for quantity (number, possibly with decimal)
                qty_match = re.match(r'^(\d+(?:\.\d+)?)\s*$', cell)
                if qty_match and not size_match:
                    potential_qty = float(qty_match.group(1))
                    # Quantities are typically > 1 and not standard pipe sizes
                    # Unless they're EA counts which can be 1-10
                    if potential_qty > 0:
                        qty = potential_qty

                # Add to description if it has text
                if re.search(r'[A-Za-z]', cell) and cell_upper not in units:
                    description_parts.append(cell)

            # Validate we have enough info to create an item
            if material and qty is not None and qty > 0:
                # Apply size/qty validation if we have size
                if size:
                    size, qty = self._validate_and_swap_size_qty(size, qty)
                else:
                    # Try to extract size from description
                    for part in description_parts:
                        size_in_desc = re.search(r'(\d{1,2})\s*["\u201d\u2033]', part)
                        if size_in_desc:
                            potential_size = int(size_in_desc.group(1))
                            if self._is_valid_pipe_size(potential_size):
                                size = potential_size
                                break

                # Build description
                desc = ' '.join(description_parts) if description_parts else f'{size}" {material}' if size else material

                item_key = f"table_{material}_{size}_{qty}"
                if item_key in seen:
                    continue
                seen.add(item_key)

                item = {
                    'pay_item_no': f'{material}-{size}' if size else material,
                    'description': desc[:100] + ('...' if len(desc) > 100 else ''),
                    'unit': unit or 'LF',  # Default to LF for pipes
                    'quantity': qty,
                    'matched': False,
                    'unit_price': None,
                    'line_cost': None,
                    'source': 'table',
                    'confidence': 'medium'
                }

                # Attach source location if available (using row content as match text)
                row_text = ' '.join(row)
                self._attach_source_location(item, row_text)
                items.append(item)

        return items

    def _normalize_size(self, size_str: str) -> str:
        """Normalize size string to standard format (e.g., '15"')."""
        if not size_str:
            return ''
        # Extract numeric part
        match = re.search(r'(\d+(?:\.\d+)?)', str(size_str))
        if match:
            return f'{match.group(1)}"'
        return ''

    def _is_valid_pipe_size(self, size: int) -> bool:
        """Check if size is a valid catalog pipe diameter."""
        return size in VALID_PIPE_SIZES

    def _validate_and_swap_size_qty(self, size: int, qty: float) -> Tuple[int, float]:
        """
        Validate size/quantity and swap if they appear reversed.

        Common OCR/parsing error: size and quantity get swapped because:
        - "51 LF 18" RCP" could be parsed as size=51, qty=18 (wrong)
        - Should be size=18, qty=51

        Rules:
        1. If size is not in catalog but qty is a valid size → swap
        2. If size > MAX_PIPE_SIZE (96"), it's likely the quantity → swap
        3. Otherwise keep as-is

        Args:
            size: Extracted pipe size (inches)
            qty: Extracted quantity

        Returns:
            Tuple of (corrected_size, corrected_qty)
        """
        qty_int = int(qty) if qty == int(qty) else None

        # Rule 1: If size not in catalog but qty is a valid pipe size, swap
        if size not in VALID_PIPE_SIZES and qty_int in VALID_PIPE_SIZES:
            return qty_int, float(size)

        # Rule 2: If size is impossibly large (>96"), it's probably the quantity
        if size > MAX_PIPE_SIZE and qty <= MAX_PIPE_SIZE:
            # Only swap if qty looks like a valid pipe size
            if qty_int in VALID_PIPE_SIZES:
                return qty_int, float(size)

        # Keep as-is
        return size, qty

    def _extract_fdot_prefix(self, pay_item_no: str) -> Optional[str]:
        """
        Extract FDOT code prefix for matching.
        e.g., "430-175-118" -> try "430-175-118", then "430-175", then "430"
        """
        if not pay_item_no:
            return None
        # Clean up the pay item number
        code = pay_item_no.strip().upper()
        return code

    def match_to_price_list(self, pay_item: Dict) -> Optional[Dict]:
        """
        Match a pay item to the price list.

        Matching priority:
        1. FDOT code exact match (e.g., "430-030")
        2. FDOT code prefix match (e.g., "430-030" from "430-030-118")
        3. Size + product type keyword match (fallback)
        """
        pay_item_no = pay_item.get('pay_item_no', '')
        desc = pay_item.get('description', '').lower()

        # Extract and normalize size from description
        size_match = re.search(r'(\d{1,2})["\s]*(?:inch)?', desc)
        extracted_size = f'{size_match.group(1)}"' if size_match else ''

        # Strategy 1: Try FDOT code matching
        if pay_item_no and self.price_list_by_fdot:
            # Try exact match first
            if pay_item_no in self.price_list_by_fdot:
                candidates = self.price_list_by_fdot[pay_item_no]
                return self._select_best_candidate(candidates, extracted_size, desc)

            # Try prefix match (e.g., "430-030" from "430-030-118")
            parts = pay_item_no.split('-')
            for i in range(len(parts), 0, -1):
                prefix = '-'.join(parts[:i])
                if prefix in self.price_list_by_fdot:
                    candidates = self.price_list_by_fdot[prefix]
                    return self._select_best_candidate(candidates, extracted_size, desc)

        # Strategy 2: Fallback to size + keyword matching
        for key, product in self.price_list.items():
            product_size = self._normalize_size(product.get('size', ''))

            if extracted_size and extracted_size == product_size:
                # Check product type keywords
                product_type = product.get('product_type', '').lower()
                if 'endwall' in desc and 'endwall' in product_type:
                    return product
                if ('rcp' in desc or 'pipe' in desc or 'culvert' in desc) and 'pipe' in product_type:
                    return product
                if 'inlet' in desc and 'inlet' in product_type:
                    return product
                if 'manhole' in desc and 'manhole' in product_type:
                    return product
                if 'mes' in desc and 'mes' in product_type:
                    return product

        # Strategy 3: Fuzzy matching fallback
        fuzzy_match = self._fuzzy_match_product(desc, extracted_size)
        if fuzzy_match:
            return fuzzy_match

        # Strategy 4: Structure matcher for inlets, manholes, junction boxes, catch basins
        if STRUCTURE_MATCHER_AVAILABLE:
            structure_match = self._match_structure(pay_item)
            if structure_match:
                return structure_match

        return None
    
    def _match_structure(self, pay_item: Dict) -> Optional[Dict]:
        """
        Match structures (inlets, manholes, etc.) using the structure_matcher module.
        
        This handles the naming mismatch between Vision extraction and FDOT codes:
        - Vision: "TYPE D INLET (MODIFIED)", "MANHOLE #1"
        - FDOT: "425-1-549", "425-2-41"
        
        Args:
            pay_item: Pay item dictionary with description
            
        Returns:
            Matched product dictionary or None
        """
        if not STRUCTURE_MATCHER_AVAILABLE:
            return None
            
        desc = pay_item.get('description', '')
        
        # Only try structure matching for structure-type items
        desc_upper = desc.upper()
        if not any(kw in desc_upper for kw in ['INLET', 'MANHOLE', 'MH', 'JUNCTION', 'CATCH BASIN', 'CB']):
            return None
        
        # Try to match using structure_matcher
        result = match_structure(desc)
        if result and result.confidence >= 0.6:
            return {
                'product_type': result.description,
                'size': result.subtype,
                'configuration': 'Standard',
                'price': result.unit_price,
                'unit': result.unit,
                'fdot_code': result.fdot_code,
                'match_source': 'structure_matcher',
                'match_confidence': result.confidence,
                'match_reason': result.match_reason
            }
        
        return None

    def _fuzzy_match_product(self, description: str, size: str) -> Optional[Dict]:
        """
        Fuzzy match a description to products in the price list.

        Uses token-based similarity scoring to find the best match.
        Only returns a match if the score exceeds a confidence threshold.
        """
        if not self.price_list:
            return None

        # Tokenize and normalize the description
        desc_tokens = self._tokenize(description)
        if not desc_tokens:
            return None

        best_match = None
        best_score = 0
        min_score_threshold = 0.4  # Minimum 40% token match required

        for key, product in self.price_list.items():
            score = 0
            product_type = product.get('product_type', '')
            product_size = self._normalize_size(product.get('size', ''))
            config = product.get('configuration', '')

            # Tokenize product info
            product_tokens = self._tokenize(f"{product_type} {config}")

            # Calculate token overlap score (guard against empty token sets)
            if product_tokens and desc_tokens:
                common_tokens = desc_tokens & product_tokens
                max_tokens = max(len(desc_tokens), len(product_tokens))
                if max_tokens > 0:
                    token_score = len(common_tokens) / max_tokens
                    score += token_score * 60  # Up to 60 points for token match

            # Size match bonus
            if size and size == product_size:
                score += 30  # 30 points for size match

            # Partial size match (same diameter, different format)
            elif size and product_size:
                size_num = re.search(r'(\d+)', size)
                prod_size_num = re.search(r'(\d+)', product_size)
                if size_num and prod_size_num and size_num.group(1) == prod_size_num.group(1):
                    score += 20  # 20 points for partial size match

            # Normalize score to 0-1 range
            normalized_score = score / 100

            if normalized_score > best_score and normalized_score >= min_score_threshold:
                best_score = normalized_score
                best_match = product

        return best_match

    def _tokenize(self, text: str) -> set:
        """
        Tokenize text into normalized words for matching.
        Removes common stop words and normalizes terms.
        """
        if not text:
            return set()

        # Convert to lowercase and split
        words = re.findall(r'\b[a-z0-9]+\b', text.lower())

        # Remove common stop words that don't help matching
        stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'to', 'of', 'in', 'on', 'at', 'by', 'with'}

        # Normalize common variations
        normalizations = {
            'endwalls': 'endwall',
            'pipes': 'pipe',
            'inlets': 'inlet',
            'manholes': 'manhole',
            'culverts': 'culvert',
            'rcp': 'pipe',
            'hdpe': 'pipe',
            'pvc': 'pipe',
            'cmp': 'pipe',
            'srcp': 'pipe',
        }

        tokens = set()
        for word in words:
            if word not in stop_words and len(word) > 1:
                # Apply normalization if available
                normalized = normalizations.get(word, word)
                tokens.add(normalized)

        return tokens

    def _select_best_candidate(self, candidates: List[Dict], size: str, desc: str) -> Optional[Dict]:
        """
        Select the best matching product from candidates with the same FDOT code.
        Uses size and description keywords to narrow down.
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        # Score each candidate
        scored = []
        for product in candidates:
            score = 0
            product_size = self._normalize_size(product.get('size', ''))
            product_type = product.get('product_type', '').lower()
            config = product.get('configuration', '').lower()

            # Size match is most important
            if size and size == product_size:
                score += 100

            # Configuration matching
            if 'single' in desc and 'single' in config:
                score += 50
            elif 'double' in desc and 'double' in config:
                score += 50
            elif 'triple' in desc and 'triple' in config:
                score += 50
            elif 'quad' in desc and 'quad' in config:
                score += 50
            elif 'single' in config and 'double' not in desc and 'triple' not in desc and 'quad' not in desc:
                # Default to single if no configuration specified
                score += 10

            # Product type keyword matching
            if 'straight' in desc and 'straight' in product_type:
                score += 25
            if 'winged' in desc and 'winged' in product_type:
                score += 25
            if 'u-type' in desc and 'u-type' in product_type:
                score += 25

            scored.append((score, product))

        # Return highest scoring candidate
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None

    def match_all_items(self, pay_items: List[Dict]) -> List[Dict]:
        """
        Match all pay items to the price list.

        Args:
            pay_items: List of pay item dictionaries

        Returns:
            Updated list with matched prices
        """
        for item in pay_items:
            matched_price = self.match_to_price_list(item)
            if matched_price:
                item['matched'] = True
                item['unit_price'] = matched_price['price']
                item['line_cost'] = item['quantity'] * matched_price['price']
                item['matched_product'] = matched_price

        return pay_items

    def categorize_drainage(self, pay_items: List[Dict]) -> List[Dict]:
        """
        Categorize pay items into drainage structure types.

        Types: Pipe, Endwall, Inlet, Manhole
        """
        drainage_items = []

        for item in pay_items:
            pay_no = item.get('pay_item_no', '')
            desc = item.get('description', '').upper()

            drainage_item = None

            # Pipes (430-175-XXX, 430-084-XXX)
            if pay_no.startswith('430-175') or pay_no.startswith('430-084') or 'PIPE' in desc:
                size_match = re.search(r'(\d{1,2})["\s]*(?:INCH)?', desc)
                size = f'{size_match.group(1)}"' if size_match else "-"

                drainage_item = {
                    "type": "Pipe",
                    "size": size,
                    "description": desc[:80] + ('...' if len(desc) > 80 else ''),
                    "station": "-",
                    "quantity": item.get('quantity', 0),
                    "unit": item.get('unit', 'LF'),
                    "unit_price": item.get('unit_price'),
                    "line_total": item.get('line_cost')
                }

            # Endwalls (430-030, 430-040, 430-010, 430-518, 430-982)
            elif any(pay_no.startswith(prefix) for prefix in ['430-03', '430-04', '430-01', '430-5', '430-98']):
                size_match = re.search(r'(\d{1,2})["\s]*(?:INCH)?', desc)
                size = f'{size_match.group(1)}"' if size_match else "-"

                endwall_type = "Straight"
                if "WINGED" in desc or "WING" in desc:
                    endwall_type = "Winged"
                elif "U-TYPE" in desc or "U TYPE" in desc:
                    endwall_type = "U-Type"
                elif "MES" in desc or "MITERED" in desc:
                    endwall_type = "MES"

                drainage_item = {
                    "type": "Endwall",
                    "size": size,
                    "description": f"{endwall_type} - {desc[:60]}",
                    "station": "-",
                    "quantity": item.get('quantity', 0),
                    "unit": item.get('unit', 'EA'),
                    "unit_price": item.get('unit_price'),
                    "line_total": item.get('line_cost')
                }

            # Inlets (425-1-XXX)
            elif pay_no.startswith('425-1') or 'INLET' in desc:
                inlet_type = "Standard"
                if "TYPE D" in desc:
                    inlet_type = "Type D"
                elif "TYPE E" in desc:
                    inlet_type = "Type E"
                elif "TYPE P" in desc:
                    inlet_type = "Type P"
                elif "DITCH BOTTOM" in desc:
                    inlet_type = "Ditch Bottom"

                drainage_item = {
                    "type": "Inlet",
                    "size": inlet_type,
                    "description": desc[:80] + ('...' if len(desc) > 80 else ''),
                    "station": "-",
                    "quantity": item.get('quantity', 0),
                    "unit": item.get('unit', 'EA'),
                    "unit_price": item.get('unit_price'),
                    "line_total": item.get('line_cost')
                }

            # Manholes (425-2-XX)
            elif pay_no.startswith('425-2') or 'MANHOLE' in desc:
                mh_type = "Standard"
                if "TYPE 7" in desc:
                    mh_type = "Type 7"
                elif "TYPE P" in desc or "P-7" in desc:
                    mh_type = "Type P-7"

                drainage_item = {
                    "type": "Manhole",
                    "size": mh_type,
                    "description": desc[:80] + ('...' if len(desc) > 80 else ''),
                    "station": "-",
                    "quantity": item.get('quantity', 0),
                    "unit": item.get('unit', 'EA'),
                    "unit_price": item.get('unit_price'),
                    "line_total": item.get('line_cost')
                }

            if drainage_item:
                drainage_items.append(drainage_item)

        return drainage_items

    def extract_project_info(self, text: str, filename: str = "") -> Dict:
        """Extract project information from document text."""
        project_info = {
            "name": filename.replace(".pdf", "").replace(".PDF", "").replace("_", " "),
            "location": "",
            "owner": "",
            "engineer": ""
        }

        # Try to find project name
        name_patterns = [
            r"PROJECT[:\s]+(.+?)(?:\n|$)",
            r"PROJECT NAME[:\s]+(.+?)(?:\n|$)",
            r"TITLE[:\s]+(.+?)(?:\n|$)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                project_info["name"] = match.group(1).strip()
                break

        # Try to find location
        location_patterns = [
            r"(?:LOCATION|COUNTY)[:\s]+(.+?)(?:\n|$)",
            r"(.+?COUNTY),?\s*(?:FLORIDA|FL)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                project_info["location"] = match.group(1).strip()
                break

        # Try to find owner
        owner_patterns = [
            r"(?:OWNER|CLIENT)[:\s]+(.+?)(?:\n|$)",
            r"(?:FOR|PREPARED FOR)[:\s]+(.+?)(?:\n|$)"
        ]
        for pattern in owner_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                project_info["owner"] = match.group(1).strip()
                break

        # Try to find engineer
        engineer_patterns = [
            r"(?:ENGINEER|P\.?E\.?)[:\s]+(.+?)(?:\n|$)",
            r"PREPARED BY[:\s]+(.+?)(?:\n|$)"
        ]
        for pattern in engineer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                project_info["engineer"] = match.group(1).strip()
                break

        return project_info

    def generate_report(self, pay_items: List[Dict], project_info: Dict = None) -> Dict:
        """
        Generate complete takeoff report with category-aware match rates.
        
        Provides separate match rates for:
        - Drainage items (pipes, inlets, manholes, endwalls, MES) - what we sell
        - Non-drainage items (electrical, signage, earthwork) - not in catalog
        
        Also breaks down by confidence level.
        """
        matched_items = [p for p in pay_items if p.get('matched')]
        total_cost = sum(p.get('line_cost', 0) or 0 for p in pay_items)
        
        # Category-aware match rate (Phase 2)
        drainage_items = []
        non_drainage_items = []
        
        for item in pay_items:
            if self._is_drainage_item(item):
                drainage_items.append(item)
            else:
                non_drainage_items.append(item)
        
        matched_drainage = [p for p in drainage_items if p.get('matched')]
        drainage_match_rate = (len(matched_drainage) / len(drainage_items) * 100 
                              if drainage_items else 0)
        
        # Confidence breakdown
        high_confidence = [p for p in pay_items 
                          if p.get('confidence_score', 0) >= 0.85]
        medium_confidence = [p for p in pay_items 
                            if 0.6 <= p.get('confidence_score', 0) < 0.85]
        needs_review = [p for p in pay_items 
                       if p.get('confidence_score', 0) < 0.6 or p.get('needs_verification')]
        
        # Source breakdown (product-first vs regex fallback)
        product_first_items = [p for p in pay_items if p.get('source') == 'product_first']
        regex_fallback_items = [p for p in pay_items if p.get('source') != 'product_first']

        report = {
            'project': project_info or {},
            'pay_items': pay_items,
            'summary': {
                'total_items': len(pay_items),
                'matched_items': len(matched_items),
                'match_rate': len(matched_items) / len(pay_items) * 100 if pay_items else 0,
                'estimated_total': round(total_cost, 2),
                
                # Category-aware metrics (Phase 2)
                'drainage_items': len(drainage_items),
                'drainage_matched': len(matched_drainage),
                'drainage_match_rate': round(drainage_match_rate, 1),
                'non_drainage_items': len(non_drainage_items),
                
                # Confidence breakdown
                'high_confidence_count': len(high_confidence),
                'medium_confidence_count': len(medium_confidence),
                'needs_review_count': len(needs_review),
                
                # Source breakdown
                'product_first_count': len(product_first_items),
                'regex_fallback_count': len(regex_fallback_items),
            }
        }

        return report
    
    def _is_drainage_item(self, item: Dict) -> bool:
        """
        Check if an item is a drainage structure (what we sell).
        
        Drainage includes: pipes (RCP, PVC, HDPE, etc.), inlets, manholes,
        endwalls, MES, flared ends, pipe cradles.
        
        Non-drainage includes: electrical, signage, pavement, earthwork,
        maintenance of traffic, mobilization.
        """
        desc = item.get('description', '').upper()
        pay_item_no = item.get('pay_item_no', '').upper()
        source = item.get('source', '')
        
        # Product-first matches are always drainage (by definition)
        if source == 'product_first':
            return True
        
        # FDOT code patterns for drainage
        drainage_fdot_prefixes = [
            '430-',    # Pipes, endwalls, MES
            '425-',    # Inlets, manholes
        ]
        for prefix in drainage_fdot_prefixes:
            if pay_item_no.startswith(prefix):
                return True
        
        # Keyword detection for drainage
        drainage_keywords = [
            'PIPE', 'RCP', 'PVC', 'HDPE', 'CMP', 'DIP', 'SRCP', 'ERCP',
            'INLET', 'MANHOLE', 'CATCH BASIN', 'JUNCTION BOX',
            'ENDWALL', 'MES', 'MITERED', 'FLARED END', 'CRADLE',
            'STORM', 'DRAIN', 'CULVERT'
        ]
        for kw in drainage_keywords:
            if kw in desc:
                return True
        
        return False


def analyze_text(
    text: str,
    price_list_path: str = None,
    filename: str = "",
    validate: bool = True,
    use_product_first: bool = True
) -> Dict:
    """
    Convenience function to analyze extracted text.

    Args:
        text: Extracted text from PDF
        price_list_path: Path to FL 2025 price list CSV
        filename: Original filename for project info
        validate: Whether to run validation gates (default True)
        use_product_first: If True (default), use product-first matching.
                          Set False to use legacy regex-only mode for comparison.

    Returns:
        Dictionary with analysis results including validation_warnings section
        and category-aware match rate metrics.
    """
    analyzer = TakeoffAnalyzer(price_list_path, use_product_first=use_product_first)

    # Extract pay items (uses product-first or regex based on flag)
    pay_items = analyzer.extract_pay_items(text)
    
    # Run validation gates if enabled
    validation_report = None
    if validate:
        pay_items, validation_report = validate_all_items(pay_items)

    # Match to price list (for items not already matched by product-first)
    # Product-first items already have prices; this catches regex fallback items
    pay_items = analyzer.match_all_items(pay_items)

    # Categorize drainage
    drainage_structures = analyzer.categorize_drainage(pay_items)

    # Extract project info
    project_info = analyzer.extract_project_info(text, filename)

    # Generate summary using category-aware reporting
    report = analyzer.generate_report(pay_items, project_info)

    result = {
        "project_info": project_info,
        "pay_items": pay_items,
        "drainage_structures": drainage_structures,
        "summary": report['summary'],
        "matching_mode": "product_first" if analyzer.use_product_first else "regex_only",
    }
    
    # Add validation report if validation was run
    if validation_report:
        result["validation_warnings"] = validation_report.to_dict()
    
    return result


if __name__ == '__main__':
    import sys
    
    # Example usage with multiple format types including new patterns
    sample_text = """
    SUMMARY OF PAY ITEMS
    101-1 MOBILIZATION LS 1
    102-1 MAINTENANCE OF TRAFFIC LS 1
    425-1-549 INLET, TYPE D (MODIFIED) STRUCTURE EA 1
    425-2-41 MANHOLE, TYPE 7 STRUCTURE EA 2
    430-175-118 PIPE CULVERT, SRCP-CLASS III, ROUND 18" LF 98
    430-518-120 STRAIGHT CONCRETE ENDWALLS, 18", SINGLE, ROUND EA 1

    STORM DRAINAGE SCHEDULE
    51 LF 15" RCP CLASS V @ 0.5%
    125 LF 6" PVC SDR 35
    79 LF 3" PVC SDR 35

    NEW FORMAT TESTS:
    18" RCP - 51 LF
    24" HDPE (75 LF)
    RCP 18" @ 100 LF
    30" RCP = 200 LF

    ELLIPTICAL PIPE TESTS:
    75 LF 14"x23" RCP HE CLASS III
    ERCP 19x30
    24"X38" ERCP - 100 LF
    43"x68" ELLIPTICAL RCP

    STRUCTURES / ENDWALLS:
    18" STRAIGHT ENDWALL SINGLE
    STRAIGHT CONCRETE ENDWALL 24" DOUBLE
    WINGED ENDWALL 30" 45 DEG
    18" U-TYPE ENDWALL
    24" MES 4:1
    MES 18"
    14"x23" MES
    FLARED END 36"

    NEW STRUCTURE TESTS (U-TYPE SLOPE RATIOS):
    U-TYPE 18" 2:1
    24" U-TYPE 4:1 WITH GRATE
    U-TYPE ENDWALL 30" 6:1 WITH BAFFLES
    15" U-TYPE 3:1 WITH GRATE AND BAFFLES

    PIPE CRADLE TESTS:
    18" PIPE CRADLE
    PIPE CRADLE 24"
    36" CRADLE

    ELLIPTICAL ACCESSORIES:
    14"x23" FLARED END
    FLARED END 19x30
    24x38 FE
    14"x23" MES 4:1
    MITERED END SECTION 19x30

    GALVANIZED STEEL MES (ROUND):
    18" GALVANIZED MES
    GALV MES 24" SINGLE RUN WITH FRAME
    30" GALVANIZED STEEL MES DOUBLE RUN NO FRAME
    GALV STEEL 4:1 MES 36" TRIPLE RUN

    GALVANIZED STEEL MES (ELLIPTICAL):
    14x23 GALVANIZED MES
    GALV MES 19 X 30 DOUBLE RUN WITH FRAME
    24"x38" GALVANIZED STEEL MES SINGLE RUN NO FRAME

    PIPE SCHEDULE TABLE:
    Size    Material    Qty    Unit
    18"     RCP         150    LF
    24"     PVC         85     LF
    36"     HDPE        120    LF

    STORM DRAIN MANHOLE #1
    STORM DRAIN MANHOLE #2
    GRATE INLET #1
    CATCH BASIN

    CAD CALLOUTS:
    RCP 24
    RCP 24
    PVC 12
    15" HDPE
    """
    
    # Allow toggling between modes via command line
    use_product_first = '--regex-only' not in sys.argv
    
    print(f"=== Testing {'PRODUCT-FIRST' if use_product_first else 'REGEX-ONLY'} mode ===\n")
    
    # Get price list path
    price_list_path = Path(__file__).parent.parent / 'references' / 'fl_2025_prices.csv'
    if not price_list_path.exists():
        print(f"Warning: Price list not found at {price_list_path}")
        price_list_path = None
    
    result = analyze_text(
        sample_text, 
        price_list_path=str(price_list_path) if price_list_path else None,
        filename="sample_project.pdf",
        use_product_first=use_product_first
    )
    
    print(f"Matching mode: {result.get('matching_mode', 'unknown')}")
    print(f"\n=== Summary ===")
    summary = result['summary']
    print(f"Total items: {summary['total_items']}")
    print(f"Matched items: {summary['matched_items']}")
    print(f"Overall match rate: {summary['match_rate']:.1f}%")
    
    # Category-aware metrics (Phase 2)
    if 'drainage_items' in summary:
        print(f"\n=== Drainage Metrics (Category-Aware) ===")
        print(f"Drainage items: {summary['drainage_items']}")
        print(f"Drainage matched: {summary['drainage_matched']}")
        print(f"Drainage match rate: {summary['drainage_match_rate']:.1f}%")
        print(f"Non-drainage items skipped: {summary['non_drainage_items']}")
    
    if 'high_confidence_count' in summary:
        print(f"\n=== Confidence Breakdown ===")
        print(f"High confidence: {summary['high_confidence_count']}")
        print(f"Medium confidence: {summary['medium_confidence_count']}")
        print(f"Needs review: {summary['needs_review_count']}")
    
    if 'product_first_count' in summary:
        print(f"\n=== Source Breakdown ===")
        print(f"Product-first matches: {summary['product_first_count']}")
        print(f"Regex fallback matches: {summary['regex_fallback_count']}")
    
    print(f"\n=== Items ===")
    for item in result['pay_items']:
        source = item.get('source', 'unknown')
        conf = item.get('confidence', 'unknown')
        conf_score = item.get('confidence_score', 0)
        price = item.get('unit_price')
        price_str = f"${price:.2f}" if price else "N/A"
        print(f"  [{source}/{conf}/{conf_score:.2f}] {item['pay_item_no']}: "
              f"{item['description'][:50]} - {item['quantity']} {item['unit']} @ {price_str}")
