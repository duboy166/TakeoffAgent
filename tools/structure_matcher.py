"""
Structure Matcher - Matches inlets, manholes, and other drainage structures to FDOT pay items.

Handles the naming mismatch between Vision extraction and FDOT price lists:
- Vision: "TYPE C INLET #3", "MANHOLE #1", "TYPE D INLET (MODIFIED)"
- FDOT: "425-1-521 INLET, CURB, TYPE C", "425-2-41 MANHOLE, TYPE J STRUCTURE"
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from difflib import SequenceMatcher


@dataclass
class StructureMatch:
    """Result of structure matching."""
    fdot_code: str
    description: str
    structure_type: str  # INLET, MANHOLE, JUNCTION_BOX, CATCH_BASIN
    subtype: str         # Type A, Type C, Type D Modified, etc.
    unit: str
    unit_price: float
    confidence: float    # 0.0 to 1.0
    match_reason: str


# =============================================================================
# FDOT Structure Database
# Prices are estimates based on FDOT historical averages (2024-2025)
# =============================================================================

FDOT_STRUCTURES: Dict[str, Dict] = {
    # ----- CURB INLETS (425-1-XXX) -----
    # Type numbers indicate riser height categories
    
    # Type A - Curb Opening Inlet (small)
    "425-1-101": {"type": "INLET", "subtype": "CURB TYPE A", "price": 3200.00, "depth": "<4'"},
    "425-1-102": {"type": "INLET", "subtype": "CURB TYPE A", "price": 3800.00, "depth": "4'-6'"},
    "425-1-103": {"type": "INLET", "subtype": "CURB TYPE A", "price": 4500.00, "depth": "6'-8'"},
    
    # Type B - Curb Opening Inlet (standard)
    "425-1-201": {"type": "INLET", "subtype": "CURB TYPE B", "price": 3500.00, "depth": "<4'"},
    "425-1-202": {"type": "INLET", "subtype": "CURB TYPE B", "price": 4200.00, "depth": "4'-6'"},
    "425-1-203": {"type": "INLET", "subtype": "CURB TYPE B", "price": 5000.00, "depth": "6'-8'"},
    
    # Type C - Ditch Bottom Inlet
    "425-1-521": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE C", "price": 4200.00, "depth": "<4'"},
    "425-1-522": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE C", "price": 4800.00, "depth": "4'-6'"},
    "425-1-523": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE C", "price": 5500.00, "depth": "6'-8'"},
    "425-1-541": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE C", "price": 4200.00, "depth": "Standard"},
    
    # Type D - Ditch Bottom Inlet (larger)
    "425-1-531": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE D", "price": 4800.00, "depth": "<4'"},
    "425-1-532": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE D", "price": 5500.00, "depth": "4'-6'"},
    "425-1-533": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE D", "price": 6200.00, "depth": "6'-8'"},
    "425-1-549": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE D MODIFIED", "price": 5800.00, "depth": "Standard"},
    
    # Type E - Ditch Bottom Inlet
    "425-1-551": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE E", "price": 4000.00, "depth": "<4'"},
    "425-1-552": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE E", "price": 4600.00, "depth": "4'-6'"},
    "425-1-553": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE E", "price": 5200.00, "depth": "6'-8'"},
    
    # Type H - Ditch Bottom Inlet (high capacity)
    "425-1-561": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE H", "price": 5500.00, "depth": "<4'"},
    "425-1-562": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE H", "price": 6200.00, "depth": "4'-6'"},
    "425-1-563": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE H", "price": 7000.00, "depth": "6'-8'"},
    
    # Type J - Grate Inlet / Ditch Bottom
    "425-1-571": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE J", "price": 5200.00, "depth": "<4'"},
    "425-1-572": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE J", "price": 5900.00, "depth": "4'-6'"},
    "425-1-573": {"type": "INLET", "subtype": "DITCH BOTTOM TYPE J", "price": 6600.00, "depth": "6'-8'"},
    
    # Type P - Curb Inlet with grate (large, high capacity)
    "425-1-601": {"type": "INLET", "subtype": "CURB TYPE P", "price": 6500.00, "depth": "<6'"},
    "425-1-602": {"type": "INLET", "subtype": "CURB TYPE P", "price": 7500.00, "depth": "6'-8'"},
    "425-1-603": {"type": "INLET", "subtype": "CURB TYPE P", "price": 8500.00, "depth": "8'-10'"},
    "425-1-604": {"type": "INLET", "subtype": "CURB TYPE P", "price": 9500.00, "depth": ">10'"},
    "425-1-621": {"type": "INLET", "subtype": "CURB TYPE P-5", "price": 7200.00, "depth": "Standard"},
    "425-1-631": {"type": "INLET", "subtype": "CURB TYPE P-6", "price": 8800.00, "depth": "Standard"},
    
    # Type 9 - Curb Inlet (common in urban areas)
    "425-1-901": {"type": "INLET", "subtype": "CURB TYPE 9", "price": 5500.00, "depth": "<6'"},
    "425-1-902": {"type": "INLET", "subtype": "CURB TYPE 9", "price": 6500.00, "depth": "6'-8'"},
    "425-1-903": {"type": "INLET", "subtype": "CURB TYPE 9", "price": 7500.00, "depth": "8'-10'"},
    
    # Type S - Slotted Drain Inlet
    "425-1-701": {"type": "INLET", "subtype": "SLOTTED TYPE S", "price": 4800.00, "depth": "<4'"},
    "425-1-702": {"type": "INLET", "subtype": "SLOTTED TYPE S", "price": 5500.00, "depth": "4'-6'"},
    
    # Type V - Valley Gutter Inlet
    "425-1-751": {"type": "INLET", "subtype": "VALLEY GUTTER TYPE V", "price": 4200.00, "depth": "<4'"},
    "425-1-752": {"type": "INLET", "subtype": "VALLEY GUTTER TYPE V", "price": 4900.00, "depth": "4'-6'"},
    
    # ----- MANHOLES (425-2-XX) -----
    # Structure bottoms - Type J and P are most common
    
    # Type J Structure Bottom
    "425-2-31": {"type": "MANHOLE", "subtype": "TYPE J", "price": 4500.00, "depth": "<4'"},
    "425-2-32": {"type": "MANHOLE", "subtype": "TYPE J", "price": 5200.00, "depth": "4'-6'"},
    "425-2-33": {"type": "MANHOLE", "subtype": "TYPE J", "price": 6000.00, "depth": "6'-8'"},
    "425-2-34": {"type": "MANHOLE", "subtype": "TYPE J", "price": 6800.00, "depth": "8'-10'"},
    "425-2-41": {"type": "MANHOLE", "subtype": "TYPE 7", "price": 5500.00, "depth": "Standard"},  # Type 7 = Type J standard
    
    # Type P Structure Bottom
    "425-2-71": {"type": "MANHOLE", "subtype": "TYPE P", "price": 5500.00, "depth": "<6'"},
    "425-2-72": {"type": "MANHOLE", "subtype": "TYPE P", "price": 6500.00, "depth": "6'-8'"},
    "425-2-73": {"type": "MANHOLE", "subtype": "TYPE P", "price": 7500.00, "depth": "8'-10'"},
    "425-2-74": {"type": "MANHOLE", "subtype": "TYPE P", "price": 8500.00, "depth": ">10'"},
    "425-2-81": {"type": "MANHOLE", "subtype": "TYPE P-7", "price": 6200.00, "depth": "Standard"},
    
    # Junction Boxes
    "425-3-11": {"type": "JUNCTION_BOX", "subtype": "STANDARD", "price": 3800.00, "depth": "<4'"},
    "425-3-12": {"type": "JUNCTION_BOX", "subtype": "STANDARD", "price": 4500.00, "depth": "4'-6'"},
    "425-3-13": {"type": "JUNCTION_BOX", "subtype": "STANDARD", "price": 5200.00, "depth": "6'-8'"},
    
    # Catch Basins (similar to inlets but with sump)
    "425-4-11": {"type": "CATCH_BASIN", "subtype": "STANDARD", "price": 3500.00, "depth": "<4'"},
    "425-4-12": {"type": "CATCH_BASIN", "subtype": "STANDARD", "price": 4200.00, "depth": "4'-6'"},
    "425-4-13": {"type": "CATCH_BASIN", "subtype": "STANDARD", "price": 5000.00, "depth": "6'-8'"},
}

# Default prices by structure type (when type cannot be determined)
DEFAULT_STRUCTURE_PRICES: Dict[str, Tuple[str, float]] = {
    "INLET": ("425-1-541", 4200.00),          # Default to Type C
    "MANHOLE": ("425-2-41", 5500.00),         # Default to Type 7/J
    "JUNCTION_BOX": ("425-3-11", 3800.00),
    "CATCH_BASIN": ("425-4-11", 3500.00),
}


# =============================================================================
# Mapping Patterns
# Maps common callout text patterns to FDOT structure types
# =============================================================================

STRUCTURE_TYPE_PATTERNS: List[Tuple[str, str, str]] = [
    # Pattern, Structure Type, Subtype
    # Inlet patterns - more specific first (MODIFIED patterns must come first!)
    (r"\bTYPE\s*D\s*(?:INLET\s*)?[\(\[]?\s*(?:MODIFIED|MOD)", "INLET", "DITCH BOTTOM TYPE D MODIFIED"),
    (r"\bINLET.*TYPE\s*D.*(?:MODIFIED|MOD)", "INLET", "DITCH BOTTOM TYPE D MODIFIED"),
    (r"(?:MODIFIED|MOD).*TYPE\s*D\s*INLET", "INLET", "DITCH BOTTOM TYPE D MODIFIED"),
    (r"\bD\s*(?:MODIFIED|MOD)\s*INLET\b", "INLET", "DITCH BOTTOM TYPE D MODIFIED"),
    (r"\bMODIFIED\s*(?:TYPE\s*)?D\b", "INLET", "DITCH BOTTOM TYPE D MODIFIED"),
    (r"\bTYPE\s*D\s*INLET\b", "INLET", "DITCH BOTTOM TYPE D"),
    (r"\bTYPE\s*D\b.*\bINLET\b(?!\s*[\(\[]?\s*(?:MODIFIED|MOD))", "INLET", "DITCH BOTTOM TYPE D"),
    (r"\bINLET.*\bTYPE\s*D\b(?!\s*[\(\[]?\s*(?:MODIFIED|MOD))", "INLET", "DITCH BOTTOM TYPE D"),
    (r"\bD\s*INLET\b", "INLET", "DITCH BOTTOM TYPE D"),
    
    (r"\bTYPE\s*C\s*INLET\b", "INLET", "DITCH BOTTOM TYPE C"),
    (r"\bTYPE\s*C\b.*\bINLET\b", "INLET", "DITCH BOTTOM TYPE C"),
    (r"\bINLET.*\bTYPE\s*C\b", "INLET", "DITCH BOTTOM TYPE C"),
    (r"\bC\s*INLET\b", "INLET", "DITCH BOTTOM TYPE C"),
    
    (r"\bTYPE\s*E\s*INLET\b", "INLET", "DITCH BOTTOM TYPE E"),
    (r"\bTYPE\s*E\b.*\bINLET\b", "INLET", "DITCH BOTTOM TYPE E"),
    (r"\bINLET.*\bTYPE\s*E\b", "INLET", "DITCH BOTTOM TYPE E"),
    (r"\bDITCH\s*BOTTOM\s*INLET\s*TYPE\s*E\b", "INLET", "DITCH BOTTOM TYPE E"),
    (r"\bE\s*INLET\b", "INLET", "DITCH BOTTOM TYPE E"),
    
    (r"\bTYPE\s*H\s*INLET\b", "INLET", "DITCH BOTTOM TYPE H"),
    (r"\bTYPE\s*H\b.*\bINLET\b", "INLET", "DITCH BOTTOM TYPE H"),
    (r"\bH\s*INLET\b", "INLET", "DITCH BOTTOM TYPE H"),
    
    (r"\bTYPE\s*J\s*INLET\b", "INLET", "DITCH BOTTOM TYPE J"),
    (r"\bTYPE\s*J\b.*\bINLET\b", "INLET", "DITCH BOTTOM TYPE J"),
    (r"\bJ\s*(?:BOTTOM\s*)?INLET\b", "INLET", "DITCH BOTTOM TYPE J"),
    
    (r"\bTYPE\s*P[-\s]?[5-7]?\s*INLET\b", "INLET", "CURB TYPE P"),
    (r"\bP[-\s]?5\s*INLET\b", "INLET", "CURB TYPE P-5"),
    (r"\bP[-\s]?6\s*INLET\b", "INLET", "CURB TYPE P-6"),
    (r"\bCURB\s*INLET\s*TYPE\s*P\b", "INLET", "CURB TYPE P"),
    
    (r"\bTYPE\s*9\s*INLET\b", "INLET", "CURB TYPE 9"),
    (r"\bTYPE\s*9\b.*\bINLET\b", "INLET", "CURB TYPE 9"),
    (r"\bCURB\s*INLET\s*TYPE\s*9\b", "INLET", "CURB TYPE 9"),
    
    (r"\bTYPE\s*A\s*INLET\b", "INLET", "CURB TYPE A"),
    (r"\bTYPE\s*B\s*INLET\b", "INLET", "CURB TYPE B"),
    
    (r"\bTYPE\s*S\s*INLET\b", "INLET", "SLOTTED TYPE S"),
    (r"\bSLOTTED\s*(?:DRAIN\s*)?INLET\b", "INLET", "SLOTTED TYPE S"),
    
    (r"\bTYPE\s*V\s*INLET\b", "INLET", "VALLEY GUTTER TYPE V"),
    (r"\bVALLEY\s*(?:GUTTER\s*)?INLET\b", "INLET", "VALLEY GUTTER TYPE V"),
    (r"\bGUTTER\s*INLET\b", "INLET", "VALLEY GUTTER TYPE V"),
    
    (r"\bDITCH\s*BOTTOM\s*INLET\b(?!\s*TYPE)", "INLET", "DITCH BOTTOM TYPE C"),  # Default ditch bottom (no type specified)
    (r"\bCURB\s*INLET\b", "INLET", "CURB TYPE B"),  # Default curb inlet
    
    # Generic inlet (catch all)
    (r"\bINLET\s*#?\d*\b", "INLET", None),
    
    # Manhole patterns
    (r"\bTYPE\s*7\s*MANHOLE\b", "MANHOLE", "TYPE 7"),
    (r"\bTYPE\s*7\b.*\bMANHOLE\b", "MANHOLE", "TYPE 7"),
    (r"\bMANHOLE.*\bTYPE\s*7\b", "MANHOLE", "TYPE 7"),
    (r"\bMH\s*TYPE\s*7\b", "MANHOLE", "TYPE 7"),
    
    (r"\bTYPE\s*J\s*MANHOLE\b", "MANHOLE", "TYPE J"),
    (r"\bTYPE\s*J\b.*\bMANHOLE\b", "MANHOLE", "TYPE J"),
    (r"\bMANHOLE.*\bTYPE\s*J\b", "MANHOLE", "TYPE J"),
    (r"\bJ\s*BOTTOM\s*MANHOLE\b", "MANHOLE", "TYPE J"),
    
    (r"\bTYPE\s*P[-\s]?7?\s*MANHOLE\b", "MANHOLE", "TYPE P"),
    (r"\bP[-\s]?7\s*MANHOLE\b", "MANHOLE", "TYPE P-7"),
    
    # Generic manhole
    (r"\bMANHOLE\s*#?\d*\b", "MANHOLE", None),
    (r"\bMH\s*#?\d*\b", "MANHOLE", None),
    
    # Junction box patterns
    (r"\bJUNCTION\s*BOX\b", "JUNCTION_BOX", "STANDARD"),
    (r"\bJCT\s*BOX\b", "JUNCTION_BOX", "STANDARD"),
    (r"\bJB\s*#?\d*\b", "JUNCTION_BOX", "STANDARD"),
    
    # Catch basin patterns
    (r"\bCATCH\s*BASIN\b", "CATCH_BASIN", "STANDARD"),
    (r"\bCB\s*#?\d*\b", "CATCH_BASIN", "STANDARD"),
]


def extract_structure_number(text: str) -> Optional[str]:
    """Extract structure number from text like 'INLET #35' or 'MH #103'."""
    match = re.search(r'#\s*(\d+)', text)
    if match:
        return match.group(1)
    return None


def extract_depth_hint(text: str) -> Optional[str]:
    """Extract depth information from text if available."""
    # Look for patterns like "6' DEEP", "8-10'", "DEPTH: 6'"
    depth_patterns = [
        r"(\d+)['\s]*(?:-\s*(\d+)['\s]*)?DEEP",
        r"DEPTH[:\s]*(\d+)['\s]*(?:-\s*(\d+)['\s]*)?",
        r"(\d+)['\s]*TO['\s]*(\d+)['\s]*(?:DEEP|DEPTH)?",
    ]
    
    for pattern in depth_patterns:
        match = re.search(pattern, text.upper())
        if match:
            min_depth = int(match.group(1))
            max_depth = int(match.group(2)) if match.group(2) else min_depth
            avg = (min_depth + max_depth) / 2
            
            if avg < 4:
                return "<4'"
            elif avg < 6:
                return "4'-6'"
            elif avg < 8:
                return "6'-8'"
            elif avg < 10:
                return "8'-10'"
            else:
                return ">10'"
    
    return None


def match_structure(text: str, include_generic: bool = True) -> Optional[StructureMatch]:
    """
    Match a structure description to an FDOT pay item.
    
    Args:
        text: Structure description (e.g., "TYPE D INLET (MODIFIED)", "MANHOLE #1")
        include_generic: If True, match generic structures without type specified
        
    Returns:
        StructureMatch with FDOT code, price, and confidence, or None if no match
    """
    text_upper = text.upper().strip()
    
    if not text_upper:
        return None
    
    # Try to match against known patterns
    matched_type = None
    matched_subtype = None
    best_confidence = 0.0
    match_reason = ""
    
    for pattern, struct_type, subtype in STRUCTURE_TYPE_PATTERNS:
        if re.search(pattern, text_upper, re.IGNORECASE):
            # Calculate confidence based on pattern specificity
            if subtype:
                # Specific type match (e.g., "TYPE D MODIFIED")
                confidence = 0.95
                match_reason = f"Pattern match: {subtype}"
            else:
                # Generic match (e.g., just "INLET")
                if not include_generic:
                    continue
                confidence = 0.70
                match_reason = f"Generic {struct_type.lower()} match"
            
            if confidence > best_confidence:
                best_confidence = confidence
                matched_type = struct_type
                matched_subtype = subtype
    
    if not matched_type:
        return None
    
    # Find the best FDOT code for this structure
    depth_hint = extract_depth_hint(text_upper)
    fdot_code, unit_price = find_fdot_code(matched_type, matched_subtype, depth_hint)
    
    # Build description
    if matched_subtype:
        description = f"{matched_type.replace('_', ' ')}, {matched_subtype}"
    else:
        description = matched_type.replace('_', ' ')
    
    return StructureMatch(
        fdot_code=fdot_code,
        description=description,
        structure_type=matched_type,
        subtype=matched_subtype or "STANDARD",
        unit="EA",
        unit_price=unit_price,
        confidence=best_confidence,
        match_reason=match_reason
    )


def find_fdot_code(struct_type: str, subtype: Optional[str], depth: Optional[str] = None) -> Tuple[str, float]:
    """
    Find the best FDOT code and price for a structure type.
    
    Args:
        struct_type: INLET, MANHOLE, JUNCTION_BOX, or CATCH_BASIN
        subtype: Specific type like "DITCH BOTTOM TYPE D MODIFIED"
        depth: Optional depth category like "<4'" or "6'-8'"
        
    Returns:
        Tuple of (fdot_code, unit_price)
    """
    if subtype:
        subtype_upper = subtype.upper()
        
        # Direct lookup for known subtypes
        direct_mappings = {
            # Inlets
            "DITCH BOTTOM TYPE D MODIFIED": ("425-1-549", 5800.00),
            "DITCH BOTTOM TYPE D": ("425-1-531", 4800.00),
            "DITCH BOTTOM TYPE C": ("425-1-541", 4200.00),
            "DITCH BOTTOM TYPE E": ("425-1-551", 4000.00),
            "DITCH BOTTOM TYPE H": ("425-1-561", 5500.00),
            "DITCH BOTTOM TYPE J": ("425-1-571", 5200.00),
            "CURB TYPE A": ("425-1-101", 3200.00),
            "CURB TYPE B": ("425-1-201", 3500.00),
            "CURB TYPE P": ("425-1-601", 6500.00),
            "CURB TYPE P-5": ("425-1-621", 7200.00),
            "CURB TYPE P-6": ("425-1-631", 8800.00),
            "CURB TYPE 9": ("425-1-901", 5500.00),
            "SLOTTED TYPE S": ("425-1-701", 4800.00),
            "VALLEY GUTTER TYPE V": ("425-1-751", 4200.00),
            # Manholes
            "TYPE 7": ("425-2-41", 5500.00),
            "TYPE J": ("425-2-31", 4500.00),
            "TYPE P": ("425-2-71", 5500.00),
            "TYPE P-7": ("425-2-81", 6200.00),
            # Others
            "STANDARD": DEFAULT_STRUCTURE_PRICES.get(struct_type, ("425-1-541", 4200.00)),
        }
        
        # Check for direct match
        if subtype_upper in direct_mappings:
            return direct_mappings[subtype_upper]
        
        # Look for fuzzy subtype match
        best_match = None
        best_score = 0
        
        for code, data in FDOT_STRUCTURES.items():
            if data["type"] != struct_type:
                continue
            
            # Score based on subtype similarity
            score = SequenceMatcher(None, subtype_upper, data["subtype"].upper()).ratio()
            
            # Bonus for depth match
            if depth and data.get("depth") == depth:
                score += 0.1
            
            if score > best_score:
                best_score = score
                best_match = (code, data["price"])
        
        if best_match and best_score > 0.5:
            return best_match
    
    # Fall back to default for structure type
    if struct_type in DEFAULT_STRUCTURE_PRICES:
        return DEFAULT_STRUCTURE_PRICES[struct_type]
    
    # Ultimate fallback
    return ("425-1-541", 4200.00)


def match_structures_in_text(text: str) -> List[StructureMatch]:
    """
    Find all structure matches in a block of text.
    
    Args:
        text: Text to search (e.g., from plan callouts)
        
    Returns:
        List of StructureMatch objects
    """
    matches = []
    text_upper = text.upper()
    
    # Find all potential structure mentions
    structure_patterns = [
        r"(?:TYPE\s*[A-Z][-\s]?\d?|[A-Z][-\s]?\d?\s*TYPE)?\s*INLET(?:\s*#?\d+)?(?:\s*\([^)]+\))?",
        r"(?:TYPE\s*[A-Z0-9][-\s]?\d?|[A-Z0-9][-\s]?\d?\s*TYPE)?\s*MANHOLE(?:\s*#?\d+)?",
        r"JUNCTION\s*BOX(?:\s*#?\d+)?",
        r"CATCH\s*BASIN(?:\s*#?\d+)?",
        r"(?:MH|CB|JB|INL?)\s*#?\d+",
    ]
    
    seen_positions = set()
    
    for pattern in structure_patterns:
        for match in re.finditer(pattern, text_upper, re.IGNORECASE):
            # Avoid duplicate matches at same position
            pos = (match.start(), match.end())
            if pos in seen_positions:
                continue
            seen_positions.add(pos)
            
            match_text = match.group(0).strip()
            result = match_structure(match_text)
            if result:
                matches.append(result)
    
    return matches


def get_structure_price(fdot_code: str) -> Optional[float]:
    """Get price for a specific FDOT structure code."""
    if fdot_code in FDOT_STRUCTURES:
        return FDOT_STRUCTURES[fdot_code]["price"]
    return None


def list_inlet_types() -> List[Tuple[str, str, float]]:
    """List all inlet types with codes and prices."""
    return [
        (code, data["subtype"], data["price"])
        for code, data in FDOT_STRUCTURES.items()
        if data["type"] == "INLET"
    ]


def list_manhole_types() -> List[Tuple[str, str, float]]:
    """List all manhole types with codes and prices."""
    return [
        (code, data["subtype"], data["price"])
        for code, data in FDOT_STRUCTURES.items()
        if data["type"] == "MANHOLE"
    ]


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    # Test cases
    test_cases = [
        "TYPE C INLET #3",
        "MANHOLE #1",
        "TYPE D INLET (MODIFIED)",
        "INLET, TYPE D (MODIFIED) STRUCTURE",
        "MANHOLE, TYPE 7 STRUCTURE",
        "MANHOLE, TYPE J",
        "CURB INLET TYPE 9",
        "DITCH BOTTOM INLET TYPE E",
        "P-5 INLET",
        "INLET #35",
        "JUNCTION BOX",
        "CATCH BASIN #12",
        "MH TYPE J 6' DEEP",
    ]
    
    print("=" * 70)
    print("Structure Matcher Test Results")
    print("=" * 70)
    
    for test in test_cases:
        result = match_structure(test)
        if result:
            print(f"\nInput: {test}")
            print(f"  FDOT Code: {result.fdot_code}")
            print(f"  Description: {result.description}")
            print(f"  Price: ${result.unit_price:,.2f}")
            print(f"  Confidence: {result.confidence:.0%}")
            print(f"  Reason: {result.match_reason}")
        else:
            print(f"\nInput: {test}")
            print("  No match found")
