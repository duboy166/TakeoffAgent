"""
Validation Gates for AutoWork Pipeline

Centralized validation logic with configurable thresholds for:
- Quantity validation by unit type
- Size validation for pipes
- Description validation
- Price sanity checks

All thresholds are documented in docs/VALIDATION_GATES.md
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    REJECT = "reject"       # Item rejected from output
    WARNING = "warning"     # Item included but flagged
    INFO = "info"           # Informational only


class ValidationCategory(Enum):
    """Categories of validation checks."""
    QUANTITY = "quantity"
    SIZE = "size"
    DESCRIPTION = "description"
    PRICE = "price"
    DUPLICATE = "duplicate"


# =============================================================================
# CONFIGURABLE THRESHOLDS
# =============================================================================

# Quantity limits by unit type
QUANTITY_LIMITS = {
    'LF': {'min': 0, 'max': 10_000, 'typical_max': 5_000},    # Linear feet
    'EA': {'min': 0, 'max': 200, 'typical_max': 50},          # Each (structures)
    'SY': {'min': 0, 'max': 100_000, 'typical_max': 50_000},  # Square yards
    'SF': {'min': 0, 'max': 100_000, 'typical_max': 50_000},  # Square feet
    'CY': {'min': 0, 'max': 50_000, 'typical_max': 10_000},   # Cubic yards
    'TON': {'min': 0, 'max': 100_000, 'typical_max': 10_000}, # Tons
    'GAL': {'min': 0, 'max': 100_000, 'typical_max': 10_000}, # Gallons
    'AC': {'min': 0, 'max': 1_000, 'typical_max': 100},       # Acres
    'LS': {'min': 1, 'max': 1, 'typical_max': 1},             # Lump sum
}

# Suspicious "round" quantities that warrant review
SUSPICIOUS_ROUND_QUANTITIES: Set[float] = {
    100, 200, 250, 500, 1000, 1500, 2000, 2500, 3000, 5000, 10000
}

# Valid FDOT round pipe sizes (inches)
VALID_PIPE_SIZES: Set[int] = {12, 15, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 84, 96}

# Valid elliptical pipe sizes (rise x span)
VALID_ELLIPTICAL_SIZES: Set[Tuple[int, int]] = {
    (12, 18), (14, 23), (19, 30), (24, 38), (29, 45),
    (34, 53), (38, 60), (43, 68), (48, 76), (53, 83), (58, 91)
}

# Maximum round pipe size in catalog
MAX_PIPE_SIZE = 96

# Price sanity thresholds
PRICE_THRESHOLDS = {
    'max_line_cost': 1_000_000,         # Flag if line cost > $1M
    'max_unit_price_lf': 1_000,         # Flag if unit price > $1000/LF
    'max_unit_price_ea': 100_000,       # Flag if unit price > $100K/EA
    'max_unit_price_sy': 500,           # Flag if unit price > $500/SY
    'min_unit_price': 0.01,             # Flag if unit price < $0.01
}

# Description validation
MIN_DESCRIPTION_LENGTH = 3
INVALID_DESCRIPTION_PATTERNS = [
    r'^[\d\.\-\s]+$',                   # Purely numeric (with decimals/dashes)
    r'^(LF|EA|SY|SF|CY|TON|GAL|AC|LS)$',  # Just a unit
    r'^[\s]*$',                          # Empty or whitespace only
]

# =============================================================================
# NON-PIPE MATERIAL DETECTION (False Positive Prevention)
# =============================================================================
# Keywords that indicate membrane/liner materials that should NOT match pipes
MEMBRANE_KEYWORDS: Set[str] = {
    'GEOMEMBRANE', 'MEMBRANE', 'LINER', 'FLEXIBLE', 
    'GEOTEXTILE', 'FABRIC', 'SHEET', 'WATERPROOFING'
}

# Membrane thickness values in mils - these are NOT pipe diameters
# "30 MIL", "40 MIL", "60 MIL" etc.
MEMBRANE_THICKNESS_MILS: Set[int] = {20, 30, 40, 60, 80, 100}

# FDOT categories that are NOT drainage pipes
NON_PIPE_FDOT_PREFIXES: Set[str] = {
    '900-',   # Geomembranes, liners, erosion control
    '570-',   # Seeding, sodding (landscaping)
}


@dataclass
class ValidationWarning:
    """A single validation warning/rejection."""
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    field: str = ""
    original_value: str = ""
    corrected_value: str = ""
    item_index: int = -1
    pay_item_no: str = ""
    auto_corrected: bool = False

    def to_dict(self) -> Dict:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "field": self.field,
            "original_value": str(self.original_value) if self.original_value else "",
            "corrected_value": str(self.corrected_value) if self.corrected_value else "",
            "item_index": self.item_index,
            "pay_item_no": self.pay_item_no,
            "auto_corrected": self.auto_corrected,
        }


@dataclass
class ValidationReport:
    """Complete validation report for a takeoff."""
    passed: List[ValidationWarning] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    rejected: List[ValidationWarning] = field(default_factory=list)
    auto_corrections: List[ValidationWarning] = field(default_factory=list)
    
    @property
    def total_issues(self) -> int:
        return len(self.warnings) + len(self.rejected)
    
    @property
    def total_warnings(self) -> int:
        return len(self.warnings)
    
    @property
    def total_rejected(self) -> int:
        return len(self.rejected)
    
    @property
    def total_auto_corrected(self) -> int:
        return len(self.auto_corrections)

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_issues": self.total_issues,
                "total_warnings": self.total_warnings,
                "total_rejected": self.total_rejected,
                "total_auto_corrected": self.total_auto_corrected,
            },
            "warnings": [w.to_dict() for w in self.warnings],
            "rejected": [r.to_dict() for r in self.rejected],
            "auto_corrections": [c.to_dict() for c in self.auto_corrections],
        }


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_quantity(
    quantity: float,
    unit: str,
    item_index: int = -1,
    pay_item_no: str = ""
) -> Tuple[Optional[float], List[ValidationWarning]]:
    """
    Validate and potentially correct a quantity value.
    
    Args:
        quantity: The quantity to validate
        unit: Unit of measure (LF, EA, SY, etc.)
        item_index: Index of item for reporting
        pay_item_no: Pay item number for reporting
    
    Returns:
        Tuple of (corrected_quantity or None if rejected, list of warnings)
    """
    warnings = []
    unit_upper = unit.upper() if unit else ""
    
    # Get limits for this unit type
    limits = QUANTITY_LIMITS.get(unit_upper, {'min': 0, 'max': float('inf'), 'typical_max': float('inf')})
    
    # Check for negative quantities
    if quantity < limits['min']:
        warnings.append(ValidationWarning(
            category=ValidationCategory.QUANTITY,
            severity=ValidationSeverity.REJECT,
            message=f"Quantity {quantity} is below minimum {limits['min']} for unit {unit_upper}",
            field="quantity",
            original_value=str(quantity),
            item_index=item_index,
            pay_item_no=pay_item_no,
        ))
        return None, warnings
    
    # Check for quantities exceeding hard limits
    if quantity > limits['max']:
        warnings.append(ValidationWarning(
            category=ValidationCategory.QUANTITY,
            severity=ValidationSeverity.REJECT,
            message=f"Quantity {quantity} exceeds maximum {limits['max']:,} for unit {unit_upper}",
            field="quantity",
            original_value=str(quantity),
            item_index=item_index,
            pay_item_no=pay_item_no,
        ))
        return None, warnings
    
    # Check for suspicious round numbers
    if quantity in SUSPICIOUS_ROUND_QUANTITIES:
        warnings.append(ValidationWarning(
            category=ValidationCategory.QUANTITY,
            severity=ValidationSeverity.WARNING,
            message=f"Suspicious round quantity {quantity:,.0f} - verify accuracy",
            field="quantity",
            original_value=str(quantity),
            item_index=item_index,
            pay_item_no=pay_item_no,
        ))
    
    # Check for quantities exceeding typical maximums (warning only)
    elif quantity > limits['typical_max']:
        warnings.append(ValidationWarning(
            category=ValidationCategory.QUANTITY,
            severity=ValidationSeverity.WARNING,
            message=f"Quantity {quantity:,.0f} exceeds typical maximum {limits['typical_max']:,} for {unit_upper}",
            field="quantity",
            original_value=str(quantity),
            item_index=item_index,
            pay_item_no=pay_item_no,
        ))
    
    return quantity, warnings


def validate_pipe_size(
    size: int,
    is_elliptical: bool = False,
    rise: int = 0,
    span: int = 0,
    item_index: int = -1,
    pay_item_no: str = ""
) -> Tuple[bool, List[ValidationWarning]]:
    """
    Validate pipe size against catalog.
    
    Args:
        size: Pipe diameter in inches (for round pipes)
        is_elliptical: Whether this is an elliptical pipe
        rise: Rise dimension for elliptical pipes
        span: Span dimension for elliptical pipes
        item_index: Index of item for reporting
        pay_item_no: Pay item number for reporting
    
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = []
    
    if is_elliptical:
        if (rise, span) not in VALID_ELLIPTICAL_SIZES and (span, rise) not in VALID_ELLIPTICAL_SIZES:
            warnings.append(ValidationWarning(
                category=ValidationCategory.SIZE,
                severity=ValidationSeverity.WARNING,
                message=f"Non-standard elliptical size {rise}\"x{span}\" - not in FL 2025 catalog",
                field="size",
                original_value=f"{rise}x{span}",
                item_index=item_index,
                pay_item_no=pay_item_no,
            ))
            return False, warnings
    else:
        if size not in VALID_PIPE_SIZES:
            # Determine severity - very large sizes are likely errors
            if size > MAX_PIPE_SIZE:
                warnings.append(ValidationWarning(
                    category=ValidationCategory.SIZE,
                    severity=ValidationSeverity.REJECT,
                    message=f"Pipe size {size}\" exceeds maximum catalog size {MAX_PIPE_SIZE}\" - likely a quantity parsed as size",
                    field="size",
                    original_value=str(size),
                    item_index=item_index,
                    pay_item_no=pay_item_no,
                ))
                return False, warnings
            else:
                warnings.append(ValidationWarning(
                    category=ValidationCategory.SIZE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Non-standard pipe size {size}\" - not in FL 2025 catalog",
                    field="size",
                    original_value=str(size),
                    item_index=item_index,
                    pay_item_no=pay_item_no,
                ))
                return False, warnings
    
    return True, warnings


def validate_description(
    description: str,
    item_index: int = -1,
    pay_item_no: str = ""
) -> Tuple[Optional[str], List[ValidationWarning]]:
    """
    Validate item description.
    
    Args:
        description: The description to validate
        item_index: Index of item for reporting
        pay_item_no: Pay item number for reporting
    
    Returns:
        Tuple of (validated description or None if rejected, list of warnings)
    """
    warnings = []
    
    if not description:
        warnings.append(ValidationWarning(
            category=ValidationCategory.DESCRIPTION,
            severity=ValidationSeverity.REJECT,
            message="Empty description",
            field="description",
            item_index=item_index,
            pay_item_no=pay_item_no,
        ))
        return None, warnings
    
    desc_stripped = description.strip()
    
    # Check minimum length
    if len(desc_stripped) < MIN_DESCRIPTION_LENGTH:
        warnings.append(ValidationWarning(
            category=ValidationCategory.DESCRIPTION,
            severity=ValidationSeverity.REJECT,
            message=f"Description too short ({len(desc_stripped)} chars, minimum {MIN_DESCRIPTION_LENGTH})",
            field="description",
            original_value=description,
            item_index=item_index,
            pay_item_no=pay_item_no,
        ))
        return None, warnings
    
    # Check invalid patterns
    for pattern in INVALID_DESCRIPTION_PATTERNS:
        if re.match(pattern, desc_stripped, re.IGNORECASE):
            warnings.append(ValidationWarning(
                category=ValidationCategory.DESCRIPTION,
                severity=ValidationSeverity.REJECT,
                message=f"Invalid description pattern: '{desc_stripped}'",
                field="description",
                original_value=description,
                item_index=item_index,
                pay_item_no=pay_item_no,
            ))
            return None, warnings
    
    return description, warnings


def is_membrane_material(
    text: str,
    size: Optional[int] = None
) -> bool:
    """
    Check if text describes a membrane/liner rather than a pipe.
    
    This prevents false positives like "30 MIL PVC GEOMEMBRANE" 
    being matched to "30 inch RCP".
    
    Args:
        text: Description or context text to check
        size: Optional size value to validate against membrane thicknesses
    
    Returns:
        True if this appears to be a membrane/liner (NOT a pipe)
    """
    if not text:
        return False
    
    text_upper = text.upper()
    
    # Check 1: "MIL" after a number indicates membrane thickness
    # Pattern: "30 MIL", "40 MIL", "60 MIL" etc.
    if re.search(r'\b\d+\s*MIL\b', text_upper):
        return True
    
    # Check 2: Membrane/liner keywords
    if any(kw in text_upper for kw in MEMBRANE_KEYWORDS):
        # Additional check: is the size a common membrane thickness?
        if size is not None and size in MEMBRANE_THICKNESS_MILS:
            return True
        # GEOMEMBRANE or MEMBRANE = definitely not a pipe
        if 'GEOMEMBRANE' in text_upper or 'MEMBRANE' in text_upper:
            return True
        if 'LINER' in text_upper:
            return True
    
    # Check 3: FDOT categories for non-pipe materials
    for prefix in NON_PIPE_FDOT_PREFIXES:
        if prefix in text_upper:
            return True
    
    return False


def validate_not_membrane(
    description: str,
    source_text: str = "",
    size: Optional[int] = None,
    item_index: int = -1,
    pay_item_no: str = ""
) -> Tuple[bool, List[ValidationWarning]]:
    """
    Validate that an item is NOT a membrane/liner (false pipe match).
    
    Args:
        description: Item description
        source_text: Original source text context
        size: Pipe size if known
        item_index: Index of item for reporting
        pay_item_no: Pay item number for reporting
    
    Returns:
        Tuple of (is_valid_pipe, list of warnings)
        is_valid_pipe is False if this appears to be a membrane
    """
    warnings = []
    
    # Check both description and source text
    check_text = f"{description} {source_text}"
    
    if is_membrane_material(check_text, size):
        warnings.append(ValidationWarning(
            category=ValidationCategory.DESCRIPTION,
            severity=ValidationSeverity.REJECT,
            message=f"Appears to be membrane/liner, not pipe: '{description[:50]}...'",
            field="description",
            original_value=description[:100],
            item_index=item_index,
            pay_item_no=pay_item_no,
        ))
        return False, warnings
    
    return True, warnings


def validate_price(
    unit_price: Optional[float],
    line_cost: Optional[float],
    unit: str,
    quantity: float,
    item_index: int = -1,
    pay_item_no: str = ""
) -> List[ValidationWarning]:
    """
    Validate pricing for sanity.
    
    Args:
        unit_price: Unit price
        line_cost: Total line cost
        unit: Unit of measure
        quantity: Quantity
        item_index: Index of item for reporting
        pay_item_no: Pay item number for reporting
    
    Returns:
        List of validation warnings
    """
    warnings = []
    unit_upper = unit.upper() if unit else ""
    
    # Check line cost threshold
    if line_cost and line_cost > PRICE_THRESHOLDS['max_line_cost']:
        warnings.append(ValidationWarning(
            category=ValidationCategory.PRICE,
            severity=ValidationSeverity.WARNING,
            message=f"Line cost ${line_cost:,.2f} exceeds ${PRICE_THRESHOLDS['max_line_cost']:,} - verify accuracy",
            field="line_cost",
            original_value=f"${line_cost:,.2f}",
            item_index=item_index,
            pay_item_no=pay_item_no,
        ))
    
    # Check unit price thresholds by unit type
    if unit_price:
        if unit_price < PRICE_THRESHOLDS['min_unit_price']:
            warnings.append(ValidationWarning(
                category=ValidationCategory.PRICE,
                severity=ValidationSeverity.WARNING,
                message=f"Unit price ${unit_price:.4f} seems too low",
                field="unit_price",
                original_value=f"${unit_price:.4f}",
                item_index=item_index,
                pay_item_no=pay_item_no,
            ))
        
        if unit_upper == 'LF' and unit_price > PRICE_THRESHOLDS['max_unit_price_lf']:
            warnings.append(ValidationWarning(
                category=ValidationCategory.PRICE,
                severity=ValidationSeverity.WARNING,
                message=f"Unit price ${unit_price:,.2f}/LF exceeds typical ${PRICE_THRESHOLDS['max_unit_price_lf']:,}/LF",
                field="unit_price",
                original_value=f"${unit_price:,.2f}",
                item_index=item_index,
                pay_item_no=pay_item_no,
            ))
        
        elif unit_upper == 'EA' and unit_price > PRICE_THRESHOLDS['max_unit_price_ea']:
            warnings.append(ValidationWarning(
                category=ValidationCategory.PRICE,
                severity=ValidationSeverity.WARNING,
                message=f"Unit price ${unit_price:,.2f}/EA exceeds typical ${PRICE_THRESHOLDS['max_unit_price_ea']:,}/EA",
                field="unit_price",
                original_value=f"${unit_price:,.2f}",
                item_index=item_index,
                pay_item_no=pay_item_no,
            ))
        
        elif unit_upper in ('SY', 'SF') and unit_price > PRICE_THRESHOLDS['max_unit_price_sy']:
            warnings.append(ValidationWarning(
                category=ValidationCategory.PRICE,
                severity=ValidationSeverity.WARNING,
                message=f"Unit price ${unit_price:,.2f}/{unit_upper} exceeds typical ${PRICE_THRESHOLDS['max_unit_price_sy']:,}/{unit_upper}",
                field="unit_price",
                original_value=f"${unit_price:,.2f}",
                item_index=item_index,
                pay_item_no=pay_item_no,
            ))
    
    return warnings


def validate_pay_item(
    item: Dict,
    item_index: int = -1
) -> Tuple[Optional[Dict], ValidationReport]:
    """
    Validate a single pay item through all gates.
    
    Args:
        item: Pay item dictionary
        item_index: Index of item for reporting
    
    Returns:
        Tuple of (validated item or None if rejected, ValidationReport)
    """
    report = ValidationReport()
    pay_item_no = item.get('pay_item_no', '')
    
    # Validate description first
    desc_result, desc_warnings = validate_description(
        item.get('description', ''),
        item_index=item_index,
        pay_item_no=pay_item_no
    )
    
    for w in desc_warnings:
        if w.severity == ValidationSeverity.REJECT:
            report.rejected.append(w)
        else:
            report.warnings.append(w)
    
    if desc_result is None:
        return None, report
    
    # Validate quantity
    qty_result, qty_warnings = validate_quantity(
        item.get('quantity', 0),
        item.get('unit', ''),
        item_index=item_index,
        pay_item_no=pay_item_no
    )
    
    for w in qty_warnings:
        if w.severity == ValidationSeverity.REJECT:
            report.rejected.append(w)
        else:
            report.warnings.append(w)
    
    if qty_result is None:
        return None, report
    
    # Validate size if applicable (for pipes)
    desc_upper = item.get('description', '').upper()
    if any(mat in desc_upper for mat in ['RCP', 'PVC', 'HDPE', 'CMP', 'DIP', 'SRCP', 'ERCP', 'PIPE']):
        # Try to extract size
        size_match = re.search(r'(\d+)["\u201d]', desc_upper)
        ellip_match = re.search(r'(\d+)\s*[xX×]\s*(\d+)', desc_upper)
        
        if ellip_match:
            rise = int(ellip_match.group(1))
            span = int(ellip_match.group(2))
            is_valid, size_warnings = validate_pipe_size(
                size=0,
                is_elliptical=True,
                rise=rise,
                span=span,
                item_index=item_index,
                pay_item_no=pay_item_no
            )
            for w in size_warnings:
                if w.severity == ValidationSeverity.REJECT:
                    report.rejected.append(w)
                else:
                    report.warnings.append(w)
            if not is_valid and any(w.severity == ValidationSeverity.REJECT for w in size_warnings):
                return None, report
                
        elif size_match:
            size = int(size_match.group(1))
            is_valid, size_warnings = validate_pipe_size(
                size=size,
                is_elliptical=False,
                item_index=item_index,
                pay_item_no=pay_item_no
            )
            for w in size_warnings:
                if w.severity == ValidationSeverity.REJECT:
                    report.rejected.append(w)
                else:
                    report.warnings.append(w)
            if not is_valid and any(w.severity == ValidationSeverity.REJECT for w in size_warnings):
                return None, report
    
    # Validate price if present
    if item.get('unit_price') or item.get('line_cost'):
        price_warnings = validate_price(
            unit_price=item.get('unit_price'),
            line_cost=item.get('line_cost'),
            unit=item.get('unit', ''),
            quantity=qty_result,
            item_index=item_index,
            pay_item_no=pay_item_no
        )
        for w in price_warnings:
            report.warnings.append(w)
    
    return item, report


def validate_all_items(items: List[Dict]) -> Tuple[List[Dict], ValidationReport]:
    """
    Validate all pay items and return filtered list with report.
    
    Args:
        items: List of pay item dictionaries
    
    Returns:
        Tuple of (filtered valid items, combined validation report)
    """
    combined_report = ValidationReport()
    valid_items = []
    
    for idx, item in enumerate(items):
        validated_item, item_report = validate_pay_item(item, item_index=idx)
        
        # Merge reports
        combined_report.warnings.extend(item_report.warnings)
        combined_report.rejected.extend(item_report.rejected)
        combined_report.auto_corrections.extend(item_report.auto_corrections)
        
        if validated_item is not None:
            valid_items.append(validated_item)
    
    # Log summary
    logger.info(
        f"Validation complete: {len(valid_items)}/{len(items)} items passed, "
        f"{combined_report.total_warnings} warnings, {combined_report.total_rejected} rejected"
    )
    
    return valid_items, combined_report
