"""
Product Catalog - Product-first matching for AutoWork

Loads products from price list CSV and generates search patterns
for matching against plan text. This is the foundation of the
product-first matching approach.
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Dict


@dataclass
class Product:
    """A product from the price catalog with generated search patterns."""
    id: str                         # Unique identifier (e.g., "RCP_18_CL_III")
    category: str                   # End_Sections, RCP_Price_List, MES_Pipe_Grates
    product_type: str               # "Straight Endwalls", "Round Reinforced Concrete Pipe"
    size: str                       # "18"", "24"X38"", etc.
    configuration: str              # "Single", "CL III", "Double Run With Frame"
    price: float                    # Unit price (0.0 if "call for pricing")
    unit: str                       # EA, LF
    fdot_code: Optional[str]        # FDOT code for reference (not primary matching)
    search_patterns: List[str] = field(default_factory=list)
    keywords: Set[str] = field(default_factory=set)
    notes: str = ""
    
    @property
    def price_display(self) -> str:
        """Return price as display string."""
        if self.price == 0.0:
            return "Call for pricing"
        return f"${self.price:,.2f}"


class ProductCatalog:
    """
    Catalog of products loaded from price list CSV.
    
    Products are indexed for efficient searching:
    - By category
    - By size
    - By keywords (for quick filtering)
    """
    
    def __init__(self):
        self.products: List[Product] = []
        self.by_category: Dict[str, List[Product]] = {}
        self.by_size: Dict[str, List[Product]] = {}
        self.keyword_index: Dict[str, Set[int]] = {}  # keyword -> product indices
    
    @classmethod
    def from_csv(cls, csv_path: Path) -> 'ProductCatalog':
        """Load catalog from FL 2025 prices CSV."""
        catalog = cls()
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                product = catalog._parse_row(row)
                if product:
                    catalog._add_product(product)
        
        return catalog
    
    def _parse_row(self, row: Dict[str, str]) -> Optional[Product]:
        """Parse a CSV row into a Product."""
        try:
            # Parse price - handle "call for pricing" and dollar signs
            price_str = row.get('Price', '0').strip()
            if 'call' in price_str.lower():
                price = 0.0
            else:
                price = float(price_str.replace('$', '').replace(',', ''))
            
            # Build unique ID
            category = row.get('Document', '').strip()
            product_type = row.get('Product_Type', '').strip()
            size = row.get('Size', '').strip()
            config = row.get('Configuration', '').strip()
            
            # Create safe ID
            safe_id = f"{category}_{product_type}_{size}_{config}"
            safe_id = re.sub(r'[^\w]', '_', safe_id)
            safe_id = re.sub(r'_+', '_', safe_id).strip('_')
            
            product = Product(
                id=safe_id,
                category=category,
                product_type=product_type,
                size=size,
                configuration=config,
                price=price,
                unit=row.get('Unit', 'EA').strip(),
                fdot_code=row.get('FDOT_Code', '').strip() or None,
                notes=row.get('Notes', '').strip()
            )
            
            # Generate search patterns
            product.search_patterns = generate_patterns(product)
            product.keywords = generate_keywords(product)
            
            return product
            
        except (ValueError, KeyError) as e:
            # Skip malformed rows
            return None
    
    def _add_product(self, product: Product):
        """Add product to catalog and indices."""
        idx = len(self.products)
        self.products.append(product)
        
        # Index by category
        if product.category not in self.by_category:
            self.by_category[product.category] = []
        self.by_category[product.category].append(product)
        
        # Index by size (normalized)
        size_key = normalize_size(product.size)
        if size_key not in self.by_size:
            self.by_size[size_key] = []
        self.by_size[size_key].append(product)
        
        # Index by keywords
        for kw in product.keywords:
            if kw not in self.keyword_index:
                self.keyword_index[kw] = set()
            self.keyword_index[kw].add(idx)
    
    def get_candidates(self, text: str) -> List[Product]:
        """
        Quick filter: return products whose keywords appear in text.
        This is the first pass before pattern matching.
        """
        text_upper = text.upper()
        candidates = []
        
        for product in self.products:
            # Check if ALL required keywords are present
            if all(kw in text_upper for kw in product.keywords):
                candidates.append(product)
        
        return candidates
    
    def __len__(self):
        return len(self.products)
    
    def __iter__(self):
        return iter(self.products)


def normalize_size(size: str) -> str:
    """Normalize size string for indexing (remove quotes, spaces)."""
    return size.replace('"', '').replace("'", '').replace(' ', '').upper()


def generate_keywords(product: Product) -> Set[str]:
    """
    Generate required keywords for quick filtering.
    
    Keywords must ALL be present in text for the product to be a candidate.
    Keep these minimal to avoid false negatives.
    
    Note: We use a flexible check in get_candidates() that handles
    OCR variants (RGP for RCP, etc.)
    """
    keywords = set()
    
    # Extract size number(s)
    size_nums = re.findall(r'\d+', product.size)
    if size_nums:
        # Always require the primary size
        keywords.add(size_nums[0])
    
    # Category-specific keywords - keep minimal!
    # We intentionally don't add RCP/ENDWALL here because:
    # 1. OCR errors (RGP instead of RCP) would cause false negatives
    # 2. The pattern matching will do the actual filtering
    # 3. Size is usually enough for initial candidate filtering
    
    if product.category == 'RCP_Price_List':
        if 'Elliptical' in product.product_type:
            # Elliptical is distinctive enough
            pass  # Size dimensions are sufficient
    
    elif product.category == 'End_Sections':
        pass  # Size + pattern matching will handle it
    
    elif product.category == 'MES_Pipe_Grates':
        keywords.add('MES')
    
    return keywords


def generate_patterns(product: Product) -> List[str]:
    """
    Generate all reasonable search patterns for a product.
    
    Patterns are regex strings that might match this product in plan text.
    Order matters: more specific patterns first for better confidence scoring.
    """
    patterns = []
    size = product.size.replace('"', '').replace("'", "").strip()
    config = product.configuration
    
    # Round RCP
    if product.category == 'RCP_Price_List' and 'Round' in product.product_type:
        patterns.extend(_generate_rcp_patterns(size, config))
    
    # Elliptical RCP
    elif product.category == 'RCP_Price_List' and 'Elliptical' in product.product_type:
        patterns.extend(_generate_elliptical_patterns(size, config))
    
    # End Sections
    elif product.category == 'End_Sections':
        patterns.extend(_generate_endwall_patterns(product, size, config))
    
    # MES Pipe Grates
    elif product.category == 'MES_Pipe_Grates':
        patterns.extend(_generate_mes_patterns(size, config))
    
    return patterns


def _generate_rcp_patterns(size: str, config: str) -> List[str]:
    """Generate patterns for Round Reinforced Concrete Pipe."""
    patterns = []
    
    # RCP variants including OCR errors
    rcp_variants = ['RCP', 'RGP', 'PCP', 'RCE', r'R\.?C\.?P\.?']
    
    # Normalize class notation
    class_match = re.match(r'CL\s*([IVX]+)', config)
    if class_match:
        class_num = class_match.group(1)
        class_variants = [
            f"CL\\s*{class_num}",
            f"CLASS\\s*{class_num}",
            f"C-{class_num}",
        ]
        
        # Most specific: size + RCP + class
        for rcp in rcp_variants:
            for cv in class_variants:
                # Standard formats
                patterns.append(rf'{size}["\']?\s*(?:INCH\s+)?{rcp}\s*,?\s*{cv}')
                patterns.append(rf'{size}["\']?\s*{cv}\s+{rcp}')
                patterns.append(rf'{rcp}\s+{size}["\']?\s*,?\s*{cv}')
                patterns.append(rf'{size}\s*(?:INCH|IN\.?)\s*{rcp}\s*,?\s*{cv}')
        
        # FDOT format with parentheses: (18") (RCP) (CL III)
        for cv in class_variants:
            patterns.append(rf'\({size}["\']?\)\s*\(RCP\)\s*\({cv}\)')
            patterns.append(rf'PIPE.*?\({size}["\']?\).*?\(RCP\).*?\({cv}\)')
            patterns.append(rf'CULVERT.*?{size}["\']?.*?RCP.*?{cv}')
        
        # Size + pipe type + class (for tabular formats)
        for cv in class_variants:
            patterns.append(rf'PIPE\s*[,\-]?\s*{size}["\']?\s*.*?RCP.*?{cv}')
            patterns.append(rf'{size}["\']?\s*(?:REINFORCED\s+)?(?:CONCRETE\s+)?PIPE\s+{cv}')
    
    # MES 4:1 pattern
    elif 'MES' in config:
        patterns.append(rf'{size}["\']?\s*(?:RCP\s+)?MES\s*4:1')
        patterns.append(rf'MES\s+{size}["\']?\s*(?:RCP)?')
        patterns.append(rf'{size}["\']?\s*MITERED\s+END\s+SECTION')
    
    # Flared End pattern
    elif 'Flared' in config:
        patterns.append(rf'{size}["\']?\s*(?:RCP\s+)?FLARED\s+END')
        patterns.append(rf'FLARED\s+END\s+(?:SECTION\s+)?{size}["\']?')
        patterns.append(rf'{size}["\']?\s*FES')  # Flared End Section abbreviation
    
    # Pipe Cradle pattern
    elif 'Cradle' in config:
        patterns.append(rf'{size}["\']?\s*PIPE\s+CRADLE')
        patterns.append(rf'PIPE\s+CRADLE\s+{size}["\']?')
        patterns.append(rf'CRADLE\s+{size}["\']?')
    
    # Generic RCP patterns (less specific - lower confidence)
    for rcp in rcp_variants:
        patterns.append(rf'{size}["\']?\s*{rcp}')
        patterns.append(rf'{rcp}\s+{size}["\']?')
    
    # INCH format (county/municipal)
    patterns.append(rf'{size}\s*(?:INCH|IN\.?)\s*RCP')
    patterns.append(rf'{size}["\']?\s+REINFORCED\s+CONCRETE\s+PIPE')
    
    return patterns


def _generate_elliptical_patterns(size: str, config: str) -> List[str]:
    """Generate patterns for Elliptical RCP (e.g., 14"X23")."""
    patterns = []
    
    # Parse elliptical size (e.g., "14X23" or "14"X23"")
    size_clean = size.replace('"', '').replace("'", "")
    match = re.match(r'(\d+)\s*[Xx]\s*(\d+)', size_clean)
    if not match:
        return patterns
    
    dim1, dim2 = match.groups()
    
    # Size patterns - various separators
    size_patterns = [
        rf'{dim1}["\']?\s*[Xx×]\s*{dim2}["\']?',
        rf'{dim1}\s*BY\s*{dim2}',
        rf'{dim1}-{dim2}',
    ]
    
    # Class variants
    class_match = re.match(r'CL\s*([IVX]+)', config)
    if class_match:
        class_num = class_match.group(1)
        class_variants = [f"CL\\s*{class_num}", f"CLASS\\s*{class_num}"]
        
        for sp in size_patterns:
            for cv in class_variants:
                patterns.append(rf'{sp}\s*ELLIPTICAL.*?{cv}')
                patterns.append(rf'ELLIPTICAL.*?{sp}.*?{cv}')
                patterns.append(rf'{sp}.*?RCP.*?{cv}')
    
    # MES pattern
    elif 'MES' in config:
        for sp in size_patterns:
            patterns.append(rf'{sp}\s*(?:ELLIPTICAL\s+)?MES')
            patterns.append(rf'MES\s+{sp}')
    
    # Flared End pattern
    elif 'Flared' in config:
        for sp in size_patterns:
            patterns.append(rf'{sp}\s*FLARED\s+END')
            patterns.append(rf'FLARED\s+END.*?{sp}')
    
    # Generic elliptical
    for sp in size_patterns:
        patterns.append(rf'{sp}\s*ELLIPTICAL')
        patterns.append(rf'ELLIPTICAL\s*{sp}')
        patterns.append(rf'{sp}\s*(?:E\.?R\.?C\.?P\.?|ERCP)')
    
    return patterns


def _generate_endwall_patterns(product: Product, size: str, config: str) -> List[str]:
    """Generate patterns for End Sections (endwalls)."""
    patterns = []
    product_type = product.product_type
    
    if 'Straight' in product_type:
        # Configuration-specific (most specific)
        if 'Single' in config:
            patterns.append(rf'{size}["\']?\s*SINGLE\s+ENDWALL')
            patterns.append(rf'SINGLE\s+ENDWALL\s+{size}["\']?')
            patterns.append(rf'{size}["\']?\s*(?:STRAIGHT\s+)?ENDWALL\s*[-,]?\s*SINGLE')
        elif 'Double' in config:
            patterns.append(rf'{size}["\']?\s*DOUBLE\s+ENDWALL')
            patterns.append(rf'DOUBLE\s+ENDWALL\s+{size}["\']?')
            patterns.append(rf'{size}["\']?\s*(?:STRAIGHT\s+)?ENDWALL\s*[-,]?\s*DOUBLE')
        elif 'Triple' in config:
            patterns.append(rf'{size}["\']?\s*TRIPLE\s+ENDWALL')
            patterns.append(rf'TRIPLE\s+ENDWALL\s+{size}["\']?')
            patterns.append(rf'{size}["\']?\s*(?:STRAIGHT\s+)?ENDWALL\s*[-,]?\s*TRIPLE')
        elif 'Quad' in config:
            patterns.append(rf'{size}["\']?\s*QUAD(?:RUPLE)?\s+ENDWALL')
            patterns.append(rf'QUAD(?:RUPLE)?\s+ENDWALL\s+{size}["\']?')
        
        # Generic straight endwall
        patterns.append(rf'{size}["\']?\s*STRAIGHT\s+ENDWALL')
        patterns.append(rf'STRAIGHT\s+ENDWALL\s+{size}["\']?')
        patterns.append(rf'{size}["\']?\s*ENDWALL')
        patterns.append(rf'ENDWALL\s+{size}["\']?')
        patterns.append(rf'{size}["\']?\s*END\s+WALL')
    
    elif 'Winged' in product_type:
        if '45' in config:
            patterns.append(rf'{size}["\']?\s*(?:WINGED?\s+)?ENDWALL\s*[-,]?\s*45\s*(?:DEG(?:REE)?)?')
            patterns.append(rf'45\s*(?:DEG(?:REE)?)?\s+WINGED?\s+ENDWALL\s+{size}["\']?')
        elif 'U-Type' in config or 'U TYPE' in config:
            patterns.append(rf'{size}["\']?\s*WINGED?\s+ENDWALL\s*[-,]?\s*U[-\s]?TYPE')
            patterns.append(rf'U[-\s]?TYPE\s+WINGED?\s+ENDWALL\s+{size}["\']?')
        
        patterns.append(rf'{size}["\']?\s*WINGED?\s+ENDWALL')
        patterns.append(rf'WINGED?\s+ENDWALL\s+{size}["\']?')
        patterns.append(rf'{size}["\']?\s*WING\s+WALL')
    
    elif 'U-Type' in product_type:
        # Parse slope ratio if present
        slope_match = re.match(r'(\d+):1', config)
        if slope_match:
            ratio = slope_match.group(1)
            patterns.append(rf'{size}["\']?\s*U[-\s]?TYPE\s+(?:ENDWALL\s+)?{ratio}:1')
            patterns.append(rf'{ratio}:1\s+U[-\s]?TYPE\s+(?:ENDWALL\s+)?{size}["\']?')
            patterns.append(rf'{size}["\']?\s*{ratio}:1\s+U[-\s]?TYPE')
            
            # With grate/baffle variants
            if 'Grate' in config and 'Baffle' in config:
                patterns.append(rf'{size}["\']?\s*U[-\s]?TYPE.*?{ratio}:1.*?GRATE.*?BAFFLE')
            elif 'Grate' in config:
                patterns.append(rf'{size}["\']?\s*U[-\s]?TYPE.*?{ratio}:1.*?GRATE')
            elif 'Baffle' in config:
                patterns.append(rf'{size}["\']?\s*U[-\s]?TYPE.*?{ratio}:1.*?BAFFLE')
        
        # Basic U-Type
        patterns.append(rf'{size}["\']?\s*U[-\s]?TYPE\s+ENDWALL')
        patterns.append(rf'U[-\s]?TYPE\s+ENDWALL\s+{size}["\']?')
        patterns.append(rf'{size}["\']?\s*U[-\s]?TYPE')
    
    return patterns


def _generate_mes_patterns(size: str, config: str) -> List[str]:
    """Generate patterns for MES Pipe Grates (galvanized steel)."""
    patterns = []
    
    # Check if elliptical size (e.g., "12 X 18") or round (e.g., "15"")
    size_clean = size.replace('"', '').replace("'", "").strip()
    
    # Parse run configuration
    run_type = ""
    if 'Single' in config:
        run_type = "SINGLE"
    elif 'Double' in config:
        run_type = "DOUBLE"
    elif 'Triple' in config:
        run_type = "TRIPLE"
    
    has_frame = "Frame" in config and "No Frame" not in config
    
    # Elliptical size
    if 'X' in size_clean.upper():
        match = re.match(r'(\d+)\s*[Xx]\s*(\d+)', size_clean)
        if match:
            dim1, dim2 = match.groups()
            size_pat = rf'{dim1}["\']?\s*[Xx×]\s*{dim2}["\']?'
            
            if run_type:
                patterns.append(rf'{size_pat}\s*(?:4:1\s+)?MES\s+{run_type}')
                patterns.append(rf'{run_type}\s+(?:RUN\s+)?(?:4:1\s+)?MES\s+{size_pat}')
                patterns.append(rf'MES\s+{size_pat}\s+{run_type}')
                
                if has_frame:
                    patterns.append(rf'{size_pat}.*?MES.*?{run_type}.*?(?:WITH\s+)?FRAME')
                else:
                    patterns.append(rf'{size_pat}.*?MES.*?{run_type}.*?NO\s+FRAME')
            
            patterns.append(rf'{size_pat}\s*(?:4:1\s+)?MES')
            patterns.append(rf'MES\s+(?:4:1\s+)?{size_pat}')
    else:
        # Round size
        if run_type:
            patterns.append(rf'{size_clean}["\']?\s*(?:4:1\s+)?MES\s+{run_type}')
            patterns.append(rf'{run_type}\s+(?:RUN\s+)?(?:4:1\s+)?MES\s+{size_clean}["\']?')
            patterns.append(rf'MES\s+{size_clean}["\']?\s+{run_type}')
            
            if has_frame:
                patterns.append(rf'{size_clean}["\']?.*?MES.*?{run_type}.*?(?:WITH\s+)?FRAME')
            else:
                patterns.append(rf'{size_clean}["\']?.*?MES.*?{run_type}.*?NO\s+FRAME')
        
        patterns.append(rf'{size_clean}["\']?\s*(?:4:1\s+)?MES')
        patterns.append(rf'MES\s+(?:4:1\s+)?{size_clean}["\']?')
        patterns.append(rf'{size_clean}["\']?\s*MITERED\s+END\s+SECTION')
    
    # Galvanized steel variant
    patterns.append(rf'GALVANIZED.*?MES.*?{size_clean}')
    
    return patterns


# Default catalog path (relative to this file)
DEFAULT_CATALOG_PATH = Path(__file__).parent.parent / 'references' / 'fl_2025_prices.csv'


def load_catalog(path: Optional[Path] = None) -> ProductCatalog:
    """Load the default product catalog."""
    if path is None:
        path = DEFAULT_CATALOG_PATH
    return ProductCatalog.from_csv(path)
