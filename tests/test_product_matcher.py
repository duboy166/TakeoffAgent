"""
Tests for Product Catalog and Product Matcher

Tests pattern generation, text matching, and quantity extraction.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.product_catalog import (
    Product, ProductCatalog, load_catalog,
    generate_patterns, generate_keywords, normalize_size,
    _generate_rcp_patterns, _generate_endwall_patterns
)
from tools.product_matcher import (
    find_products_in_text, search_with_quantity, Match,
    extract_quantity_with_unit, parse_number, is_valid_quantity,
    deduplicate_matches, is_summary_page
)


class TestProductCatalog:
    """Tests for product catalog loading and pattern generation."""
    
    @pytest.fixture
    def catalog(self):
        """Load the actual catalog for testing."""
        return load_catalog()
    
    def test_catalog_loads(self, catalog):
        """Catalog should load products from CSV."""
        assert len(catalog) > 0
        assert len(catalog.products) > 100  # We have ~300 products
    
    def test_catalog_has_categories(self, catalog):
        """Catalog should organize products by category."""
        assert 'RCP_Price_List' in catalog.by_category
        assert 'End_Sections' in catalog.by_category
        assert 'MES_Pipe_Grates' in catalog.by_category
    
    def test_catalog_indexes_sizes(self, catalog):
        """Catalog should index products by size."""
        assert '18' in catalog.by_size
        assert '24' in catalog.by_size
        # Elliptical sizes
        assert '14X23' in catalog.by_size or '14 X 23' in catalog.by_size.keys()
    
    def test_product_has_patterns(self, catalog):
        """Each product should have search patterns generated."""
        for product in catalog.products[:10]:
            assert len(product.search_patterns) > 0, f"{product.id} has no patterns"
    
    def test_product_has_keywords(self, catalog):
        """Each product should have keywords for quick filtering."""
        for product in catalog.products[:10]:
            assert len(product.keywords) > 0, f"{product.id} has no keywords"


class TestPatternGeneration:
    """Tests for search pattern generation."""
    
    def test_rcp_patterns(self):
        """RCP should generate multiple pattern variants."""
        patterns = _generate_rcp_patterns("18", "CL III")
        
        # Should have patterns for different formats
        assert any('18' in p and 'RCP' in p for p in patterns)
        assert any('CL' in p or 'CLASS' in p for p in patterns)
        
        # Should have OCR error variants
        assert any('RGP' in p for p in patterns), "Missing RGP (OCR error) variant"
    
    def test_rcp_mes_patterns(self):
        """RCP MES should generate MES-specific patterns."""
        patterns = _generate_rcp_patterns("24", "MES 4:1")
        
        assert any('MES' in p for p in patterns)
        assert any('24' in p for p in patterns)
    
    def test_rcp_flared_end_patterns(self):
        """RCP Flared End should generate FES patterns."""
        patterns = _generate_rcp_patterns("30", "Flared End")
        
        assert any('FLARED' in p for p in patterns)
        assert any('FES' in p for p in patterns)
    
    def test_endwall_patterns(self):
        """Endwall should generate appropriate patterns."""
        product = Product(
            id="test",
            category="End_Sections",
            product_type="Straight Endwalls",
            size="18",
            configuration="Single",
            price=1264.0,
            unit="EA",
            fdot_code="FDOT 430-030"
        )
        
        patterns = _generate_endwall_patterns(product, "18", "Single")
        
        assert any('ENDWALL' in p for p in patterns)
        assert any('SINGLE' in p for p in patterns)
        assert any('18' in p for p in patterns)
    
    def test_normalize_size(self):
        """Size normalization should handle quotes and spaces."""
        assert normalize_size('18"') == '18'
        assert normalize_size("18'") == '18'
        assert normalize_size('14"X23"') == '14X23'
        assert normalize_size('14 X 23') == '14X23'


class TestKeywordGeneration:
    """Tests for keyword generation."""
    
    def test_rcp_keywords(self):
        """RCP should require size as keyword (for broad matching)."""
        product = Product(
            id="test",
            category="RCP_Price_List",
            product_type="Round Reinforced Concrete Pipe",
            size="18",
            configuration="CL III",
            price=40.0,
            unit="LF",
            fdot_code="ASTM C-76"
        )
        
        keywords = generate_keywords(product)
        # Size is required, but not RCP (to allow OCR variants like RGP)
        assert '18' in keywords
    
    def test_elliptical_keywords(self):
        """Elliptical RCP should require size as keyword."""
        product = Product(
            id="test",
            category="RCP_Price_List",
            product_type="Elliptical Reinforced Concrete Pipe",
            size='14"X23"',
            configuration="CL III",
            price=62.0,
            unit="LF",
            fdot_code="ASTM C-507"
        )
        
        keywords = generate_keywords(product)
        # Size number is required
        assert '14' in keywords
    
    def test_endwall_keywords(self):
        """Endwall should require size as keyword."""
        product = Product(
            id="test",
            category="End_Sections",
            product_type="Straight Endwalls",
            size="18",
            configuration="Single",
            price=1264.0,
            unit="EA",
            fdot_code="FDOT 430-030"
        )
        
        keywords = generate_keywords(product)
        # Size is required
        assert '18' in keywords
    
    def test_mes_keywords(self):
        """MES products should require MES keyword."""
        product = Product(
            id="test",
            category="MES_Pipe_Grates",
            product_type="Galvanized Steel 4:1 MES",
            size="18",
            configuration="Single Run With Frame",
            price=952.0,
            unit="EA",
            fdot_code=""
        )
        
        keywords = generate_keywords(product)
        assert '18' in keywords
        assert 'MES' in keywords


class TestProductMatching:
    """Tests for matching products against plan text."""
    
    @pytest.fixture
    def catalog(self):
        return load_catalog()
    
    def test_match_simple_rcp(self, catalog):
        """Should match simple RCP format."""
        text = "18\" RCP CLASS III - 450 LF"
        matches = find_products_in_text(text, catalog)
        
        assert len(matches) >= 1
        rcp_match = [m for m in matches if 'RCP' in m.product.category]
        assert len(rcp_match) >= 1
        assert rcp_match[0].quantity == 450
        assert rcp_match[0].unit == 'LF'
    
    def test_match_rcp_no_quotes(self, catalog):
        """Should match RCP without quotes."""
        text = "18 INCH RCP CL III 200 LF"
        matches = find_products_in_text(text, catalog)
        
        rcp_match = [m for m in matches if 'RCP' in m.product.category and '18' in m.product.size]
        assert len(rcp_match) >= 1
    
    def test_match_endwall(self, catalog):
        """Should match endwall format."""
        text = "STRAIGHT ENDWALL 18\" SINGLE - 2 EA"
        matches = find_products_in_text(text, catalog)
        
        endwall_match = [m for m in matches if 'End_Sections' in m.product.category]
        assert len(endwall_match) >= 1
        # Should find quantity
        assert any(m.quantity == 2 for m in endwall_match)
    
    def test_match_tabular_format(self, catalog):
        """Should match tabular format with dots."""
        text = """
        DRAINAGE STRUCTURES
        18" RCP CLASS III ........................ 450 LF
        24" RCP CLASS III ........................ 225 LF
        """
        matches = find_products_in_text(text, catalog)
        
        # Should find both pipes
        sizes_found = set()
        for m in matches:
            if 'RCP' in m.product.category:
                sizes_found.add(m.product.size.replace('"', ''))
        
        assert '18' in sizes_found
        assert '24' in sizes_found
    
    def test_match_ocr_error(self, catalog):
        """Should match common OCR errors like RGP instead of RCP."""
        text = "18\" RGP CL III - 300 LF"
        matches = find_products_in_text(text, catalog)
        
        # Should still find 18" RCP even with OCR error
        rcp_match = [m for m in matches if '18' in m.product.size and 'RCP' in m.product.category]
        assert len(rcp_match) >= 1
    
    def test_confidence_scoring(self, catalog):
        """Confidence should be higher for exact matches."""
        # Exact match with unit
        text1 = "18\" RCP CL III - 450 LF"
        matches1 = find_products_in_text(text1, catalog)
        
        # Vague match without unit
        text2 = "18 RCP somewhere 450"
        matches2 = find_products_in_text(text2, catalog)
        
        if matches1 and matches2:
            # Exact match should have higher confidence
            assert max(m.confidence for m in matches1) >= max(m.confidence for m in matches2)
    
    def test_multiple_products_same_text(self, catalog):
        """Should find multiple products in same text block."""
        text = """
        DRAINAGE:
        18" RCP CL III - 450 LF
        24" RCP CL III - 225 LF
        18" STRAIGHT ENDWALL SINGLE - 2 EA
        """
        matches = find_products_in_text(text, catalog)
        
        # Should find at least 3 products
        assert len(matches) >= 3


class TestQuantityExtraction:
    """Tests for quantity extraction from text."""
    
    def test_quantity_after_product(self):
        """Should extract quantity after product mention."""
        context = "18\" RCP CL III - 450 LF more text here"
        result = extract_quantity_with_unit(
            context, context.upper(), "LF", 0, 15
        )
        
        assert result is not None
        qty, conf, unit = result
        assert qty == 450
        assert unit == 'LF'
    
    def test_quantity_before_product(self):
        """Should extract quantity before product mention."""
        context = "450 LF of 18\" RCP CL III"
        result = extract_quantity_with_unit(
            context, context.upper(), "LF", 10, 25
        )
        
        assert result is not None
        qty, conf, unit = result
        assert qty == 450
    
    def test_quantity_with_comma(self):
        """Should handle quantities with commas."""
        context = "18\" RCP - 1,500 LF"
        result = extract_quantity_with_unit(
            context, context.upper(), "LF", 0, 10
        )
        
        assert result is not None
        qty, conf, unit = result
        assert qty == 1500
    
    def test_quantity_linear_feet(self):
        """Should recognize 'LINEAR FEET' as LF."""
        context = "18\" RCP - 450 LINEAR FEET"
        result = extract_quantity_with_unit(
            context, context.upper(), "LF", 0, 10
        )
        
        assert result is not None
        qty, conf, unit = result
        assert qty == 450
        assert unit == 'LF'
    
    def test_quantity_each(self):
        """Should recognize 'EACH' as EA."""
        context = "18\" ENDWALL - 2 EACH"
        result = extract_quantity_with_unit(
            context, context.upper(), "EA", 0, 12
        )
        
        assert result is not None
        qty, conf, unit = result
        assert qty == 2
        assert unit == 'EA'
    
    def test_parse_number_with_comma(self):
        """parse_number should handle commas."""
        assert parse_number("1,500") == 1500.0
        assert parse_number("1,234,567") == 1234567.0
        assert parse_number("450.5") == 450.5


class TestQuantityValidation:
    """Tests for quantity validation/sanity checks."""
    
    def test_reject_pipe_sizes_as_lf(self):
        """Should reject common pipe sizes as LF quantities."""
        # 18 is a pipe size, not likely a valid LF quantity on its own
        assert is_valid_quantity(18, "LF") is False
        assert is_valid_quantity(24, "LF") is False
        assert is_valid_quantity(36, "LF") is False
    
    def test_accept_reasonable_lf(self):
        """Should accept reasonable LF quantities."""
        assert is_valid_quantity(100, "LF") is True
        assert is_valid_quantity(450, "LF") is True
        assert is_valid_quantity(1500, "LF") is True
    
    def test_reject_large_pipe_sizes_as_ea(self):
        """Should reject large pipe sizes as EA quantities."""
        assert is_valid_quantity(48, "EA") is False
        assert is_valid_quantity(72, "EA") is False
    
    def test_accept_small_ea(self):
        """Should accept small EA quantities."""
        assert is_valid_quantity(2, "EA") is True
        assert is_valid_quantity(6, "EA") is True
        assert is_valid_quantity(10, "EA") is True


class TestDeduplication:
    """Tests for match deduplication."""
    
    def test_deduplicate_same_product(self):
        """Should keep highest confidence match for same product."""
        product = Product(
            id="test_product",
            category="RCP_Price_List",
            product_type="Round Reinforced Concrete Pipe",
            size="18",
            configuration="CL III",
            price=40.0,
            unit="LF",
            fdot_code="ASTM C-76"
        )
        
        matches = [
            Match(product, quantity=450, confidence=0.9, 
                  source_text="18\" RCP - 450 LF", pattern_used=".*", unit="LF"),
            Match(product, quantity=450, confidence=0.7, 
                  source_text="18 RCP 450", pattern_used=".*", unit="LF"),
        ]
        
        result = deduplicate_matches(matches)
        
        assert len(result) == 1
        assert result[0].confidence == 0.9
    
    def test_deduplicate_different_products(self):
        """Should keep matches for different products."""
        product1 = Product(
            id="product_18",
            category="RCP_Price_List",
            product_type="Round Reinforced Concrete Pipe",
            size="18",
            configuration="CL III",
            price=40.0,
            unit="LF",
            fdot_code="ASTM C-76"
        )
        product2 = Product(
            id="product_24",
            category="RCP_Price_List",
            product_type="Round Reinforced Concrete Pipe",
            size="24",
            configuration="CL III",
            price=62.0,
            unit="LF",
            fdot_code="ASTM C-76"
        )
        
        matches = [
            Match(product1, quantity=450, confidence=0.9, 
                  source_text="18\" RCP - 450 LF", pattern_used=".*", unit="LF"),
            Match(product2, quantity=225, confidence=0.85, 
                  source_text="24\" RCP - 225 LF", pattern_used=".*", unit="LF"),
        ]
        
        result = deduplicate_matches(matches)
        
        assert len(result) == 2


class TestSummaryPageDetection:
    """Tests for summary page detection."""
    
    def test_detect_summary_page(self):
        """Should detect summary page indicators."""
        text1 = "SUMMARY OF PAY ITEMS\n\nItem No. Description Qty Unit"
        assert is_summary_page(text1) is True
        
        text2 = "ESTIMATE OF QUANTITIES\n\nDRAINAGE STRUCTURES"
        assert is_summary_page(text2) is True
        
        text3 = "BID SCHEDULE\n\nPipe 18\" RCP - 450 LF"
        assert is_summary_page(text3) is True
    
    def test_not_summary_page(self):
        """Should not flag regular pages as summary."""
        text = "DRAINAGE PLAN SHEET 1\n\n18\" RCP shown"
        assert is_summary_page(text) is False


class TestRealWorldScenarios:
    """Tests with realistic plan text formats."""
    
    @pytest.fixture
    def catalog(self):
        return load_catalog()
    
    def test_fdot_format(self, catalog):
        """Should handle FDOT format text."""
        text = """
        PIPE CULVERT (ROUND) (18") (RCP) (CL III) ........... 450 LF
        PIPE CULVERT (ROUND) (24") (RCP) (CL III) ........... 225 LF
        """
        matches = find_products_in_text(text, catalog)
        
        assert len(matches) >= 2
    
    def test_county_format(self, catalog):
        """Should handle county/municipal format."""
        text = """
        STORM DRAINAGE
        
        18 INCH RCP, CLASS III
        QUANTITY: 450 LINEAR FEET
        
        24 INCH RCP, CLASS III  
        QUANTITY: 225 LINEAR FEET
        """
        matches = find_products_in_text(text, catalog)
        
        sizes = [m.product.size.replace('"', '') for m in matches if 'RCP' in m.product.category]
        assert '18' in sizes
        assert '24' in sizes
    
    def test_summary_table_format(self, catalog):
        """Should handle summary table format."""
        text = """
        SUMMARY OF PAY ITEMS
        
        ITEM NO.    DESCRIPTION                         QTY    UNIT
        430-174-118 18" RCP CL III                      450    LF
        430-174-124 24" RCP CL III                      225    LF
        430-030     18" STRAIGHT ENDWALL (SINGLE)       2      EA
        """
        matches = find_products_in_text(text, catalog)
        
        # Should find all three items
        assert len(matches) >= 3
        
        # Verify quantities
        quantities = {(m.product.size, m.quantity) for m in matches}
        assert any(q == 450 for _, q in quantities)
        assert any(q == 225 for _, q in quantities)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
