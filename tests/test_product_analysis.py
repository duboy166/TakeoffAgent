"""
Unit tests for product-aware hybrid extraction functions.

Tests:
- PageProductAnalysis dataclass
- analyze_page_product_quality() function
- generate_material_summary() function
- Shared constants consistency
"""

import pytest
from tools.analyze_takeoff import (
    PageProductAnalysis,
    analyze_page_product_quality,
    generate_material_summary,
    PIPE_MATERIALS,
    STRUCTURE_TYPE_KEYWORDS,
    PRODUCT_KEYWORDS,
)


class TestPageProductAnalysis:
    """Tests for the PageProductAnalysis dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        analysis = PageProductAnalysis(page_num=1)
        assert analysis.page_num == 1
        assert analysis.product_keywords_found == 0
        assert analysis.complete_items_found == 0
        assert analysis.incomplete_items_found == 0
        assert analysis.callouts_without_quantity == 0
        assert analysis.has_complex_tables is False
        assert analysis.has_pipe_schedule is False
        assert analysis.needs_vision is False
        assert analysis.reasons == []

    def test_custom_values(self):
        """Test that custom values can be set."""
        analysis = PageProductAnalysis(
            page_num=5,
            product_keywords_found=10,
            complete_items_found=3,
            needs_vision=True,
            reasons=["test_reason"]
        )
        assert analysis.page_num == 5
        assert analysis.product_keywords_found == 10
        assert analysis.complete_items_found == 3
        assert analysis.needs_vision is True
        assert analysis.reasons == ["test_reason"]


class TestAnalyzePageProductQuality:
    """Tests for the analyze_page_product_quality function."""

    def test_empty_text(self):
        """Test analysis of empty text."""
        result = analyze_page_product_quality("", page_num=1)
        assert result.page_num == 1
        assert result.product_keywords_found == 0
        assert result.needs_vision is False

    def test_detects_pipe_keywords(self):
        """Test that pipe material keywords are detected."""
        text = "RCP 18\" PVC HDPE CMP"
        result = analyze_page_product_quality(text, page_num=1)
        assert result.product_keywords_found >= 4

    def test_detects_structure_keywords(self):
        """Test that structure keywords are detected."""
        text = "INLET MANHOLE CATCH BASIN ENDWALL"
        result = analyze_page_product_quality(text, page_num=1)
        assert result.product_keywords_found >= 4

    def test_complete_item_detection(self):
        """Test detection of complete items with size and quantity."""
        text = '51 LF 18" RCP CLASS III'
        result = analyze_page_product_quality(text, page_num=1)
        assert result.complete_items_found >= 1

    def test_fdot_format_complete_item(self):
        """Test detection of FDOT format complete items."""
        text = "430-175-118 PIPE CULVERT 18\" LF 98"
        result = analyze_page_product_quality(text, page_num=1)
        assert result.complete_items_found >= 1

    def test_callout_without_quantity(self):
        """Test detection of callouts without quantities."""
        text = "RCP 24\nRCP 18\nPVC 12"
        result = analyze_page_product_quality(text, page_num=1)
        assert result.callouts_without_quantity >= 1

    def test_flags_vision_for_incomplete_items(self):
        """Test that Vision is flagged when products detected but incomplete."""
        text = "RCP INLET MANHOLE PIPE"  # Keywords but no complete items
        result = analyze_page_product_quality(text, page_num=1)
        assert result.needs_vision is True
        assert any("product_keywords_found" in r for r in result.reasons)

    def test_flags_vision_for_callouts(self):
        """Test that Vision is flagged for callouts without quantities."""
        text = "RCP 24 RCP 18 HDPE 15"  # Callouts without LF amounts
        result = analyze_page_product_quality(text, page_num=1)
        assert result.callouts_without_quantity > 0
        assert result.needs_vision is True

    def test_pipe_schedule_detection(self):
        """Test detection of pipe schedule tables."""
        text = """
        PIPE SCHEDULE
        SIZE    QTY    LENGTH
        18"     5      100 LF
        """
        result = analyze_page_product_quality(text, page_num=1)
        assert result.has_pipe_schedule is True
        assert result.has_complex_tables is True
        assert result.needs_vision is True

    def test_no_vision_for_complete_items(self):
        """Test that Vision is not flagged when items are complete."""
        text = '100 LF 18" RCP CLASS III @ 0.5%'
        result = analyze_page_product_quality(text, page_num=1)
        # Complete item found, so should not necessarily need vision
        assert result.complete_items_found >= 1


class TestGenerateMaterialSummary:
    """Tests for the generate_material_summary function."""

    def test_empty_items(self):
        """Test with empty item list."""
        result = generate_material_summary([])
        assert result['pipe_summary'] == {}
        assert result['structure_summary'] == {}
        assert result['totals']['total_pipe_lf'] == 0
        assert result['totals']['total_structures'] == 0

    def test_pipe_aggregation(self):
        """Test that pipes are aggregated by size and material."""
        items = [
            {'description': '18" RCP CLASS III', 'quantity': 100, 'unit': 'LF'},
            {'description': '18" RCP CLASS III', 'quantity': 50, 'unit': 'LF'},
            {'description': '24" RCP CLASS V', 'quantity': 75, 'unit': 'LF'},
        ]
        result = generate_material_summary(items)

        assert '18"_RCP' in result['pipe_summary']
        assert result['pipe_summary']['18"_RCP']['total_lf'] == 150
        assert result['pipe_summary']['18"_RCP']['count'] == 2

        assert '24"_RCP' in result['pipe_summary']
        assert result['pipe_summary']['24"_RCP']['total_lf'] == 75
        assert result['pipe_summary']['24"_RCP']['count'] == 1

        assert result['totals']['total_pipe_lf'] == 225
        assert result['totals']['total_pipe_sizes'] == 2

    def test_structure_aggregation(self):
        """Test that structures are aggregated by type."""
        items = [
            {'description': 'INLET TYPE D', 'quantity': 3, 'unit': 'EA'},
            {'description': 'INLET TYPE E', 'quantity': 2, 'unit': 'EA'},
            {'description': 'MANHOLE #1', 'quantity': 1, 'unit': 'EA'},
        ]
        result = generate_material_summary(items)

        assert 'INLET' in result['structure_summary']
        assert result['structure_summary']['INLET']['count'] == 5

        assert 'MANHOLE' in result['structure_summary']
        assert result['structure_summary']['MANHOLE']['count'] == 1

        assert result['totals']['total_structures'] == 6
        assert result['totals']['total_structure_types'] == 2

    def test_mixed_items(self):
        """Test with a mix of pipes and structures."""
        items = [
            {'description': '18" RCP CLASS III', 'quantity': 100, 'unit': 'LF'},
            {'description': 'INLET TYPE D', 'quantity': 3, 'unit': 'EA'},
            {'description': '18" STRAIGHT ENDWALL', 'quantity': 2, 'unit': 'EA'},
        ]
        result = generate_material_summary(items)

        assert result['totals']['total_pipe_lf'] == 100
        assert result['totals']['total_structures'] == 5  # 3 inlets + 2 endwalls

    def test_ignores_non_lf_pipes(self):
        """Test that pipes with non-LF units are not counted in pipe summary."""
        items = [
            {'description': '18" RCP', 'quantity': 5, 'unit': 'EA'},  # EA not LF
        ]
        result = generate_material_summary(items)
        # Should be counted as structure (EA), not pipe
        assert result['totals']['total_pipe_lf'] == 0

    def test_handles_none_quantity(self):
        """Test handling of None quantity values."""
        items = [
            {'description': 'INLET TYPE D', 'quantity': None, 'unit': 'EA'},
        ]
        result = generate_material_summary(items)
        assert result['structure_summary']['INLET']['count'] == 1

    def test_handles_zero_quantity(self):
        """Test handling of zero quantity values."""
        items = [
            {'description': '18" RCP', 'quantity': 0, 'unit': 'LF'},
        ]
        result = generate_material_summary(items)
        assert result['totals']['total_pipe_lf'] == 0


class TestSharedConstants:
    """Tests for shared constant consistency."""

    def test_pipe_materials_in_product_keywords(self):
        """Test that all pipe materials are in product keywords."""
        for mat in PIPE_MATERIALS:
            assert mat in PRODUCT_KEYWORDS, f"{mat} not in PRODUCT_KEYWORDS"

    def test_structure_types_in_product_keywords(self):
        """Test that all structure types are in product keywords."""
        for struct in STRUCTURE_TYPE_KEYWORDS:
            assert struct in PRODUCT_KEYWORDS, f"{struct} not in PRODUCT_KEYWORDS"

    def test_product_keywords_not_empty(self):
        """Test that product keywords set is not empty."""
        assert len(PRODUCT_KEYWORDS) > 0

    def test_pipe_materials_uppercase(self):
        """Test that all pipe materials are uppercase."""
        for mat in PIPE_MATERIALS:
            assert mat == mat.upper(), f"{mat} is not uppercase"

    def test_structure_types_uppercase(self):
        """Test that all structure types are uppercase."""
        for struct in STRUCTURE_TYPE_KEYWORDS:
            assert struct == struct.upper(), f"{struct} is not uppercase"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
