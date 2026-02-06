"""
Node 4: Report Generation
Generates JSON and CSV takeoff reports from processed data.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from ..state import TakeoffState

logger = logging.getLogger(__name__)


def _generate_csv_report(
    pay_items: List[Dict],
    output_path: Path,
    filename_stem: str,
    drainage_structures: List[Dict] = None,
    ai_matched_items: List[Dict] = None,
    items_for_review: List[Dict] = None,
    material_summary: Dict = None
) -> str:
    """
    Generate CSV report from pay items and drainage structures.

    Args:
        pay_items: List of pay item dictionaries
        output_path: Output directory path
        filename_stem: Base filename (without extension)
        drainage_structures: Optional list of drainage structure dictionaries
        ai_matched_items: Optional list of AI-matched items
        items_for_review: Optional list of items needing review

    Returns:
        Path to generated CSV file
    """
    csv_filename = f"{filename_stem}_takeoff.csv"
    csv_path = output_path / csv_filename

    # Define CSV columns (extended for agentic features)
    fieldnames = [
        'Pay Item No',
        'Description',
        'Unit',
        'Quantity',
        'Unit Price',
        'Line Cost',
        'Page',
        'Source',
        'Confidence',
        'Matched',
        'Match Source',
        'AI Match Reason'
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in pay_items:
            # Extract page from source_location if available
            source_loc = item.get('source_location', {})
            page = source_loc.get('page', '') if source_loc else ''

            writer.writerow({
                'Pay Item No': item.get('pay_item_no', ''),
                'Description': item.get('description', ''),
                'Unit': item.get('unit', ''),
                'Quantity': item.get('quantity', 0),
                'Unit Price': item.get('unit_price', '') if item.get('unit_price') else '',
                'Line Cost': item.get('line_cost', '') if item.get('line_cost') else '',
                'Page': page,
                'Source': item.get('source', ''),
                'Confidence': item.get('confidence', ''),
                'Matched': 'Yes' if item.get('matched') else 'No',
                'Match Source': item.get('match_source', 'rules') if item.get('matched') else '',
                'AI Match Reason': item.get('ai_match_reason', '')
            })

    # Append drainage structures section if there are items
    if drainage_structures:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            f.write('\n')
            f.write('DRAINAGE STRUCTURES\n')

            struct_fieldnames = ['Type', 'ID', 'Size', 'Description', 'Quantity', 'Page', 'Confidence']
            writer = csv.DictWriter(f, fieldnames=struct_fieldnames)
            writer.writeheader()

            for struct in drainage_structures:
                writer.writerow({
                    'Type': struct.get('type', ''),
                    'ID': struct.get('id', ''),
                    'Size': struct.get('size', ''),
                    'Description': struct.get('description', ''),
                    'Quantity': struct.get('quantity', 1),
                    'Page': struct.get('page', ''),
                    'Confidence': struct.get('confidence', '')
                })

    # Append needs_review section if there are items
    if items_for_review:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            f.write('\n')
            f.write('ITEMS NEEDING REVIEW\n')

            review_fieldnames = ['Description', 'Quantity', 'Unit', 'Page', 'Reason', 'Region Hint']
            writer = csv.DictWriter(f, fieldnames=review_fieldnames)
            writer.writeheader()

            for item in items_for_review:
                writer.writerow({
                    'Description': item.get('description', ''),
                    'Quantity': item.get('quantity', 0),
                    'Unit': item.get('unit', ''),
                    'Page': item.get('page', ''),
                    'Reason': item.get('reason', ''),
                    'Region Hint': item.get('region_hint', '')
                })

    # Append MATERIAL TAKEOFF SUMMARY section
    if material_summary:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            f.write('\n')
            f.write('MATERIAL TAKEOFF SUMMARY\n')
            f.write('\n')

            # Pipe Summary
            pipe_summary = material_summary.get('pipe_summary', {})
            if pipe_summary:
                f.write('PIPE SUMMARY\n')
                pipe_fieldnames = ['Size/Material', 'Total LF', 'Count']
                writer = csv.DictWriter(f, fieldnames=pipe_fieldnames)
                writer.writeheader()

                # Sort by size for consistent output
                for key in sorted(pipe_summary.keys()):
                    data = pipe_summary[key]
                    writer.writerow({
                        'Size/Material': key,
                        'Total LF': data.get('total_lf', 0),
                        'Count': data.get('count', 0)
                    })

                f.write('\n')

            # Structure Summary
            structure_summary = material_summary.get('structure_summary', {})
            if structure_summary:
                f.write('STRUCTURE SUMMARY\n')
                struct_fieldnames = ['Type', 'Count']
                writer = csv.DictWriter(f, fieldnames=struct_fieldnames)
                writer.writeheader()

                for struct_type, data in sorted(structure_summary.items()):
                    writer.writerow({
                        'Type': struct_type,
                        'Count': data.get('count', 0)
                    })

                f.write('\n')

            # Grand Totals
            totals = material_summary.get('totals', {})
            if totals:
                f.write('GRAND TOTALS\n')
                totals_fieldnames = ['Metric', 'Value']
                writer = csv.DictWriter(f, fieldnames=totals_fieldnames)
                writer.writeheader()

                writer.writerow({'Metric': 'Total Pipe LF', 'Value': totals.get('total_pipe_lf', 0)})
                writer.writerow({'Metric': 'Total Pipe Sizes', 'Value': totals.get('total_pipe_sizes', 0)})
                writer.writerow({'Metric': 'Total Structures', 'Value': totals.get('total_structures', 0)})
                writer.writerow({'Metric': 'Total Structure Types', 'Value': totals.get('total_structure_types', 0)})

    return str(csv_path)


def generate_report_node(state: TakeoffState) -> Dict[str, Any]:
    """
    Generate JSON and CSV takeoff reports.

    Steps:
    1. Load processed data from state
    2. Build JSON report structure
    3. Save JSON and CSV reports to output directory
    4. Update batch totals

    Args:
        state: Current workflow state

    Returns:
        State updates with report_path, csv_path, updated totals, or last_error
    """
    from tools.analyze_takeoff import generate_material_summary

    current_file = state.get("current_file", "")
    output_path = state.get("output_path", "")
    priced_items = state.get("priced_items") or []
    drainage_structures = state.get("drainage_structures") or []
    project_info = state.get("project_info", {})
    extraction_method = state.get("extraction_method", "unknown")
    page_count = state.get("page_count", 0)

    # Agentic features data (use 'or' to handle None values)
    ai_matched_items = state.get("ai_matched_items") or []
    validation_issues = state.get("validation_issues") or []
    items_for_review = state.get("items_for_review") or []
    items_corrected = state.get("items_corrected") or 0
    validation_confidence = state.get("validation_confidence") or 1.0
    recommended_extraction = state.get("recommended_extraction") or ""
    analysis_notes = state.get("analysis_notes") or []
    pages_product_analysis = state.get("pages_product_analysis") or []

    # Step 1: Validate inputs
    if not output_path:
        logger.error("No output path specified")
        return {"last_error": "No output path specified"}

    if not current_file:
        logger.error("No current file in state")
        return {"last_error": "No current file specified"}

    # Create output directory if needed
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate summary stats
    matched_count = sum(1 for item in priced_items if item.get("matched"))
    ai_matched_count = sum(1 for item in priced_items if item.get("match_source") == "ai")
    rule_matched_count = matched_count - ai_matched_count
    total_cost = sum(item.get("line_cost", 0) or 0 for item in priced_items)

    # Generate material takeoff summary
    material_summary = generate_material_summary(priced_items)
    totals = material_summary.get('totals', {})
    logger.info(f"Material summary: {totals.get('total_pipe_lf', 0)} LF pipe, {totals.get('total_structures', 0)} structures")

    logger.info(f"Generating report: {len(priced_items)} items, ${total_cost:,.2f} (AI matched: {ai_matched_count})")

    try:
        # Step 2: Build JSON report data
        extracted_text = state.get("extracted_text", "")
        
        # Collect all validation warnings from individual items
        item_validation_warnings = []
        items_with_warnings = 0
        for idx, item in enumerate(priced_items):
            item_warnings = item.get("validation_warnings", [])
            if item_warnings:
                items_with_warnings += 1
                for w in item_warnings:
                    w_copy = w.copy()
                    w_copy["item_index"] = idx
                    w_copy["pay_item_no"] = item.get("pay_item_no", "")
                    item_validation_warnings.append(w_copy)
        
        # Combine with validation_issues from state
        all_validation_warnings = {
            "from_extraction": validation_issues,  # From validate_items_node
            "from_items": item_validation_warnings,  # From validation_gates during extraction/pricing
            "summary": {
                "total_warnings": len(validation_issues) + len(item_validation_warnings),
                "items_with_warnings": items_with_warnings,
            }
        }

        report_data = {
            "project_info": project_info,
            "pay_items": priced_items,
            "drainage_structures": drainage_structures,
            "material_summary": material_summary,
            "ai_matched_items": ai_matched_items,
            "validation_warnings": all_validation_warnings,
            "needs_review": items_for_review,
            "pages_product_analysis": pages_product_analysis,
            "summary": {
                "pages_analyzed": page_count,
                "total_items": len(priced_items),
                "matched_items": matched_count,
                "rule_matched_items": rule_matched_count,
                "ai_matched_items": ai_matched_count,
                "items_corrected": items_corrected,
                "items_for_review": len(items_for_review),
                "validation_confidence": validation_confidence,
                "total_cost": total_cost,
                "total_pipe_lf": totals.get('total_pipe_lf', 0),
                "total_structures": totals.get('total_structures', 0)
            },
            "extraction_info": {
                "method": extraction_method,
                "recommended_method": recommended_extraction,
                "analysis_notes": analysis_notes
            },
            "extracted_text": extracted_text,  # Embedded for debugging
            "notes": [
                f"Extracted via {extraction_method}",
                f"Source: {Path(current_file).name}",
                f"Pages: {page_count}",
                f"AI matched: {ai_matched_count} items",
                f"Needs review: {len(items_for_review)} items",
                f"Material: {totals.get('total_pipe_lf', 0)} LF pipe, {totals.get('total_structures', 0)} structures"
            ]
        }

        # Step 3: Save JSON report
        filename_stem = Path(current_file).stem
        report_filename = f"{filename_stem}_takeoff.json"
        report_path = output_dir / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"JSON report saved: {report_path}")

        # Step 3b: Save CSV report
        csv_path = _generate_csv_report(
            priced_items,
            output_dir,
            filename_stem,
            drainage_structures=drainage_structures,
            ai_matched_items=ai_matched_items,
            items_for_review=items_for_review,
            material_summary=material_summary
        )
        logger.info(f"CSV report saved: {csv_path}")

        # Step 4: Return updates including batch totals
        # Update running totals directly (for single file case where advance_file is skipped)
        current_total_estimate = state.get("total_estimate", 0)
        current_total_pay_items = state.get("total_pay_items", 0)
        current_total_matched_items = state.get("total_matched_items", 0)
        current_total_pages = state.get("total_pages", 0)
        files_completed = state.get("files_completed") or []

        return {
            "report_path": str(report_path),
            "csv_path": csv_path,
            "last_error": None,
            # Store material summary in state
            "material_summary": material_summary,
            # Update running totals directly
            "total_estimate": current_total_estimate + total_cost,
            "total_pay_items": current_total_pay_items + len(priced_items),
            "total_matched_items": current_total_matched_items + matched_count,
            "total_pages": current_total_pages + page_count,
            # Add to completed list (for single file case where advance_file is skipped)
            "files_completed": files_completed + [current_file],
            # Also store file result for advance_to_next_file (multi-file case)
            "_file_result": {
                "filename": Path(current_file).name,
                "filepath": current_file,
                "success": True,
                "page_count": page_count,
                "pay_items_count": len(priced_items),
                "matched_items_count": matched_count,
                "estimated_total": total_cost,
                "extraction_method": extraction_method,
                "report_path": str(report_path),
                "csv_path": csv_path,
                "errors": [],
                "material_summary": material_summary
            }
        }

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {"last_error": f"Report generation failed: {str(e)}"}
